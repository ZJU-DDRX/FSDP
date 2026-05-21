#!/usr/bin/env python3
import numpy as np
from gp.Sparse_GP_function import sparse_gp_model_train, sparse_gp_model_predict
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import torch
import gpytorch

# First step, the incoming data should be in a reasonable range
def filter_by_range(x_incoming, y_incoming, y_min, y_max):
    """Filter incoming data to ensure y_incoming values are within the range of the existing target values."""
    mask = (y_incoming >= y_min) & (y_incoming <= y_max)
    x_filtered, y_filtered = x_incoming[mask], y_incoming[mask]
    # Check if there’s valid data after filtering
    has_valid_data = x_filtered.size > 0
    return x_filtered, y_filtered, has_valid_data

# Second step, the incoming data should be in the old gp 95% confidence interval only when the current dataset is higher than 2/3 of the target size
def filter_by_confidence_interval(y_incoming, gp_mean, gp_std, threshold=1.96):
    """Filter incoming data to lie within the GP's confidence interval."""
    within_confidence = (y_incoming >= (gp_mean - threshold * gp_std)) & \
                        (y_incoming <= (gp_mean + threshold * gp_std))
    return within_confidence

# Third step, the variance of the new data should be higher than the mean variance of the old gp
def filter_by_variance(x_incoming, y_incoming, new_data_variance, mean_var_train):
    """Retain data points with variance higher than the mean variance of the training data."""
    high_variance_indices = np.where(new_data_variance > mean_var_train)[0]
    # Ensure indices are within bounds
    high_variance_indices = high_variance_indices[high_variance_indices < len(x_incoming)]

    return x_incoming[high_variance_indices], y_incoming[high_variance_indices]

# if the data meets the above three conditions, then the data will be added to the training set

def prune_low_variance_data(x_new, y_new, model, likelihood, target_size, num_iter_delete=None):
    """
    Prunes data points from x_new and y_new by iteratively removing points with the lowest GP variance
    until the dataset size is reduced to the target_size. The pruning is based on the GP variance predictions.

    Parameters:
    -----------
    x_new : np.ndarray
        The feature data points to be pruned.
    y_new : np.ndarray
        The target values corresponding to x_new.
    model : gpytorch.models.ExactGP
        The trained GP model used to predict variances and update the model during pruning.
    likelihood : gpytorch.likelihoods.GaussianLikelihood
        The likelihood function associated with the GP model.
    target_size : int
        The desired number of data points after pruning.
    num_iter_delete : int or None, optional
        The number of data points to delete in each iteration. If None, the function will determine
        how many points to delete based on the difference between the current size and target_size.

    Returns:
    --------
    x_new_reduced : np.ndarray
        The pruned feature data points.
    y_new_reduced : np.ndarray
        The pruned target values corresponding to x_new_reduced.
    """
    Gpytorch_SVGP_var_new_data = full_gp_variance(model, likelihood, x_new)
    while x_new.shape[0] > target_size:
        if num_iter_delete is None:
            num_to_delete = x_new.shape[0] - target_size
            min_index = np.argsort(Gpytorch_SVGP_var_new_data)[:num_to_delete]
            x_new = np.delete(x_new, min_index, axis=0)
            y_new = np.delete(y_new, min_index, axis=0)
        else:
            num_to_delete = min(num_iter_delete, x_new.shape[0] - target_size)
            min_index = np.argsort(Gpytorch_SVGP_var_new_data)[:num_to_delete]
            x_new = np.delete(x_new, min_index, axis=0)
            y_new = np.delete(y_new, min_index, axis=0)
            model, likelihood = sparse_gp_model_train(x_new, y_new, 50, 1, predic_warmup=True, train_viz_disable=True)
            _, Gpytorch_SVGP_var_new_data, _ = sparse_gp_model_predict(model, likelihood, x_new)
    x_new_reduced, y_new_reduced = x_new, y_new
    return x_new_reduced, y_new_reduced

def prune_data_with_inducing_cluster(x_new, y_new, inducing_point, model, likelihood, target_size):
    """
    Prunes the data points based on their similarity to the inducing points in a Gaussian Process (GP) model.
    This method aims to reduce the data size while maintaining the feature space coverage by selectively
    removing data points with the smallest GP variance within each cluster.

    Parameters:
    -----------
    x_new : np.ndarray
        The feature data points to be pruned.
    y_new : np.ndarray
        The target values corresponding to x_new.
    model : gpytorch.models.ExactGP
        The trained GP model used to obtain the inducing points and calculate variances.
    likelihood : gpytorch.likelihoods.GaussianLikelihood
        The likelihood function associated with the GP model.
    target_size : int
        The desired number of data points after pruning.

    Returns:
    --------
    x_new_reduced : np.ndarray
        The pruned feature data points.
    y_new_reduced : np.ndarray
        The pruned target values corresponding to x_new_reduced.
    """

    train_data = np.hstack((x_new, y_new))  # Combine train_x and train_y for 2D clustering
    Gpytorch_SVGP_var_new_data = simple_sparse_gp_variance(model, likelihood, x_new)

    # Step 1: Find the nearest points in train_x to the inducing points
    nn_model = NearestNeighbors(n_neighbors=1)
    nn_model.fit(x_new)
    nearest_indices = nn_model.kneighbors(inducing_point, return_distance=False).flatten()
    nearest_points = np.hstack((x_new[nearest_indices], y_new[nearest_indices]))

    # Step 2: Perform K-means clustering using the nearest points as initial centers
    kmeans = KMeans(n_clusters=inducing_point.shape[0], init=nearest_points, n_init=1, random_state=0)
    kmeans.fit(train_data)

    # Get cluster labels and calculate variance for each cluster
    cluster_labels = kmeans.labels_
    variances = []

    # Calculate the mean GP variance for each cluster
    for cluster in range(inducing_point.shape[0]):
        cluster_points_y = Gpytorch_SVGP_var_new_data[cluster_labels == cluster]
        cluster_variance = np.mean(cluster_points_y)
        variances.append((cluster, cluster_variance))

    # Step 3: delete the points with the smallest variance in each cluster
    current_size = len(x_new)

    # Create a boolean mask to keep track of which points to retain
    indices_to_keep = np.ones(current_size, dtype=bool)

    # Calculate the number of points to delete for each cluster
    total_points = len(x_new)
    delete_ratios = []

    # Determine the number of points to delete from each cluster
    for cluster, variance in variances:
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_size = len(cluster_indices)

        # Calculate the base delete ratio to ensure each cluster contributes to the data reduction
        delete_ratio = (1 - target_size / total_points)
        num_to_delete = round(cluster_size * delete_ratio)  # Use round() to get a more accurate integer
        delete_ratios.append((cluster, num_to_delete, cluster_indices))

    # Delete points with the smallest GP variance within each cluster
    for cluster, num_to_delete, cluster_indices in delete_ratios:
        if current_size <= target_size:
            break  # Stop if the target size has been reached

        if num_to_delete <= 0 or num_to_delete >= len(cluster_indices):
            continue  # Skip if the number of points to delete is 0 or too large

        # Sort the points within the cluster by GP variance in ascending order
        cluster_points_variance = Gpytorch_SVGP_var_new_data[cluster_indices]
        sorted_indices_within_cluster = np.argsort(cluster_points_variance)

        # Delete the specified number of points with the smallest variance
        for idx in sorted_indices_within_cluster[:num_to_delete]:
            if current_size <= target_size:
                break  # Stop if the target size has been reached
            indices_to_keep[cluster_indices[idx]] = False
            current_size -= 1

    # Apply the boolean mask to retain the selected data points
    x_new_reduced = x_new[indices_to_keep]
    y_new_reduced = y_new[indices_to_keep]

    return x_new_reduced, y_new_reduced

def full_gp_variance(model, likelihood, x_train, x_test=None):
    """
    Calculate the leave-one-out full gp variances using Schur complement formula.
    kxx- kxf@kff^{-1}@kuf@kxf
    Parameters:
    -----------
    x_train : np.ndarray
        Training data features.
    model : gpytorch.models.ExactGP
        The trained GP model.
    likelihood : gpytorch.likelihoods.GaussianLikelihood
        The likelihood function associated with the GP model.

    Returns:
    --------
    loo_variances : np.ndarray
        Leave-one-out variances calculated using Schur complement.
    """
    # Convert x_train to torch tensor
    x_train_torch = torch.from_numpy(x_train).float()
    x_train_torch = (x_train_torch - model.X_mean) / model.X_std
    if x_test is None:
        x_test_torch = x_train_torch
    else:
        x_test_torch = torch.from_numpy(x_test).float()
        x_test_torch = (x_test_torch - model.X_mean) / model.X_std

    # Evaluate model and likelihood in evaluation mode
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Compute the full covariance matrix
        K = model.covar_module.base_kernel(x_train_torch).evaluate()
        noise_variance = likelihood.noise
        K_noisy = K + noise_variance * torch.eye(K.size(0))
        # Compute the inverse of the noisy covariance matrix
        K_inv = torch.linalg.inv(K_noisy)
        if x_test is None:
            # LOO variances are the inverse of the diagonal elements of K_inv
            variances = 1 / torch.diag(K_inv)
        else:
            K_xx = model.covar_module.base_kernel(x_test_torch).evaluate()
            K_xf = model.covar_module.base_kernel(x_test_torch, x_train_torch).evaluate()
            variances = K_xx - K_xf @ K_inv @ K_xf.T + noise_variance
            variances = torch.diag(variances)

        variances = variances.sqrt().detach().cpu().numpy()
        #rescale the predictive
        variances = variances * (model.Y_max.numpy() - model.Y_min.numpy()) / 2
    # Return the LOO variances as a numpy array
    return variances

def simple_sparse_gp_variance(model, likelihood, x_train, x_test=None):
    """
    Calculate the leave-one-out variances for a Sparse GP model using Schur complement.
    kxx- kxz@kzz^{-1}@kxz
    Parameters:
    -----------
    x_train : np.ndarray
        Training data features.
    model : gpytorch.models.ApproximateGP
        The trained Sparse GP model using inducing points.
    likelihood : gpytorch.likelihoods.GaussianLikelihood
        The likelihood function associated with the GP model.

    Returns:
    --------
    loo_variances : np.ndarray
        Leave-one-out variances for the Sparse GP model.
    """
    # Convert x_train to torch tensor and normalize
    x_train_torch = torch.from_numpy(x_train).float()

    x_train_torch = (x_train_torch - model.X_mean) / model.X_std
    if x_test is None:
        x_test_torch = x_train_torch
    else:
        x_test_torch = torch.from_numpy(x_test).float()
        x_test_torch = (x_test_torch - model.X_mean) / model.X_std

    # Get the inducing points and the corresponding covariances
    inducing_points = model.covar_module.inducing_points
    K_zz = model.covar_module.base_kernel(inducing_points).evaluate()  # Covariance among inducing points
    K_xz = model.covar_module.base_kernel(x_test_torch, inducing_points).evaluate()  # Covariance between data and inducing points
    diag_K_xx = torch.diag(model.covar_module.base_kernel(x_test_torch).evaluate())

    # Compute the inverse of K_zz + noise
    noise_variance = likelihood.noise
    K_zz_noisy = K_zz + noise_variance * torch.eye(K_zz.size(0))
    K_zz_inv = torch.linalg.inv(K_zz_noisy)

    # Compute the Schur complement for LOO variances
    with torch.no_grad():
        # Compute the diagonal of the approximate covariance matrix
        diag_K_xz_K_zz_inv_K_zx = torch.sum(K_xz @ K_zz_inv * K_xz, dim=1)

        # Approximate LOO variances using the Schur complement
        loo_variances = diag_K_xx - diag_K_xz_K_zz_inv_K_zx + noise_variance
        loo_variances = loo_variances.sqrt().detach().cpu().numpy()

        #rescale the predictive
        loo_variances = loo_variances * (model.Y_max.numpy() - model.Y_min.numpy()) / 2

    return loo_variances

def Normalize_data(x, y, model):
    x = (x - model.X_mean.numpy()) / model.X_std.numpy()
    y = 2 * ((y - model.Y_min.numpy()) / (model.Y_max.numpy() - model.Y_min.numpy())) - 1
    return x, y

def Normalized_recovery_data(x, y, model):
    x = x * model.X_std.numpy() + model.X_mean.numpy()
    y = (y + 1) * (model.Y_max.numpy() - model.Y_min.numpy()) / 2 + model.Y_min.numpy()
    return x, y

def data_selection(model, likelihood, x, y, x_incoming, y_incoming, target_size, dataset_ymin, dataset_ymax, selection_mode="inducing_cluster"):
    """
    Main function to select and prune data based on predictive variance, range, and confidence interval.

    Parameters:
    ----------
    model : gpytorch.models.ExactGP
        The trained Gaussian Process model on the initial dataset.
    likelihood : gpytorch.likelihoods.GaussianLikelihood
        The likelihood function associated with the Gaussian Process model.
    x : np.ndarray
        Original training data features.
    y : np.ndarray
        Original training data target values.
    x_incoming : np.ndarray
        New incoming data points (features) to be considered for inclusion in the training set.
    y_incoming : np.ndarray
        Target values associated with the new incoming data points.
    target_size : int
        Target size for the training dataset.
    delete_mode : str, optional
        Mode for removing data points with the lowest predictive variance.
    num_iter_delete : int, optional
        Number of low-variance data points to remove in each iteration.

    Returns:
    -------
    x_new : np.ndarray
        Updated training data (features).
    y_new : np.ndarray
        Updated training data (target values).
    """

    # Step 1: Filter incoming data based on range of y
    x_incoming_filtered, y_incoming_filtered, has_valid_data = filter_by_range(x_incoming, y_incoming, dataset_ymin, dataset_ymax)

    if has_valid_data is False:
        print("No data met the filtering criteria.")
        return x, y


    # Step 2: Check if data lies within the old GP’s 95% confidence interval if current size > 2/3 of target size
    try:
        # Attempt to execute the sparse GP model prediction
        new_x_gp_mean, new_x_gp_std, inducing_points = sparse_gp_model_predict(model, likelihood, x_incoming_filtered)
        if len(x) > (2 * target_size) / 3:
            within_confidence = filter_by_confidence_interval(y_incoming_filtered, new_x_gp_mean, new_x_gp_std)
            x_incoming_filtered = x_incoming_filtered[within_confidence]
            y_incoming_filtered = y_incoming_filtered[within_confidence]
    except Exception as e:
        # _, _, inducing_points = sparse_gp_model_predict(model, likelihood, x_incoming_filtered[:1])
        print("there is a bug in gpytorch, data selection will be skipped")
        return x, y


    # # Step 3: Retain data points with measure distance higher than the mean variance of the training data
    #measures the time

    if selection_mode == "inducing_cluster":
        train_x_gp_dis = simple_sparse_gp_variance(model, likelihood, x)
        new_x_gp_dis = simple_sparse_gp_variance(model, likelihood, x, x_incoming_filtered)

    elif selection_mode == "low_variance":
        train_x_gp_dis = full_gp_variance(model, likelihood, x)
        new_x_gp_dis = full_gp_variance(model, likelihood, x, x_incoming_filtered)

    mean_dis_train = np.mean(train_x_gp_dis)

    x_incoming_final, y_incoming_final = filter_by_variance(
        x_incoming_filtered, y_incoming_filtered, new_x_gp_dis, mean_dis_train
    )

    # # Combine old and new data
    x_incoming_final = x_incoming_final.reshape(-1, 1)  # Reshape to be a column vector
    y_incoming_final = y_incoming_final.reshape(-1, 1)  # Reshape to be a column vector
    x_new = np.vstack((x, x_incoming_final))
    y_new = np.vstack((y, y_incoming_final))

    # # Remove low-variance data if dataset size exceeds target size
    if x_new.shape[0] > target_size:

        if selection_mode == "inducing_cluster":
            x_new, y_new = prune_data_with_inducing_cluster(x_new, y_new, inducing_points, model, likelihood, target_size)

        elif selection_mode == "low_variance":
            x_new, y_new = prune_low_variance_data(x_new, y_new, model, likelihood, target_size)


    return x_new, y_new

# if the dataset is larger than the target size, then the data will be removed
def prune_random(x_new, y_new, seed_value=42, data_size=None):

    # Fix the random seed for reproducibility
    np.random.seed(seed_value)
    # If data_size is None, no pruning is done
    if data_size is None:
        x_new_reduced = x_new
        y_new_reduced = y_new
    else:
        # Randomly sample from the combined dataset to keep the size at data_size
        indices = np.random.choice(x_new.shape[0], data_size, replace=False)

        # Update the training data with the randomly sampled points
        x_new_reduced = x_new[indices]
        y_new_reduced = y_new[indices]

    return x_new_reduced, y_new_reduced

def prune_data_with_histogram(x_new, y_new, target_size, bins=50):
    """
    Prunes the dataset so that its size equals max_size. The oldest data points or those in bins with fewer samples will
    be removed first until the dataset size equals target_size.

    :param X_train: list of input training data
    :param y_train: list of target training data
    :param max_size: the maximum allowable size of the dataset
    :param bins: number of bins for histogram
    :return: pruned X_train and y_train datasets with size equal to max_size
    """
    # Convert lists to NumPy arrays for histogram operations
    x_new = np.array(x_new)
    y_new = np.array(y_new)

    # If the dataset size is already less than or equal to max_size, no pruning is necessary
    if len(x_new) <= target_size:
        return x_new.tolist(), y_new.tolist()

    # Create histograms for both X_train and y_train
    h_x, bin_edges_x = np.histogram(x_new, bins=bins)
    h_y, bin_edges_y = np.histogram(y_new, bins=bins)

    # Sort the bins based on the number of samples (ascending order)
    sorted_bins_x = np.argsort(h_x)
    sorted_bins_y = np.argsort(h_y)

    # Initialize the number of points to remove
    remove_count = len(x_new) - target_size

    # Initialize lists to store the indices of the data points to keep
    keep_indices = np.ones(x_new.shape[0], dtype=bool)

    # Remove points from X_train first, starting from the bins with fewer samples
    removed_points = 0
    for i in sorted_bins_x:
        if removed_points >= remove_count:
            break
        bin_min = bin_edges_x[i]
        bin_max = bin_edges_x[i + 1]
        indices_to_remove = np.where((x_new >= bin_min) & (x_new < bin_max))[0]
        num_to_remove = min(len(indices_to_remove), remove_count - removed_points)
        keep_indices[indices_to_remove[:num_to_remove]] = False
        removed_points += num_to_remove

    # If more points need to be removed, continue with y_train
    if removed_points < remove_count:
        for i in sorted_bins_y:
            if removed_points >= remove_count:
                break
            bin_min = bin_edges_y[i]
            bin_max = bin_edges_y[i + 1]
            indices_to_remove = np.where((y_new >= bin_min) & (y_new < bin_max))[0]
            num_to_remove = min(len(indices_to_remove), remove_count - removed_points)
            keep_indices[indices_to_remove[:num_to_remove]] = False
            removed_points += num_to_remove

    # Prune the data
    x_new_reduced = x_new[keep_indices].tolist()
    y_new_reduced = y_new[keep_indices].tolist()

    x_new_reduced = np.array(x_new_reduced).reshape(-1, 1)
    y_new_reduced = np.array(y_new_reduced).reshape(-1, 1)

    return x_new_reduced, y_new_reduced
