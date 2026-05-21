import importlib
import importlib.machinery
import importlib.util
import os


def _workspace_extension_candidates():
    package_src = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        yield os.path.join(package_src, f"mpc_builder{suffix}")


def load_mpc_builder():
    for candidate in _workspace_extension_candidates():
        if os.path.exists(candidate):
            loader = importlib.machinery.ExtensionFileLoader("mpc_builder", candidate)
            spec = importlib.util.spec_from_file_location("mpc_builder", candidate, loader=loader)
            module = importlib.util.module_from_spec(spec)
            loader.exec_module(module)
            return module

    return importlib.import_module("mpc_builder")
