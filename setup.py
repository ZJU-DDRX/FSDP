from setuptools import Extension, find_packages, setup
import numpy
import pybind11


eigen_include = "/usr/include/eigen3"
cpp_args = ["-O3", "-Wall", "-shared", "-std=c++14", "-fPIC", "-march=native"]

ext_modules = [
    Extension(
        "mpc_builder",
        ["src/mpc/cpp_mpc/mpc_builder.cpp"],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            numpy.get_include(),
            eigen_include,
        ],
        language="c++",
        extra_compile_args=cpp_args,
    ),
]

setup(
    name="fsdp",
    version="0.0.0",
    author="ZJU_Automation_Engineer",
    description="FSDP runtime modules and MPC C++ extension",
    ext_modules=ext_modules,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
