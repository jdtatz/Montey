from setuptools import setup

setup(
    name="montey",
    version="0.1.0",
    packages=["montey"],
    install_requires=[
        "cupy",
        "numpy",
        "numba",
        "xarray",
        "pint",
    ],
    python_requires='~=3.7',
    package_data={"montey": ["kernel.ptx"]},
)
