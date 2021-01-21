from setuptools import setup

setup(
    name="montey",
    version="0.0.2",
    author="Julia Tatz",
    author_email="tatz.j@northeastern.edu",
    url="https://github.com/jdtatz/Montey/",
    packages=["montey"],
    install_requires=[
        "cupy~=8.0",
        "numpy~=1.19",
        "numba~=0.50",
        "xarray~=0.16",
        "pint~=0.15",
    ],
    python_requires='~=3.7',
    package_data={"montey": ["kernel.ptx"]},
)
