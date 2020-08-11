from setuptools import setup

setup(
    name="Montey",
    version="0.1",
    packages=['montey'],
    python_requires="~=3.7",
    install_requires=["numpy", "numba"]
)
