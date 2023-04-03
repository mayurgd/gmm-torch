import setuptools

try:
    with open("README.md", "r") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = ""


setuptools.setup(
    name="gmm_torch",
    version="v1.1.0",
    url="https://github.com/mayurgd/gmm-torch",
    install_requires=["numpy", "scipy"],
    packages=setuptools.find_packages(),
)
