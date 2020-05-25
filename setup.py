from setuptools import setup, find_packages

def read_readme():
    with open("README.md") as fp:
        long_desc = fp.read()
    return long_desc

setup(
    name = "pyvision",
    version = "1.0.0",
    author = "Pranjal Datta",
    description = ("Ready-to-use implementations of some of the most common "
                "computer vision algorithms."),
    license = "MIT",
    long_description = read_readme(),
    url = "https://github.com/pranjaldatta/PyVision",

    packages = find_packages(),
    include_package_data = True,
)