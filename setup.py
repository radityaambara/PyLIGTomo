from setuptools import setup, find_packages

setup(
    name="PyLIGTomo",
    version="1.0",
    packages=find_packages(),
    install_requires=["numpy<=2.0",
                      "scipy",
                      "matplotlib",
                      "pyevtk",
                      "pandas"],
    author="Putu Raditya",
    author_email="radityaambara@gmail.com",
    description="Local Tomography Package using irregular flexible grid",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/radityaambara/PyLIGTomo",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)