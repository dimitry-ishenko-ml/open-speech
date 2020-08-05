import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(name="open_speech",
    version="5.0",
    author="Dimitry Ishenko",
    author_email="dimitry.ishenko@gmail.com",
    description="Open Speech Datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dimitry-ishenko-ml/open-speech",
    packages=setuptools.find_packages(),
    install_requires=[ "pandas", "tensorflow" ],
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
)