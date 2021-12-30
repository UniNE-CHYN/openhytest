import setuptools, os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()
    
setuptools.setup(
    name="openhytest",
    version="1.0",
    author="Nathan Dutler, ",
    author_email="neissenod@gmail.com",
    description='Package for well test analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    project_urls={
        "Bug Tracker": "https://github.com/UniNE-CHYN/openhytest",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=install_requires
)