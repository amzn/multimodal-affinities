import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="multimodal-affinities",
    version="1.0.0",
    author="nondisclosed",
    author_email="author@nondisclosed.com",
    description="Code for the paper: 'Learning Multimodal Affinities for Textual Editing in Images'",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/multimodal-affinities",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
