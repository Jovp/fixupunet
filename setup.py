import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fixupunet",
    version="0.0.2",
    author="Julien Philip",
    author_email="juphilip@adobe.com",
    description="Pytorch implementation of a modern unet using fixup initialization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jovp/fixupunet",
    project_urls={
        "Bug Tracker": "https://github.com/Jovp/fixupunet/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    include_package_data=True,
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
    ],
)
