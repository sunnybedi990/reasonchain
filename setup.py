from setuptools import setup, find_packages

setup(
    name="reasonchain",
    version="0.1.0",
    description="A modular AI reasoning library for building intelligent agents.",
    long_description=open("Readme.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sunnybedi990/reasonchain",
    author="Your Name",
    author_email="baljindersinghbedi409@gmail.com",
    license="MIT",
    packages=find_packages(exclude=("tests", "examples")),
    install_requires=[
        "openai",
        "transformers",
        "faiss-cpu",
        "numpy",
        "scikit-learn",
        "torch",
        "tqdm",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
