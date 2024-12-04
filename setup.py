from setuptools import setup, find_packages

setup(
    name="reasonchain",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "faiss-cpu",
        "sentence-transformers",
        "transformers",
        "torch",
    ],
    description="A reasoning library with RAG integration and advanced pipelines.",
    author="Your Name",
    url="https://github.com/sunnybedi990/reasonchain",
)
