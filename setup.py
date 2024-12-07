from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import subprocess


class PostInstallCommand(install):
    """Custom post-installation for downloading Spacy model."""
    def run(self):
        # Run the original install code
        install.run(self)
        try:
            # Automatically download the Spacy English model
            subprocess.check_call([os.sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        except Exception as e:
            print(f"Error downloading Spacy model: {e}")


setup(
    name="reasonchain",
    version="0.1.1",
    description="A modular AI reasoning library for building intelligent agents.",
    long_description=open("Readme.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sunnybedi990/reasonchain",
    author="Baljindersingh Bedi",
    author_email="baljindersinghbedi409@gmail.com",
    license="MIT",
    packages=find_packages(exclude=("tests", "examples")),
    install_requires=[
        # Core Libraries
        "numpy<2",
        "scikit-learn",
        "torch",
        "tqdm",
        
        # Database Libraries
        "faiss-cpu",
        "pymilvus",
        "pinecone",
        "qdrant-client",
        "weaviate-client",
        
        # LLM Integration
        "transformers",
        "ollama",
        "groq",
        "openai",
        
        # Retrieval-Augmented Generation (RAG)
        "tabula-py",
        "camelot-py",
        "pymupdf",
        "sentence-transformers",
        "tensorflow_hub",
        "gensim",
        "layoutparser",
        "pdf2image",
        "pytesseract",
        "pdfplumber",
        "spacy",
        "fastapi[standard]",
        "jpype1",
        "llama-index-core==0.12.2",
        "llama-parse",
        "llama-index-readers-file",
        "python-dotenv",
        "opencv-python",
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
    cmdclass={
        'install': PostInstallCommand,
    },
)
