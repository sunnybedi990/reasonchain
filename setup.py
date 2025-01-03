from setuptools import setup, find_packages

# Core dependencies required for basic functionality
CORE_REQUIREMENTS = [
    "numpy>=1.21.0,<2.0.0",
    "scipy>=1.8.0,<1.14.0",
    "scikit-learn>=1.0.0",
    "torch",
    "tqdm",
    "python-dotenv",
    "psutil",
]

# Vector database dependencies
DB_REQUIREMENTS = [
    "faiss-cpu",  # Use faiss-gpu for GPU support
    "pymilvus>=2.0.0",
    "pinecone-client",
    "qdrant-client",
    "weaviate-client>=4.0.0",  # Specify v4 explicitly
]

# LLM-related dependencies
LLM_REQUIREMENTS = [
    "transformers",
    "sentence-transformers",
    "ollama",
    "groq",
    "openai>=1.0.0",
]

# RAG-related dependencies
RAG_REQUIREMENTS = [
    "matplotlib",
    "tabula-py",
    "camelot-py[cv]",  # [cv] includes OpenCV dependencies
    "pymupdf",
    "tensorflow-hub",
    "gensim",
    "layoutparser",
    "pdf2image",
    "pytesseract",
    "pdfplumber",
    "fastapi[all]",  # [all] includes all optional dependencies
    "jpype1",
    "llama-index-core==0.12.2",
    "llama-parse",
    "llama-index-readers-file",
    "opencv-python",
    "datasets",
    "python-pptx",
    "moviepy",
    "SpeechRecognition",
    "ebooklib",
    "beautifulsoup4",
    "python-docx",
    "spacy",
    "striprtf",
]

# Development dependencies
DEV_REQUIREMENTS = [
    "pytest",
    "black",
    "isort",
    "flake8",
    "mypy",
]

setup(
    name="reasonchain",
    version="0.2.11",
    description="A modular AI reasoning library for building intelligent agents.",
    long_description=open("Readme.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sunnybedi990/reasonchain",
    author="Baljindersingh Bedi",
    author_email="baljindersinghbedi409@gmail.com",
    license="MIT",
    packages=find_packages(
        include=["reasonchain", "reasonchain.*"],
        exclude=["tests", "examples", "models", "pdfs"],
    ),
    install_requires=CORE_REQUIREMENTS,
    extras_require={
        "all": CORE_REQUIREMENTS + DB_REQUIREMENTS + LLM_REQUIREMENTS + RAG_REQUIREMENTS,
        "core": CORE_REQUIREMENTS,
        "db": DB_REQUIREMENTS,
        "llm": LLM_REQUIREMENTS,
        "rag": RAG_REQUIREMENTS,
        "dev": DEV_REQUIREMENTS,
    },
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
    include_package_data=True,
)
