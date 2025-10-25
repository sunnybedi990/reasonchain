import importlib
import subprocess
import sys
import logging
import threading
import asyncio
import os  # Direct import for built-in module
import pickle
import time
import yaml
import json
logging.basicConfig(level=logging.INFO)

# Assign os directly to avoid LazyImport for built-in module
os = os

class LazyImport:
    _installed_cache = set()  # Cache for installed libraries
    _lock = threading.Lock()  # Thread safety lock

    def __init__(self, module_name, package_name=None):
        self.module_name = module_name
        self.package_name = package_name or module_name
        self._module = None
        logging.info(f"LazyImport initialized for {module_name}")

    def __getattr__(self, item):
        if self._module is None:
            self._install_if_missing()
            self._module = importlib.import_module(self.module_name)
        return getattr(self._module, item)

    def _install_if_missing(self):
        with LazyImport._lock:  # Ensure thread safety
            if self.package_name in LazyImport._installed_cache:
                return
            self._check_dependencies()  # Automatically check dependencies
            try:
                importlib.import_module(self.module_name)
            except ImportError:
                logging.info(f"{self.module_name} not found. Installing {self.package_name}...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", self.package_name])
                except subprocess.CalledProcessError as e:
                    logging.error(f"Failed to install {self.package_name}: {e}. Please install it manually.")
                    raise e
            LazyImport._installed_cache.add(self.package_name)

    @staticmethod
    def uninstall_package(package_name):
        try:
            logging.info(f"Uninstalling {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package_name])
            LazyImport._installed_cache.discard(package_name)
        except Exception as e:
            logging.error(f"Error uninstalling {package_name}: {e}")

    def preload(self):
        if self._module is None:
            self._install_if_missing()
            self._module = importlib.import_module(self.module_name)

    async def _install_async(self):
        """Asynchronous installation with multiple fallback approaches."""
        install_commands = [
            [sys.executable, "-m", "pip", "install", self.package_name],
            ["pip", "install", self.package_name],
            ["pip3", "install", self.package_name]
        ]

        for cmd in install_commands:
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                if process.returncode == 0:
                    logging.info(f"Successfully installed {self.package_name}")
                    return
                else:
                    logging.warning(f"Command {' '.join(cmd)} failed: {stderr.decode()}")
            except Exception as e:
                logging.warning(f"Installation attempt failed with {' '.join(cmd)}: {e}")
                continue

        logging.error(f"All installation attempts failed for {self.package_name}")
        raise Exception(f"Failed to install {self.package_name} asynchronously")

    def _try_install(self, command, suppress_output=True):
        """Helper method to try different installation commands."""
        try:
            if suppress_output:
                result = subprocess.check_call(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                result = subprocess.check_call(command)
            return True
        except subprocess.CalledProcessError:
            return False

    def _check_dependencies(self):
        """Check and install dependencies using multiple fallback approaches."""
        try:
            logging.info(f"Checking dependencies for {self.package_name}...")
            # First try to import the module directly
            try:
                importlib.import_module(self.module_name)
                return  # Module exists and can be imported
            except ImportError:
                pass  # Continue with installation if import fails

            # Installation commands to try in order
            install_commands = [
                [sys.executable, "-m", "pip", "install", self.package_name],  # Try python -m pip first
                ["pip", "install", self.package_name],                        # Try direct pip command
                ["pip3", "install", self.package_name],                       # Try pip3 as fallback
            ]

            # Try each installation command
            for cmd in install_commands:
                logging.info(f"Attempting to install {self.package_name} using: {' '.join(cmd)}")
                if self._try_install(cmd):
                    logging.info(f"Successfully installed {self.package_name}")
                    return
                
            # If all basic attempts fail, try with --upgrade
            upgrade_commands = [
                [sys.executable, "-m", "pip", "install", "--upgrade", self.package_name],
                ["pip", "install", "--upgrade", self.package_name],
                ["pip3", "install", "--upgrade", self.package_name]
            ]

            for cmd in upgrade_commands:
                logging.info(f"Attempting to upgrade {self.package_name} using: {' '.join(cmd)}")
                if self._try_install(cmd, suppress_output=False):  # Show output for upgrade attempts
                    logging.info(f"Successfully upgraded {self.package_name}")
                    return

            raise Exception(f"Failed to install {self.package_name} after trying multiple methods")

        except Exception as e:
            logging.error(f"Error during dependency check for {self.package_name}: {e}")
            raise e





# Define lazy imports with optional package names
sentence_transformers = LazyImport("sentence_transformers", "sentence-transformers")
transformers = LazyImport("transformers","transformers")
tensorflow_hub = LazyImport("tensorflow_hub", "tensorflow_hub")
gensim_downloader = LazyImport("gensim.downloader", "gensim")

# Core Libraries
numpy = LazyImport("numpy", "numpy")
scipy = LazyImport("scipy", "scipy")
sklearn = LazyImport("sklearn", "scikit-learn")
torch = LazyImport("torch", "torch")
tqdm = LazyImport("tqdm", "tqdm")
pandas = LazyImport("pandas","pandas")
requests = LazyImport("requests","requests")
os = os

# Database Libraries
faiss = LazyImport("faiss", "faiss-cpu")
pymilvus = LazyImport("pymilvus", "pymilvus")
pinecone = LazyImport("pinecone", "pinecone")
qdrant_client = LazyImport("qdrant_client", "qdrant-client")
weaviate = LazyImport("weaviate", "weaviate-client")
psycopg2 = LazyImport("psycopg2", "psycopg2-binary")
elasticsearch = LazyImport("elasticsearch", "elasticsearch")
pickle = pickle

# LLM Integration
ollama = LazyImport("ollama", "ollama")
groq = LazyImport("groq", "groq")
openai = LazyImport("openai", "openai")
whisper = LazyImport("whisper","openai-whisper")

# Retrieval-Augmented Generation (RAG)
matplotlib = LazyImport("matplotlib", "matplotlib")
tabula_py = LazyImport("tabula", "tabula-py")
camelot = LazyImport("camelot", "camelot-py")
pymupdf = LazyImport("fitz", "pymupdf")
tensorflow_hub = LazyImport("tensorflow_hub", "tensorflow-hub")
gensim = LazyImport("gensim", "gensim")
layoutparser = LazyImport("layoutparser", "layoutparser")
pdf2image = LazyImport("pdf2image", "pdf2image")
pytesseract = LazyImport("pytesseract", "pytesseract")
pdfplumber = LazyImport("pdfplumber", "pdfplumber")
fastapi = LazyImport("fastapi", "fastapi[standard]")
jpype1 = LazyImport("jpype1", "jpype1")
llama_index_core = LazyImport("llama_index_core", "llama-index-core==0.12.2")
llama_parse = LazyImport("llama_parse", "llama-parse")
llama_index_readers_file = LazyImport("llama_index_readers_file", "llama-index-readers-file")
dotenv = LazyImport("dotenv", "python-dotenv")
opencv_python = LazyImport("cv2", "opencv-python")
datasets = LazyImport("datasets", "datasets")
pptx = LazyImport("pptx", "python-pptx")
moviepy = LazyImport("moviepy", "moviepy")
speech_recognition = LazyImport("speech_recognition", "SpeechRecognition")
ebooklib = LazyImport("ebooklib", "ebooklib")
bs4 = LazyImport("bs4", "beautifulsoup4")
docx = LazyImport("docx", "python-docx")
spacy = LazyImport("spacy","spacy")
fitz = LazyImport("fitz", "fitz")
striprtf = LazyImport("striprtf","striprtf")
psutil = LazyImport("psutil", "psutil")
time = time
yaml = yaml
json = json
