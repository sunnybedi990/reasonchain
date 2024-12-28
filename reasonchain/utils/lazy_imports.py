import importlib
import subprocess
import sys
import logging
import threading
import asyncio

logging.basicConfig(level=logging.INFO)

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
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "install", self.package_name,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            logging.error(f"Error installing {self.package_name}: {stderr.decode()}")
        else:
            logging.info(f"Successfully installed {self.package_name}")

    def _check_dependencies(self):
        # Automatically check and install all dependencies for the package being installed
        try:
            logging.info(f"Checking dependencies for {self.package_name}...")
            # Use pip's check feature or a custom dependency resolution
            subprocess.check_call([sys.executable, "-m", "pip", "check"])
        except subprocess.CalledProcessError as e:
            logging.warning(f"Dependency issues detected: {e}. Attempting to install missing dependencies...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", self.package_name, "--upgrade"])
            except subprocess.CalledProcessError as install_error:
                logging.error(f"Failed to resolve dependencies for {self.package_name}: {install_error}")
                raise install_error





# Define lazy imports with optional package names
sentence_transformers = LazyImport("sentence_transformers", "sentence-transformers")
transformers = LazyImport("transformers","transformers")
tensorflow_hub = LazyImport("tensorflow_hub", "tensorflow_hub")
gensim_downloader = LazyImport("gensim.downloader", "gensim")
os = LazyImport("os")

# Core Libraries
numpy = LazyImport("numpy", "numpy")
scipy = LazyImport("scipy", "scipy")
sklearn = LazyImport("sklearn", "scikit-learn")
torch = LazyImport("torch", "torch")
tqdm = LazyImport("tqdm", "tqdm")
pandas = LazyImport("pandas","pandas")
requests = LazyImport("requests","requests")

# Database Libraries
faiss = LazyImport("faiss", "faiss-cpu")
pymilvus = LazyImport("pymilvus", "pymilvus")
pinecone = LazyImport("pinecone", "pinecone")
qdrant_client = LazyImport("qdrant_client", "qdrant-client")
weaviate = LazyImport("weaviate", "weaviate-client")
pickle = LazyImport("pickle", "pickle")

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
time = LazyImport("time", "time")
