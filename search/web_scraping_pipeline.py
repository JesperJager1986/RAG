import hashlib

import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import requests
import os
import spacy
import faiss
from pathlib import Path

from search.model import Model


class WebScrapingPipeline:
    def __init__(self):
        self._raw_content = None
        self._cleaned_content = None
        self._embedding = None
        self._current_document = None
        self._index = None

    def get_file_name(self):
        return self.url.name

    def create_file_name_from_path(self, folder , add_name_to_name: str = None, extension =".txt") -> Path:
        if add_name_to_name is not None:
            new_stem = f"{self.url.stem}_{add_name_to_name}"
            self.url = self.url.with_name(new_stem + self.url.suffix)
        path = folder / self.url.stem

        path = path.with_suffix(extension)
        return path

    def fetch(self, url: str | Path):
        """Fetch raw HTML content, falling back to Selenium if requests is unsuccessful."""
        self.url = Path(url) #todo this not a good practice
        try:
            response = requests.get(str(url), timeout=10)
            response.raise_for_status()
            self._raw_content = response.content
        except requests.exceptions.RequestException as e:
            print(f"Requests failed: {e}")

        if self._raw_content is None:
            self._fetch_with_selenium()

        if self._raw_content is None:
            raise RuntimeError("No data is loaded")

        self.current_document = self._raw_content

        return self

    def _fetch_with_requests(self):
        """Try to fetch content using requests."""
        try:
            response = requests.get(str(self.url), timeout=10)
            response.raise_for_status()
            self._raw_content = response.content
        except requests.exceptions.RequestException as e:
            print(f"Requests failed: {e}")
        return self

    def _fetch_with_selenium(self):
        """Fetch content using Selenium for JavaScript-rendered pages."""
        print("Falling back to Selenium for JavaScript rendering...")
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        try:
            with webdriver.Chrome(
                service=Service(ChromeDriverManager().install()), options=options
            ) as driver:
                driver.get(str(self.url))
                self._raw_content = driver.page_source
                self.current_document = self._raw_content
        except Exception as e:
            print(f"Selenium failed: {e}")
            self._raw_content = None
        return self

    def format(self):
        """Extract meaningful content from raw HTML."""
        if self._raw_content is None:
            print("No content to clean.")
        else:
            soup = BeautifulSoup(self._raw_content, 'html.parser')
            content = (
                soup.find('div', {'class': 'mw-parser-output'}) or
                soup.find('div', {'id': 'content'})
            )
            self._cleaned_content = content.get_text(strip=True) if content else None
            self.current_document = self._cleaned_content
        return self

    def preprocess_text(self):
        # python - m spacy download en_core_web_sm
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(self._cleaned_content.lower())
        self._cleaned_content: list[str] = [sent.text for sent in doc.sents]
        self.current_document = self._cleaned_content
        return self

    @property
    def raw_content(self):
        """Access raw HTML content."""
        return self._raw_content

    @property
    def cleaned_content(self):
        """Access cleaned content."""
        return self._cleaned_content

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, value):
        self._embedding = value

    @property
    def current_document(self):
        if isinstance(self._current_document, list):
            return self._current_document
        elif isinstance(self._current_document, np.ndarray):
            return self._current_document
        else:
            return [self._current_document]

    @current_document.setter
    def current_document(self, value):
        self._current_document = value

    @property
    def index(self) -> None:
        return self._index

    @index.setter
    def index(self, value):
        self._index = value



    def save(self, folder: str, hashed: bool = False, extension: str  = ".txt"):
        """Save cleaned content to a file."""
        if self.current_document is not None:
            os.makedirs(folder, exist_ok=True)
            folder = Path(folder)
            file_path2 = self.create_file_name_from_path(folder = folder, extension=extension)
            if isinstance(self.current_document, np.ndarray):
                np.savetxt(file_path2, self.current_document, fmt="%.8f")
            else:
                with open(file_path2, 'w', encoding='utf-8') as file:
                    for line in self.current_document:
                        if hashed:
                            line = hashlib.sha256(line.encode('utf-8')).hexdigest()
                        file.write(line + "\n")

            print(f"Content saved to {folder}")
        else:
            print("No content to save.")

        return self

    def calc_embedding(self, model: Model):
        self.embedding = np.array([model.calc_embeddings(line) for line in self.cleaned_content])
        self.current_document = self.embedding
        return self

    def add_embedding_to_vector(self):
        if self.index is None and self.embedding is not None:
            self.index = faiss.IndexFlatL2(self.embedding.shape[1])
        else:
            RuntimeError("Embedding must be calculated")

        self.index.add(self.embedding)
        return self


