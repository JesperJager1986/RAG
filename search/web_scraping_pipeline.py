import hashlib

import ast
import numpy as np
import pandas as pd
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

import textwrap

from search.model import Model

def stop_chain_decorator(func):
    def wrapper(self, *args, **kwargs):
        if self.current_document is None :
            print(f"Skipping {func.__name__} because the chain is stopped.")
            return self  # Skip the function if the chain is stopped
        return func(self, *args, **kwargs)
    return wrapper


class WebScrapingPipeline:
    def __init__(self):
        self._raw_content: bytes | None = None
        self._cleaned_content: str | None  = None
        self._embedding: np.ndarray | None = None
        self._current_document: str | list[str] | None = None
        self._index: faiss.Index | None  = None
        self.df = pd.DataFrame([])

    @property
    def raw_content(self) -> bytes | None:
        """Access raw HTML content."""
        return self._raw_content

    @raw_content.setter
    def raw_content(self, raw_content: bytes | None) -> None:
        self._raw_content = raw_content

    @property
    def cleaned_content(self) -> str | None:
        """Access cleaned content."""
        return self._cleaned_content

    @cleaned_content.setter
    def cleaned_content(self, value):
        self._cleaned_content = value

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, value):
        self._embedding = value

    @property
    def current_document(self):
        if self._current_document is None:
            return None
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
    def index(self) -> faiss.Index:
        return self._index

    @index.setter
    def index(self, value) -> None:
        self._index = value


    @staticmethod
    def create_file_name_from_path(path: str | Path, folder , add_name_to_name: str = None, extension =".txt") -> Path:
        if isinstance(path, str):
            path = Path(path)

        if add_name_to_name is not None:
            new_stem = f"{path.stem}_{add_name_to_name}"
            path = path.with_name(new_stem + path.suffix)
        path = folder / path.stem

        path = path.with_suffix(extension)
        return path

    def fetch(self, url: str) -> "WebScrapingPipeline":
        print(f"Fetching {url}")
        """Fetch raw HTML content, falling back to Selenium if requests is unsuccessful."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                #"Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://google.com"  # Some sites may check for a valid referer
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            self.raw_content = response.content
        except requests.exceptions.RequestException as e:
            print(f"Requests failed: {e}")

        if self.raw_content is None:
            self._fetch_with_selenium(url)

        self.current_document = self.raw_content

        return self

    def _fetch_with_requests(self, url: str | Path) -> "WebScrapingPipeline":
        """Try to fetch content using requests."""
        try:
            response = requests.get(str(url), timeout=10)
            response.raise_for_status()
            self.raw_content = response.content
        except requests.exceptions.RequestException as e:
            print(f"Requests failed: {e}")
        return self

    def _fetch_with_selenium(self, url: str | Path):
        """Try to fetch content using selenium."""
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
                driver.get(str(url))
                self.raw_content = driver.page_source
                self.current_document = self.raw_content
        except Exception as e:
            print(f"Selenium failed: {e}")
            self.raw_content = None
        return self

    @stop_chain_decorator
    def extract(self) -> "WebScrapingPipeline":
        """Extract meaningful content from raw HTML."""
        soup = BeautifulSoup(self.raw_content, 'html.parser')

        all_divs = soup.find_all('div')
        hest = {div.get_text(separator=" ", strip=True) for div in all_divs if div.get_text(strip=True)}
        result = ". ".join(hest)
        self.cleaned_content =  result if result else None
        self.current_document = self.cleaned_content
        return self

    @staticmethod
    def sliding_window(lst: list[str], window_size: int, step: int = 1) -> list[str]:
        """Creates a sliding window of chunks from the list of strings.

        Args:
            lst: List of strings.
            window_size: Size of each window (chunk).
            step: Step size for sliding.

        Returns:
            A list of lists containing the chunks.
        """
        return [" ".join(lst[i:i + window_size]) for i in range(0, len(lst) - window_size + 1, step)]

    def remove_special_character(self):
        nlp = spacy.load("en_core_web_sm")

        doc = nlp(self.current_document[0].lower()) if self.current_document is not None else None

        self.cleaned_content: list[str] = [sent.text for sent in doc.sents]
        self.current_document = self.cleaned_content
        return self


    @stop_chain_decorator
    def preprocess_text(self, chunk: int):

        self.cleaned_content = self.sliding_window(self.current_document, window_size=chunk)

        self.current_document = self.cleaned_content
        return self

    @stop_chain_decorator
    def save(self, path, folder: str, hashed: bool = False, extension: str  = ".cvs", info: bool = False):
        """Save cleaned content to a file."""
        os.makedirs(folder, exist_ok=True)
        folder = Path(folder)
        file_path = self.create_file_name_from_path(path, folder = folder, extension=extension)

        document = self.current_document

        if hashed:
            document = list(map(lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest(), document))

        if isinstance(document, np.ndarray):
            np.savetxt(file_path, document, fmt="%.8f")
            print(f"Content saved to {folder}")
        else:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(f"{line}\n" for line in document)
            print(f"Content saved to {folder}")

        if info:
            df = pd.DataFrame({
                "text": document,
                "embedding": [embedding.tolist() for embedding in self.embedding]  # Convert numpy arrays to lists
            })
            df.to_csv(file_path, index=False)
        return self

    @stop_chain_decorator
    def calc_embedding(self, model: Model, ):
        self.embedding = np.array([model(line) for line in self.cleaned_content])
        self.current_document = self.embedding
        return self

    @stop_chain_decorator
    def add_embedding_to_vector(self):
        if self.index is None and self.embedding is not None:
            self.index = faiss.IndexFlatL2(self.embedding.shape[1])
        else:
            RuntimeError("Embedding must be calculated")

        self.index.add(self.embedding)  # noqa
        return self

    @staticmethod
    def __print_with_width(text, width=80):
        # Wrap the text to fit within the specified width
        wrapped_text = textwrap.fill(text, width=width)
        print(wrapped_text)

    def __call__(self, text, model_name):
        model = Model(model_name=model_name)
        query_embedding = model(text).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, 1)  # noqa   (handler error)
        for idx in indices:
            result = str(self.df["text"].loc[idx])
            self.__print_with_width(result)

    def info(self) -> None:
        print(f"library size: {len(self.df)}")

    def load(self, file_path: str | Path):

        df = pd.read_csv(file_path)

        self.current_document = df.values.tolist()
        self.embedding = np.array([ast.literal_eval(item) for item in df["embedding"].values])

        return self

    def store_to_df(self, title, hash_data: bool = False) :

        document = self.current_document

        if hash_data:
            document = list(map(lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest(), document))

        if isinstance(document, np.ndarray):
            document = [embedding for embedding in document]

        if title in self.df.columns:
            self.df[title] = pd.concat([self.df[title], pd.Series(document)], ignore_index=True)
        else:
            self.df[title] = document

        return self

    def save_df(self, url, folder, extension: str = ".csv"):
        os.makedirs(folder, exist_ok=True)
        folder = Path(folder)
        file_path = self.create_file_name_from_path(url, folder = folder, extension=extension)

        self.df.to_csv(file_path, index=False)
        return self