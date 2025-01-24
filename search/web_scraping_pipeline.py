import hashlib

import faiss
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from deeplake.core.vectorstore import DeepLakeVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import requests
import os
import spacy
from search.webpages import get_urls
from time import sleep
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path


class WebScrapingPipeline:
    def __init__(self, url: str |  Path, storing_directory: str | Path):
        self.url = Path(url)
        self._storing_directory = Path(storing_directory)
        self._raw_content = None
        self._cleaned_content = None

    def get_file_name(self):
        return self.url.name

    def create_file_name_from_path(self, add_name_to_name: str = None, extension =".txt") -> Path:
        if add_name_to_name is not None:
            new_stem = f"{self.url.stem}_{add_name_to_name}"
            self.url = self.url.with_name(new_stem + self.url.suffix)
        path = self.storing_directory / self.url.stem

        path = path.with_suffix(extension)
        return path

    def fetch(self):
        """Fetch raw HTML content, falling back to Selenium if requests is unsuccessful."""

        try:
            response = requests.get(str(self.url), timeout=10)
            response.raise_for_status()
            self._raw_content = response.content
        except requests.exceptions.RequestException as e:
            print(f"Requests failed: {e}")

        if self._raw_content is None:
            self._fetch_with_selenium()

        if self._raw_content is None:
            raise RuntimeError("No data is loaded")

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

        return self

    def preprocess_text(self):
        # python - m spacy download en_core_web_sm
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(self._cleaned_content.lower())
        self._cleaned_content: list[str] = [sent.text for sent in doc.sents]
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
    def storing_directory(self):
        return self._storing_directory

    def save(self, file_path: str):
        """Save cleaned content to a file."""
        if self._cleaned_content:
            os.makedirs(file_path, exist_ok=True)
            file_path2 = self.create_file_name_from_path()
            file_path3 = self.create_file_name_from_path("hashed")
            with open(file_path2, 'w', encoding='utf-8') as file:
                for line in self.cleaned_content:
                    file.write(line + "\n")

            with open(file_path3, 'w', encoding='utf-8') as file:
                for line in self.cleaned_content:
                    hashed_line = hashlib.sha256(line.encode('utf-8')).hexdigest()
                    file.write(hashed_line + "\n")

            print(f"Content saved to {file_path}")
        else:
            print("No content to save.")

        return self



if __name__ == "__main__":
    model = SentenceTransformer('all-MiniLM-L6-v2')
    text = "This is an example sentence to be embedded."
    embedding = model.encode(text)

    store_cleaned_data = "./cleaned_data/"
    for url in get_urls():
        pipeline = WebScrapingPipeline(url, store_cleaned_data)
        pipeline.fetch().format().preprocess_text().save(store_cleaned_data)

    documents = SimpleDirectoryReader(store_cleaned_data).load_data()
    df = pd.DataFrame([], columns=["sentence", "embedding"])
    for document in documents:
        sentences = document.text.split(".")
        for sentence in tqdm(sentences):
            sentence = sentence.replace("\n", "")

            embedding = model.encode(sentence)
            embedding_s = embedding.size
            new_row = pd.DataFrame({'sentence': [sentence], 'embedding': [embedding]})

            # Using pd.concat() to append
            df = pd.concat([df, new_row], ignore_index=True)
            sleep(1)
            print(2)
        index = faiss.IndexFlatL2(embedding_s)
        embedding = df["embedding"].to_numpy()



    index = VectorStoreIndex.from_documents(documents)
    # Set up local paths
    base_path = "./dataset_db/"
    os.makedirs(base_path, exist_ok=True)  # Create the directory if it doesn't exist

    vector_store_path = os.path.join(base_path, "vector_store")
    dataset_path = os.path.join(base_path, "dataset/")

    # Initialize the vector store locally
    # vector_store = DeepLakeVectorStore(path=dataset_path, overwrite=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create an index over the documents
    # index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    print(2)
