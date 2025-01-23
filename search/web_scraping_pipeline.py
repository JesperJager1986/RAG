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

class WebScrapingPipeline:
    def __init__(self, url: str):
        self.url = url
        self._raw_content = None
        self._cleaned_content = None

    def get_file_name(self, output_dir) -> str:
        article_name = self.url.split('/')[-1].replace('.html', '')  # Handle .html extension
        filename = os.path.join(output_dir, f"{article_name}.txt")
        return filename

    def fetch(self):
        """Fetch raw HTML content."""
        try:
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()
            self._raw_content = response.content
        except requests.exceptions.RequestException as e:
            print(f"Requests failed: {e}")
            self._raw_content = None

        # Fallback to Selenium if requests fails
        if self._raw_content is None:
            try:
                print("Falling back to Selenium for JavaScript rendering...")
                chrome_options = Options()
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--disable-gpu")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")

                driver = webdriver.Chrome(
                    service=Service(ChromeDriverManager().install()), options=chrome_options
                )
                driver.get(self.url)
                self._raw_content = driver.page_source
                driver.quit()
            except Exception as e:
                print(f"Selenium failed: {e}")
                self._raw_content = None

        return self  # Enable method chaining

    def format(self):
        """Extract meaningful content from raw HTML."""
        if not self._raw_content:
            print("No content to clean.")
            self._cleaned_content = None
        else:
            soup = BeautifulSoup(self._raw_content, 'html.parser')
            content = (
                soup.find('div', {'class': 'mw-parser-output'}) or
                soup.find('div', {'id': 'content'})
            )
            self._cleaned_content = content.get_text(strip=True) if content else None
            if not self._cleaned_content:
                print("No meaningful content found.")

        return self  # Enable method chaining

    def preprocess_text(self):
        # python - m spacy download en_core_web_sm
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(self._cleaned_content.lower())
        self._cleaned_content = [sent.text for sent in doc.sents]
        return self

    @property
    def raw_content(self):
        """Access raw HTML content."""
        return self._raw_content

    @property
    def cleaned_content(self):
        """Access cleaned content."""
        return self._cleaned_content

    def save(self, file_path: str):
        """Save cleaned content to a file."""
        if not self._cleaned_content:
            print("No content to save.")
            return self  # Allow chaining even if no content is saved

        os.makedirs(file_path, exist_ok=True)
        file_path2 = self.get_file_name(file_path)
        with open(file_path2, 'w', encoding='utf-8') as file:
            for line in self.cleaned_content:
                file.write(line + "\n")

        print(f"Content saved to {file_path}")
        return self  # Enable method chaining



if __name__ == "__main__":
    model = SentenceTransformer('all-MiniLM-L6-v2')
    text = "This is an example sentence to be embedded."
    embedding = model.encode(text)

    for url in get_urls():
        pipeline = WebScrapingPipeline(url)
        pipeline.fetch().format().preprocess_text().save("./data/")

    documents = SimpleDirectoryReader("./data/").load_data()
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
    index.add(


    index = VectorStoreIndex.from_documents(documents)
    # Set up local paths
    base_path = "./dataset_db/"
    os.makedirs(base_path, exist_ok=True)  # Create the directory if it doesn't exist

    vector_store_path = os.path.join(base_path, "vector_store")
    dataset_path = os.path.join(base_path, "dataset/")

    # Initialize the vector store locally
    vector_store = DeepLakeVectorStore(path=dataset_path, overwrite=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create an index over the documents
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    print(2)
