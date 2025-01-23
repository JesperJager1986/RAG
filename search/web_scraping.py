import re
import os
from search.webpages import get_urls
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, StorageContext
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import requests

def remove_sections_batch(content, section_titles):
    for section in content.find_all('span', id=lambda x: x in section_titles):
        parent = section.parent
        for sibling in parent.find_next_siblings():
            sibling.decompose()
        parent.decompose()
    return content

def clean_text(content):
    # Remove references and unwanted characters
    content = re.sub(r'\[\d+\]', '', content)   # Remove references
    content = re.sub(r'[^\w\s\.]', '', content)  # Remove punctuation (except periods)
    return content



def fetch_page_content(url: str):
    content = None  # Initialize a variable to hold the extracted content

    try:
        # Attempt with requests
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        content = soup.find('div', {'class': 'mw-parser-output'}) or soup.find('div', {'id': 'content'})
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page with requests: {e}")

    if not content:
        try:
            # Fallback to Selenium for JavaScript rendering
            print("Falling back to Selenium for JavaScript rendering...")
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")

            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
            driver.get(url)
            content = BeautifulSoup(driver.page_source, 'html.parser')
            driver.quit()
        except Exception as e:
            print(f"Error fetching page with Selenium: {e}")
            content = None  # Ensure content is explicitly set to None on failure

    return content



def fetch_and_clean(url: str):
    content = fetch_page_content(url)

    if content:
        content = remove_sections_batch(content, ['References', 'Bibliography', 'External links', 'See also', 'Notes'])
        text = content.get_text(separator=' ', strip=True)
        text = clean_text(text)
    else:
        text = None
        print(f"Could not get content from page: {url}")
    return text



if __name__ == "__main__":

    urls = get_urls()
    output_dir = './data/'  # More descriptive name
    os.makedirs(output_dir, exist_ok=True)
    if True:
        for url in urls:
            article_name = url.split('/')[-1].replace('.html', '')  # Handle .html extension
            filename = os.path.join(output_dir, f"{article_name}.txt")

            clean_article_text = fetch_and_clean(url)
            if clean_article_text:  # Only write to file if content exists
                with open(filename, 'w', encoding='utf-8') as file:
                    file.write(clean_article_text)
                    print(filename)
            else:
                print(f"Could not get article text from {url}")

    print(f"Content(ones that were possible) written to files in the '{output_dir}' directory.")
    documents = SimpleDirectoryReader("./data/").load_data(show_progress=True)

    # Set up local paths
    base_path = "./dataset_db/"
    os.makedirs(base_path, exist_ok=True)  # Create the directory if it doesn't exist

    vector_store_path = os.path.join(base_path, "vector_store")
    dataset_path = os.path.join(base_path, "dataset")

    # Initialize the vector store locally
    vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create an index over the documents
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    print(2)
