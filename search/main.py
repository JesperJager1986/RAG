import pandas as pd
from sentence_transformers import SentenceTransformer

from search.model import Model
from search.web_scraping_pipeline import WebScrapingPipeline
from search.webpages import get_urls

if __name__ == "__main__":

    model_name = 'all-MiniLM-L6-v2'
    text = "This is an example sentence to be embedded."

    Model = Model(model_name=model_name)


    for url in get_urls():
        pipeline = WebScrapingPipeline(url)
        (pipeline.fetch().
         format().
         save(folder="format").
         preprocess_text().
         save(folder="preprocessed").
         calc_embedding(Model).
         save(folder="embedded"))


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
