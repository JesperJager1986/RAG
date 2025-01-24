import pandas as pd
from llama_index.core import SimpleDirectoryReader
from sentence_transformers import SentenceTransformer

from search.web_scraping_pipeline import WebScrapingPipeline
from search.webpages import get_urls

if __name__ == "__main__":

    for url in get_urls():
        pipeline = WebScrapingPipeline(url, "./data_cleaned")
        pipeline.fetch().format().preprocess_text().save()

    for document in documents:
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
