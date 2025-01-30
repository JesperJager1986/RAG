from search.model import Model
from search.web_scraping_pipeline import WebScrapingPipeline
from search.webpages import get_urls, get_files_folder
from typing import Literal


if __name__ == "__main__":

    model_name = 'all-MiniLM-L6-v2'
    # text = "how many cars are there in each image"
    # text = "why do we prune roses"
    text = "when should one prune a rose"
    load_from_cvs = True
    Topic: Literal["drones", "roses"] = "roses"
    chunk: int = 3


    pipeline = WebScrapingPipeline()
    if load_from_cvs:
        for file in get_files_folder(folder_path="combined"):
            (pipeline.
             load(file).
             add_embedding_to_vector())
    else:
        for url in get_urls():
            (pipeline.
             fetch(url).
             extract().
             remove_special_character().
             preprocess_text(chunk=chunk).
             store_to_df(title="text").
             store_to_df(title="text_hashed", hash_data=True).
             calc_embedding(model=Model(model_name=model_name)).
             store_to_df(title = "embedding").
             save_df(url, folder="combined").
             add_embedding_to_vector().
             info())

    pipeline.info()
    pipeline(text, model_name)




