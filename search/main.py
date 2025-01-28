from search.model import Model
from search.web_scraping_pipeline import WebScrapingPipeline
from search.webpages import get_urls, get_files_folder
from typing import Literal


if __name__ == "__main__":

    model_name = 'all-MiniLM-L6-v2'
    text = "how many cars are there in each image"
    #text = "why do we prune roses"
    load_from_cvs = True
    Topic: Literal["drones", "roses"] = "roses"


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
         format().
         save(url, folder="format").
         preprocess_text().
         save(url, folder="preprocessed").
         store_in_pd_library().
         calc_embedding(model=Model(model_name=model_name)).
         save(url, folder="embedded").
         add_embedding_to_vector().
         save(url, folder="combined", info="all"))


    pipeline.info()
    pipeline(text, model_name)




