from search.model import Model
from search.web_scraping_pipeline import WebScrapingPipeline
from search.webpages import get_urls

if __name__ == "__main__":

    model_name = 'all-MiniLM-L6-v2'
    text = "how cars are there in each image"

    pipeline = WebScrapingPipeline()
    for url in get_urls():
        (pipeline.
         fetch(url).
         format().
         #save(folder="format").
         preprocess_text().
         #save(folder="preprocessed").
         load().
         store_in_pd_library().
         calc_embedding(model=Model(model_name=model_name)).
         save(folder="embedded").
         add_embedding_to_vector())

    pipeline.info()
    pipeline(text, model_name)




