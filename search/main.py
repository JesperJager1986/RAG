from torch.nn.functional import embedding

from search.model import Model
from search.web_scraping_pipeline import WebScrapingPipeline
from search.webpages import get_urls

if __name__ == "__main__":

    model_name = 'all-MiniLM-L6-v2'
    text = "This is an example sentence to be embedded."

    pipeline = WebScrapingPipeline()
    for url in get_urls():
        (pipeline.fetch(url).
         format().
         save(folder="format").
         preprocess_text().
         save(folder="preprocessed").
         calc_embedding(model=Model(model_name=model_name)).
         save(folder="embedded").
         add_embedding_to_vector())

    model = Model(model_name=model_name)
    query_embedding = model.calc_embeddings(text)

    distances, indices = pipeline.index.search(query_embedding, 5)

    print(2)

