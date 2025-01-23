import os
import tomllib
from dotenv import load_dotenv
from call_model import call_llm_with_full_text
from print_format import print_formatted_response
from simularity.simularity import calculate_cosine_similarity, find_best_match_keyword_search, calculate_enhanced_similarity
from db import get_db
from simularity.index_search import setup_vectorizer, find_best_match

QUERY = "define a rag store"

def read_config_file(path: str) -> dict[str, any]:
    with open(path, "rb") as file:
        config = tomllib.load(file)
    return config


def set_api_for_model_provider(model: str) -> None:
    if model == "gpt-3.5-turbo":
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        raise Exception(f"Unsupported model: {model}")

def get_client(model: str):
    if model == "gpt-3.5-turbo":
        from openai import OpenAI
        client = OpenAI()
    else:
        raise Exception(f"Model {model} not supported")
    return client


if __name__ == '__main__':
    load_dotenv()
    config =  read_config_file("config.toml")

    db: list[str] = get_db()

    best_keyword_score, best_matching_record = find_best_match_keyword_search(QUERY, db)

    score = calculate_cosine_similarity(QUERY, best_matching_record)
    score_enhanced = calculate_enhanced_similarity(QUERY, best_matching_record)

    vectorizer, tfidf_matrix = setup_vectorizer(db)
    best_similarity_score, best_index = find_best_match(QUERY, vectorizer, tfidf_matrix)

    best_matching_record = db[best_index]

    print_formatted_response(best_matching_record)

    set_api_for_model_provider(config["model"])
    client = get_client(config["model"])

    llm_response = call_llm_with_full_text(QUERY, client, config['model'])
    print_formatted_response(llm_response)
    print(2)