import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search

from sentence_transformers import SentenceTransformer

from flask import Flask
from flask import request

app = Flask(__name__)


def data_embedding_transformer(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text)
    return embeddings


# search function
def strings_ranked_by_relatedness(
        query: str,
        df: pd.DataFrame,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
        top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    # query_embedding_response = openai.Embedding.create(
    #     model=EMBEDDING_MODEL,
    #     input=query,
    # )
    # query_embedding = query_embedding_response["data"][0]["embedding"]

    query_embedding = data_embedding_transformer(query)

    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


def num_tokens(text: str, model: str) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
        query: str,
        df: pd.DataFrame,
        model: str,
        token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    question = f"\n\nQuestion: {query}"
    message = ''
    for string in strings:
        next_article = f'\n{string}\n'
        if (
                num_tokens(message + next_article + question, model=model)
                > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(
        query: str,
        df: pd.DataFrame,
        model: str,
        token_budget: int = 4096 - 500,
        print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about Solana"},
        {"role": "user", "content": message},
    ]
    # print(messages)
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response["choices"][0]["message"]["content"]
    return response_message


EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

openai.api_key = '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'
embeddings_path = r"data_embedding.csv"

DF = pd.read_csv(embeddings_path)
DF['embedding'] = DF['text'].apply(lambda x: data_embedding_transformer(x))


@app.route('/ask', methods=['POST'])
def ask_api():
    text = request.get_json().get("text")
    return ask(query=text, df=DF, model=GPT_MODEL)

@app.route('/', methods=['GET'])
def get():
    return "success"


if __name__ == '__main__':
    app.run(debug=True, port=80, host="0.0.0.0")