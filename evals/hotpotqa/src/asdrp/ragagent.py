from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # read local .env file

import bs4
import requests
from functools import lru_cache

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_web_page(url: str, bs_kwargs: dict | None = None) -> list[Document]:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.text, "html.parser", **(bs_kwargs or {}))
    return [Document(page_content=soup.get_text(), metadata={"source": url})]


urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    "https://lilianweng.github.io/posts/2025-05-01-thinking/",
    "https://lilianweng.github.io/posts/2026-06-24-scaling-laws/",
    "https://lilianweng.github.io/posts/2026-07-04-harness/"
]

docs = [load_web_page(url) for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100,
    chunk_overlap=50,
)
doc_splits = text_splitter.split_documents(docs_list)


@lru_cache(maxsize=1)
def _get_retriever():
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings(),
    )
    return vectorstore.as_retriever()