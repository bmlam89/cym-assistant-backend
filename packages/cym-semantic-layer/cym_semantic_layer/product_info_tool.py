from typing import Dict, List, Any, Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

# Import things that are needed generically
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, tool

import json
from langchain_community.document_loaders import JSONLoader
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def _load_product_documents():
	"""Load products data into documents used for retrieval"""
	path = 'products/products.json'
	loader = JSONLoader(
        file_path=path,
        jq_schema='.data[]',
        text_content=False
    )
	documents = loader.load()
	return documents

def _get_cached_embedder():
	"""Get the embedder from cached embeddings for all products"""
	path = './products/embeddings'
	store = LocalFileStore(path)
	cached_embedder = CacheBackedEmbeddings.from_bytes_store(
		OpenAIEmbeddings(), 
		store, namespace=OpenAIEmbeddings().model
	)
	return cached_embedder

@tool
def get_most_relevant_products(input: str, number_of_chunks: int = 3) -> List[Dict[str, Any]]:
	"""
	Get the k most relevant product documents based on the input string.

	This function loads locally cached product json data into documents, creates 
	cached embedder from locally cached product embeddings, creates a FAISS vector database,
	creates a retriever from the vector database, and retrieves the k most relevant product
	documents based on the input string.
	"""

	documents = _load_product_documents()
	embedder = _get_cached_embedder()
	vector_database = FAISS.from_documents(documents, embedder)
	retriever = vector_database.as_retriever(k=number_of_chunks)
	documents = retriever.get_relevant_documents(input)
	return documents


class ProductInfoInput(BaseModel):
	input: str = Field(description="User query to pass into product information tool")

class ProductInfoTool(BaseTool):
	name="ProductInfo"
	description = (
		"useful for getting contextual data to answer user questions"
	)
	args_schema: Type[BaseModel] = ProductInfoInput

	def _run(
			self,
			input: str,
			run_manager: Optional[CallbackManagerForToolRun] = None
	):
		"""Use the tool."""
		return get_most_relevant_products(input)
	async def _arun(
			self,
			input: str,
			run_manager: Optional[AsyncCallbackManagerForToolRun] = None
	):
		"""Use the tool asynchronously."""
		return get_most_relevant_products(input)