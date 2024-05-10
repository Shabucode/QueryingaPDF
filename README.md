# QueryingaPDF

## Querying PDFs with Langchain's OpenAI Model and OpenAI Embedding
This project demonstrates how to utilize Langchain's OpenAI model and OpenAI embedding for querying PDF documents. Additionally, it incorporates Astra DB (Cassandra) as a vector store for efficient retrieval of documents based on semantic similarity.

## Overview
This project leverages Langchain's OpenAI model and OpenAI embedding to generate embeddings for PDF documents. These embeddings capture the semantic meaning of the documents, enabling efficient querying based on content similarity.

## Features
PDF Querying: Extracts text content from PDF documents and generates embeddings using Langchain's OpenAI model.
OpenAI Embedding: Utilizes OpenAI embedding to represent documents as dense vectors in a high-dimensional space.
Astra DB (Cassandra): Stores document embeddings in Astra DB for fast and scalable similarity-based retrieval.
