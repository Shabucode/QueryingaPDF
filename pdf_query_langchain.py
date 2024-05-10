

!pip install cassio tiktoken langchain datasets openai

#Langchain components to use
from langchain import.vectorstores.cassandra Cassandra  #Cassandra vector db
from langchain.indexes.vectorstore import VectorStoreIndexWrapper #vector index store
from langchain.llms import OpenAI #LLM - Openai
from langchain.embeddings import OpenAIEmbeddings #Embeddings - Openaiembeddings

#Support for dataset retrieval from Hugging Face
from datasets import load_dataset

#With CassIO the engine powering the Astra DB integration in LangChain,
#Initialize DB connection
import cassio

#import pypdf to deal with pdf documents
!pip install PyPDF2

from PyPDF2 import PdfReader

"""Setup"""

#AStra DB ID,Token and Openai api key
ASTRA_DB_APPLICATION_TOKEN="ASTRA_DB_TOKEN"
ASTRA_DB_ID="ASTRA_DB_ID"

OPENAI_API_KEY = "paste your openai_api_key"

#provide the path of pdf file/files
pdfreader=PdfReader("specify filepath")

"""Read and extract text from the pdf"""

from typing_extensions import Concatenate
#read text from pdf
raw_test = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text() #page.extract_text will extract all the text from the pages
    if content:
      raw_text += content

raw_text

"""##Initialize your connection to the database

"""

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database=ASTRA_DB_ID) #will get warnings but dont worry on warnings

"""Create Langchain LLM and embeddings objects"""

llm=OpenAI(openai_api_key=OPENAI_API_KEY ) #create instance for llm openai by assigning the openai api key
embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY ) #create instance for openai embeddings using openai api key

"""Create your langchain vector store - cassandra -
backed by Astra DB
"""

astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa_mini_demo", #it will create a table, so have to give table name
    session=None,
    keyspace=None,
)

"""Text Splitter - Dividing entire document into chunks"""

from langchain.text_splitter import CharacterTextSplitter

#we need to split the text using CharacterTextSplitter such that it should not increase token size
text_splitter = ChatacterTextSplitter(
                seperator = '\n',
                chunk_size = 800,
                chunk_overlap = 200,
                length_function=len,
)
texts = text_splitter.split_text(raw_text)

texts[:50] #prints the top 50 characters

"""Load the dataset into th vector store"""

#add texts to the vector store
astra_vector_store.add_texts(texts[:50]) #it takes the first 50 characters and apply the function of
#converting them into vector and store in the vector db

print("Inserted %i headlines", % len(texts[:50])) #%i - referring to %len(...)

#creates the index for converted vectors in astra_vector_store to indexes through VectorStoreIndexManager and stores in
#astra_vector_index
astra_vector_index = VectorStoreIndexWrapper(vectorestore=astra_vector_store)

"""Write the question answer format"""

first_question = True
while True:
    if first_question:
      query_text=input("\nEnter you question (or type quit to exit)").strip()
    else:
        query_text=input("\n What is your next question (or type quit to exit)").strip()

    if query_text.lower == "quit":
      break

    if query_text == "":
      continue

    print("\n QUESTION : \"%s\"" % query_text)

    answer = astra_vector_index.query(query_text, llm=llm).strip() #searches the relevant index then in the vector store db

    print("Answer : \"%s\"\n" %answer)

    print("FIRST DOCUMENTS BY RELEVANCE :")
    for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
      print("   [%0.4f] \"%s.....\"" % (score, doc.page_content[:84]))

