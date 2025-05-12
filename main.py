import os
import glob
from dotenv import load_dotenv
import gradio as gr
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import plotly.graph_objects as go
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
import kagglehub
import pandas as pd

MODEL = "gpt-4o-mini"
db_name = "vector_db"

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')



# Download latest version of the dataset from Kaggle
# Download dataset from Kaggle
kaggle_path = kagglehub.dataset_download("shivamb/netflix-shows")
csv_path = os.path.join(kaggle_path, "netflix_titles.csv")

# Load CSV into DataFrame
df = pd.read_csv(csv_path)

# Create LangChain Documents from rows
documents = []
for _, row in df.iterrows():
    content = (
        f"Title: {row['title']}\n"
        f"Type: {row['type']}\n"
        f"Director: {row.get('director', '')}\n"
        f"Cast: {row.get('cast', '')}\n"
        f"Country: {row.get('country', '')}\n"
        f"Date Added: {row.get('date_added', '')}\n"
        f"Release Year: {row.get('release_year', '')}\n"
        f"Rating: {row.get('rating', '')}\n"
        f"Duration: {row.get('duration', '')}\n"
        f"Description: {row.get('description', '')}"
    )
    metadata = {
        "show_id": row["show_id"],
        "type": row["type"],
        "release_year": row.get("release_year", ""),
        "country": row.get("country", "")
    }
    documents.append(Document(page_content=content, metadata=metadata))
chunks = documents

print(f"Total number of chunks: {len(chunks)}")

# Put the chunks of data into a Vector Store that associates a Vector Embedding with each chunk
embeddings = OpenAIEmbeddings()

# Delete if already exists
if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

# Create vectorstore
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")

# Let's investigate the vectors
collection = vectorstore._collection
count = collection.count()

sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")

# create a new Chat with OpenAI
llm = ChatOpenAI(temperature=0.7, model_name=MODEL)

# set up the conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# the retriever is an abstraction over the VectorStore that will be used during RAG
retriever = vectorstore.as_retriever()

# putting it together: set up the conversation chain with the GPT 3.5 LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

query = "Who is the director of Midnight Mass?"
result = conversation_chain.invoke({"question": query})
print(result["answer"])

# set up a new conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# Wrapping that in a function
def chat(question, history):
    result = conversation_chain.invoke({"question": question})
    return result["answer"]


# And in Gradio:
view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)