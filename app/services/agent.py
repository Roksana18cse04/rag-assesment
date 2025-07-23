import os
from langchain.document_loaders import UnstructuredPDFLoader,  PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
from app.config import PINECONE_API_KEY, OPENAI_API_KEY, DATA_DIR
from uuid import uuid4
import hashlib
import unicodedata
import pdfplumber

def generate_stable_id(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()
import re

def clean_bangla_text(text):
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize Unicode (important for Bengali modifiers)
    text = unicodedata.normalize('NFKC', text)

    # Remove unwanted non-Bangla characters
    text = re.sub(r'[^\u0980-\u09FF\s.,?!ঃ]', '', text)

    # Fix broken conjuncts (e.g., remove space before "্")
    text = text.replace('্ ', '্')
    
    return text.strip()


class Chatbot:
    def __init__(
        self,
        pinecone_api_key: str,
        openai_api_key: str,
        index_name: str = "chatbot-index",
        embedding_model: str = "intfloat/multilingual-e5-base",
        llm_model: str = "gpt-4",
        pinecone_region: str = "us-east-1",
        pinecone_cloud: str = "aws",
        retriever_k: int = 3,
        temperature: float = 0,
    ):
        # Set environment variable for OpenAI
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name

        # Create index if not exists
        if index_name not in [i.name for i in self.pc.list_indexes()]:
            self.pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud=pinecone_cloud, region=pinecone_region),
            )

        # Initialize embeddings and vectorstore

        self.embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
        self.vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=self.embeddings,
            pinecone_api_key=pinecone_api_key,
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": retriever_k})

        # Initialize LLM
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)

        # Prompt template

        template = """
You are a helpful assistant for a RAG system.
You will receive user queries in Bangla or English.
Use only the retrieved context to answer the question.
If the answer is not found in the context, respond with:
"দুঃখিত, এই প্রশ্নের উত্তর পাওয়া যায়নি।" or "Sorry, I couldn't find the answer to your question."

Always answer in the language of the question.

Context: {context}

Chat History:
{chat_history}

Question: {question}

Answer:"""
        self.prompt = PromptTemplate.from_template(template)

        # Chat history list of (question, answer) tuples
        self.chat_history = []

    def format_docs(self, docs):
        cleaned_texts = []
        for doc in docs:
            cleaned = clean_bangla_text(doc.page_content)
            cleaned_texts.append(cleaned)

        return "\n\n".join(cleaned_texts)

    def format_chat(self):
        if not self.chat_history:
            return ""
        return "\n".join(f"Human: {q}\nAssistant: {a}" for q, a in self.chat_history)

    def get_answer(self, question: str) -> str:
        # Retrieve relevant docs
        docs = self.retriever.invoke(question)
        print("retrival docs---------", docs)
        context = self.format_docs(docs)
        print("similar context-------- ", context)

        # Format full prompt
        full_prompt = self.prompt.format(
            context=context,
            chat_history=self.format_chat(),
            question=question,
        )

        # Get answer from LLM
        response = self.llm.invoke(full_prompt)
        answer = response.content.strip()

        # Update chat history
        self.chat_history.append((question, answer))
        return answer

    def chat_loop(self):
        print("Chatbot is ready! Press Ctrl+C to stop.")
        try:
            while True:
                question = input("\nYou: ").strip()
                if not question:
                    print("Please ask something.")
                    continue

                answer = self.get_answer(question)
                print(f"Bot: {answer}")

        except KeyboardInterrupt:
            print("\nExiting chatbot. Bye!")
        except Exception as e:
            print(f"Error: {e}")

    def insert_docs_to_pinecone(self, filepath):
        loader = UnstructuredPDFLoader(filepath)
        documents = loader.load()
        print("Loaded documents:", documents)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n", "।", ".", " "]
        )
        docs_chunks = text_splitter.split_documents(documents)

        # Actually insert into Pinecone
        self.vectorstore.add_documents(docs_chunks)
        
        print(f"Inserted {len(docs_chunks)} chunks into Pinecone index '{self.index_name}'.")
from PIL import Image
import pytesseract
import fitz

# Usage example
if __name__ == "__main__":
    # bot = Chatbot(
    #     pinecone_api_key=PINECONE_API_KEY,
    #     openai_api_key=OPENAI_API_KEY,
    # )
    # insert context info into pinecone
    data_path = os.path.join(DATA_DIR, "HSC26-Bangla1st-Paper.pdf")
    # bot.insert_docs_to_pinecone(data_path)

    doc = fitz.open(data_path)

    for i in range(len(doc)):
        pix = doc[i].get_pixmap(dpi=300)  # better quality
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang='ben')
        print(text)
    

    # bot.chat_loop()