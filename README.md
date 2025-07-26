# 🛠️ Setup Guide
1. **Clone the Repository**
  ```bash
   https://github.com/marziasu/chatbot-rag-.git
   cd chatbot-rag-
  ```

2. **Create a Virtual Environment**
  ```bash
    python -m venv .venv
    .venv\Scripts\activate  # Linux: source .venv/bin/activate
  ```

3. **install Required Libraries**
  
  ```bash
    pip install -r requirements.txt
  ```
4. **Create a .env File**
Create a .env file in the project root directory and add your API keys:

  ```bash
    OPENAI_API_KEY=your-openai-api-key
    PINECONE_API_KEY=your-pinecone-api-key
  ```

5. **Run the App**
  ```bash
  uvicorn app.main:app
  ```

---

# 🧰 Tools, Libraries, and Packages Used

```pgsql
PyMuPDF (fitz)              → Extract images from scanned Bangla PDF pages
pytesseract + PIL.Image     → OCR engine to extract Bangla text from PDF page images
unicodedata + re            → Unicode normalization and cleaning of Bangla text
langchain                   → Document chunking, prompting, retrieval, and orchestration
langchain_openai            → Embedding generation using OpenAI models (e.g., text-embedding-3-large)
Pinecone + langchain_pinecone → Vector database for storing and retrieving document chunks
FastAPI                     → API framework for serving routes (start chat, end chat, history, ask, evaluate, etc.)
uvicorn                     → ASGI server to host the FastAPI backend
NumPy                       → Vector math for cosine similarity in evaluation
OpenAI API (gpt-4)          → LLM generation using GPT-4
os, hashlib, uuid           → File system access, text hashing, unique ID generation

```

---

# 🌐 Deployment
 
The QA Chatbot is deployed and accessible at:

👉 [https://chatbot-rag-mthf.onrender.com](https://chatbot-rag-mthf.onrender.com)

📘 API Docs: [https://chatbot-rag-mthf.onrender.com/docs](https://chatbot-rag-mthf.onrender.com/docs)

---

# 💬 Sample Queries and Outputs
```bash
English

Query: "Who is Anupam’s legal guardian?"
Output: "According to the document, Anupam refers to his maternal uncle as his legal guardian."

বাংলা

প্রশ্ন: "কল্যাণীর প্রকৃত বয়স কত ছিল বিয়ের সময়?"
উত্তর: "১৫ বছর"
```

---

# 📑 API Documentation 
## 🟢 `GET /QA-chatbot/start-chat`
Starts a new chat session.

**Response**
```json
{
  "session_id": "vJEW5QlUQDWmb1-LGQj0Dw",
  "message": "New chat session started."
}
```
note: # when a chat will be start then it will be call so that history can stored based on this session id. 

## 🔴 ` DELETE /QA-chatbot/end-chat/{session_id}`

Ends and deletes a specific chat session manually.  
`📌 Note: Call this when the user ends the chat to delete their session and its memory.`

## 🟡 ` POST /QA-chatbot/ask`
**Request**
```json
{
  "session_id": "Hyj1gdNjQ5elZzdg2HQ1Bw",
  "question": "who is anupam?"
}         
```    
**Response**
```json
{
  "session_id": "Hyj1gdNjQ5elZzdg2HQ1Bw",
  "question": "who is anupam?",
  "answer": "Anupam is a character from a story. He is described as helpless and personality-less, despite being highly educated. He is often busy obeying his mother's orders, which prevents the development of his independent personality. He is compared to Kartikeya, the younger brother of Ganesha, in the story. After his father's death, his uncle takes over the responsibilities of their family. Towards the end of the story, Anupam manages to break free from the constraints imposed by his mother and uncle."
}
```
**Request**
```json
{
  "session_id": "Hyj1gdNjQ5elZzdg2HQ1Bw",
  "question": "তার ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
} 
```    
**Response**
```json
{
  "session_id": "Hyj1gdNjQ5elZzdg2HQ1Bw",
  "question": "তার ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
  "answer": "শস্তুনাথবাবুকে সুপুরুষ বলা হয়েছে।"
} 
```
note: this is for chatting. for everytime, when session id is same and any chat will create then internally saved message in short term memory (in dictionary)

## 🔁 `POST /QA-chatbot/history`

Fetches chat history for a given session.

**Query Parameter**
- `session_id`: `LoA60icTQLeuxmuMn_KY0Q`
**Response**
```json
{
  "session_id": "LoA60icTQLeuxmuMn_KY0Q",
  "history": [
    {
      "question": "কাকে অনুপমের ভাগ্যদেবতা বলে উল্লেখ করা হয়েছে?",
      "answer": "অনুপমের মামা।"
    },
    {
      "question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
      "answer": "১৬ বছর"
    }
  ]
}
```
note:

## 📊 ` POST /QA-chatbot/evaluate`
**Request**
```json
{
  "data": [
  {
    "question": "কাকে অনুপমের ভাগ্যদেবতা বলে উল্লেখ করা হয়েছে?",
    "expected_answer": "মামা"
  },
  {
    "question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
    "expected_answer": "১৫ বছর"
  }
  ]
}
```
**Response**
```json
{
  "average_relevance": 0.7842567456519585,
  "average_groundedness": 0.3772497844742908,
  "results": [
    {
      "question": "কাকে অনুপমের ভাগ্যদেবতা বলে উল্লেখ করা হয়েছে?",
      "expected_answer": "মামা",
      "generated_answer": "অনুপমের মামা।",
      "relevance_score": 0.6956699414476105,
      "groundedness_score": 0.4671823197282297
    },
    {
      "question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
      "expected_answer": "১৫ বছর",
      "generated_answer": "১৬ বছর",
      "relevance_score": 0.8728435498563065,
      "groundedness_score": 0.28731724922035196
    }
  ]
} 
```

---

### ❓ What method or library did you use to extract the text, and why?

**Used:** `PyMuPDF (fitz)` and `pytesseract` (with `PIL.Image`)  
**Why:**  
- The PDF contained **scanned Bangla pages**, so the text was **not digitally encoded**, only image-based.
- `PyMuPDF` was used to extract each page as an image.
- `pytesseract` (OCR engine) was applied to those images to extract Bangla text.
- This approach was necessary because libraries like `pdfminer` or `PyPDF2` only work with selectable/digital text and cannot extract from image-based documents.

---

### ⚠️ Did you face any formatting challenges with the PDF content?

**Yes.** Common issues included:

- ❌ **Broken or joined Bangla characters** due to OCR inaccuracies.
- ❌ **Irregular spacing and line breaks** from OCR output.
- ❌ **Presence of non-Bangla characters** and noise from scanned images.

### ✅ Solution:

A custom cleaning function `clean_bangla_text()` was developed to post-process the OCR output:

```python
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
```
---
### ✂️ What chunking strategy did you choose? Why?

**Chosen Strategy:**  
Custom sentence-aware chunking using LangChain’s `RecursiveCharacterTextSplitter` with overlap and Bengali-specific separators.

**Implementation Highlights:**
- After OCR and cleaning, the entire document text is concatenated.
- Then split using `RecursiveCharacterTextSplitter` with the following config:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=50,
    separators=["\n", "।", ".", " "]
)
```

---

### 🧬 What embedding model did you use? Why?

**Used Model:** `text-embedding-3-large`  
(via `OpenAIEmbeddings` from `langchain_openai`)

```python
from langchain_openai.embeddings import OpenAIEmbeddings

self.embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)
```
📝 Additional Context

Initially, I tried using the smaller OpenAI model `text-embedding-3-small` for embeddings, but it did not provide accurate answers or relevant retrieval results for this multilingual (English-Bangla) dataset.

I also experimented with multimodal embedding models `sentence-transformers/clip-ViT-B-32-multilingual-v1` from Hugging Face, but those models struggled with Bangla characters and failed to deliver reliable semantic similarity in the Bangla language.

---

### 🔗 How are you comparing the query with stored chunks?

When a user submits a query, the system performs the following steps:

1. **Retrieve Relevant Documents:**

```python
docs = self.retriever.invoke(question)
```
Method: Uses Pinecone vector database to perform approximate nearest neighbor (ANN) search based on vector similarity  

Why:

  - Pinecone is a managed vector database optimized for fast and scalable similarity search.

  - It supports real-time indexing and querying of high-dimensional embeddings.

  - Provides low latency retrieval for large-scale datasets.

Storage Setup:

  - Document chunks are embedded and stored as vectors in the Pinecone index.

  - At query time, the query is embedded using the same embedding model.

  - Pinecone returns the top-k most similar chunks based on cosine similarity scores.

This allows efficient and relevant retrieval of context to augment the LLM’s responses.


---

### 🤖 How do you ensure meaningful comparison between query and chunks?

Approach:

- Both queries and chunks are embedded using the same model.


If the query is vague or missing context:

It may return irrelevant or generic chunks.
Solution: Add query reformulation or prompt clarification UI.



---

### 🎯 Do the results seem relevant?

Overall, the retrieval and answer generation perform well, providing mostly relevant responses. However, there is room for improvement:

- **Enhanced chunking strategies:**  
  Refining chunking methods specifically tailored for Bangla-heavy documents can improve context coherence and retrieval accuracy.

- **More specialized embedding models:**  
  Using larger or domain-specific embedding models fine-tuned on Bangla language or the specific subject matter could boost semantic understanding.

- **Improved OCR post-processing:**  
  Since the source PDFs are scanned, advanced OCR error correction and text normalization can reduce noise and improve text quality before embedding.

- **Query reformulation and prompt engineering:**  
  Adding techniques to better interpret vague or ambiguous queries can further enhance relevance.

- **Re-ranking with GPT or other models:**  
  Applying a re-ranking step on retrieved chunks using GPT-based scoring can improve answer precision.

---

## 📌 Future Improvements

Here are potential future improvements based on current limitations and insights:

- 🔍 **Advanced OCR Integration**
  - Integrate Tesseract with improved preprocessing (denoising, binarization) or switch to a Transformer-based OCR model like [TrOCR](https://huggingface.co/microsoft/trocr-base-printed) for better Bangla text recognition.
  - Add automatic page-wise OCR logging and correction for low-confidence regions.

- 🌐 **Fine-Tuned Multilingual Embeddings**
  - Fine-tune a multilingual model (e.g., `LaBSE`, `distiluse-base-multilingual-cased`) on Bangla-English corpora to enhance semantic similarity.
  - Experiment with Hugging Face models like `ai4bharat/indic-sbert` for Bangla-specific retrieval.

- 🧹 **Improved Text Cleaning**
  - Enhance the `clean_bangla_text()` function with a more robust Bangla spell checker (e.g., `bnltk`, `bangla-stemmer`).
  - Introduce language model-based error correction for misrecognized OCR characters.

- 📚 **Hybrid Retrieval Setup**
  - Combine dense vector retrieval (Pinecone) with sparse keyword-based retrieval (BM25 or Elasticsearch) to handle both semantic and lexical matching.

- 🧩 **Chunk-Level Metadata Storage**
  - Store metadata such as source page number, position, and confidence score for better debugging and user explanation.

---


👩‍💻 Author

Marzia Sultana
