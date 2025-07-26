
# 🤖 RAG-Powered QA Chatbot (Bangla + English)

A document-based chatbot that extracts answers from scanned Bangla PDFs using OCR, chunking, embedding, and GPT-4 powered LLMs. It supports chat-based querying, evaluation, and history tracking via FastAPI.

---

## 🛠️ Setup Guide

### 1. Clone the Repository
```bash
git clone https://github.com/marziasu/chatbot-rag-.git
cd chatbot-rag-
````

### 2. Create a Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Linux: source .venv/bin/activate
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` File

```dotenv
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
```

### 5. Run the App

```bash
uvicorn app.main:app
```

---

## 🔧 Tools & Libraries

| Tool/Library                               | Purpose                                      |
| ------------------------------------------ | -------------------------------------------- |
| PyMuPDF (fitz)                             | Extract page images from scanned Bangla PDFs |
| pytesseract + PIL                          | OCR engine for Bangla text                   |
| unicodedata + re                           | Clean & normalize OCR text                   |
| langchain                                  | Chunking, retrieval, LLM orchestration       |
| OpenAI Embeddings (text-embedding-3-large) | Convert text to vector                       |
| Pinecone                                   | Vector DB for fast retrieval                 |
| FastAPI                                    | API framework                                |
| NumPy                                      | Vector math                                  |
| GPT-4 via OpenAI API                       | LLM generation                               |

---

## 🌐 Deployment

* 🌍 Live: [https://chatbot-rag-mthf.onrender.com](https://chatbot-rag-mthf.onrender.com)
* 📘 Swagger Docs: [https://chatbot-rag-mthf.onrender.com/docs](https://chatbot-rag-mthf.onrender.com/docs)

---

## 💬 Sample Queries

### English

```bash
Q: "What did the main character inherit after the death of his father?"
A: "According to the document, he inherited financial responsibilities and familial obligations."
```

### বাংলা

```bash
প্রশ্ন: "অনুপমের মায়ের ভূমিকা কী ছিল?"
উত্তর: "তিনি সব সিদ্ধান্ত নিয়ন্ত্রণ করতেন এবং অনুপমের ব্যক্তিত্ব বিকাশে বাধা দিতেন।"
```

---

## 📑 API Endpoints

### 🟢 `GET /QA-chatbot/start-chat`

Starts a new chat session.

```json
{
  "session_id": "vJEW5QlUQDWmb1-LGQj0Dw",
  "message": "New chat session started."
}
```

---

### 🔴 `DELETE /QA-chatbot/end-chat/{session_id}`

Ends and deletes a specific chat session manually.

---

### 🟡 `POST /QA-chatbot/ask`

Ask a question in an active session.

```json
{
  "session_id": "abc123",
  "question": "কেন অনুপমকে নির্লিপ্ত বলা হয়েছে?"
}
```

**Response**

```json
{
  "session_id": "abc123",
  "question": "কেন অনুপমকে নির্লিপ্ত বলা হয়েছে?",
  "answer": "কারণ তিনি সব বিষয়ে মায়ের কথাই মানতেন, নিজের মতামত প্রকাশ করতেন না।"
}
```

---

### 🔁 `POST /QA-chatbot/history`

Get chat history for a session.

```json
{
  "session_id": "abc123",
  "history": [
    {
      "question": "বিয়ের সময় কল্যাণীর বয়স কত ছিল?",
      "answer": "১৫ বছর"
    }
  ]
}
```

---

### 📊 `POST /QA-chatbot/evaluate`

Evaluate generated answers vs expected ones.

```json
{
  "data": [
    {
      "question": "অনুপমের মায়ের নাম কী?",
      "expected_answer": "নির্দেশ পাওয়া যায়নি"
    },
    {
      "question": "কীভাবে অনুপম স্বাধীনতা অর্জন করে?",
      "expected_answer": "মায়ের প্রভাব থেকে বেরিয়ে এসে নিজ সিদ্ধান্ত নেয়।"
    }
  ]
}
```

---

## 🔍 Text Extraction Strategy

### Why PyMuPDF + pytesseract?

* PDFs were scanned, not digital.
* `PyMuPDF` extracts each page as image.
* `pytesseract` performs Bangla OCR.

### Common OCR Issues

* Broken conjuncts
* Irregular spacing
* Noise characters

### Cleaning Function

```python
def clean_bangla_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[^\u0980-\u09FF\s.,?!ঃ]', '', text)
    return text.replace('্ ', '্').strip()
```

---

## 📚 Chunking Strategy

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=50,
    separators=["\n", "।", ".", " "]
)
```

---

## 🔗 Embedding + Retrieval

### Embedding Model

```python
OpenAIEmbeddings(model="text-embedding-3-large")
```

### Vector DB: Pinecone

* Stores document vectors
* Top-k retrieval via cosine similarity

---

## 📈 Evaluation Logic

* Cosine similarity for **relevance**
* Context matching for **groundedness**

---

## 🚀 Future Improvements

* 🧠 Better OCR (TrOCR or denoising pre-processing)
* 🇧🇩 Fine-tuned Bangla embeddings (LaBSE, IndicSBERT)
* 🧹 Improved OCR post-cleanup
* 🔎 Hybrid retrieval (Pinecone + BM25)
* 📄 Store page metadata with chunks

---

## 👩‍💻 Author

**Roksana Akter**
AI Engineer | Data Scientist | OCR & RAG Specialist

```

Let me know if you want this as `.pdf`, `.docx`, or converted into a landing page!
```
