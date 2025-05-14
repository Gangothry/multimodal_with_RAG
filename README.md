# 📄 PDF Interactive Learning App

This is an interactive Streamlit-based learning platform powered by LLMs that allows users to upload a **PDF document** and learn from it using three engaging modes:

### ✅ Features

- 🔍 **PDF Parsing**  
  Extracts **text**, **tables**, and **images** from uploaded PDFs using `unstructured[all-docs]`.

- 🧠 **Semantic Search & QA**  
  Uses **RAG (Retrieval-Augmented Generation)** with vector embeddings to allow **question answering** based on document content.

- 💬 **Chat Mode**  
  Ask any natural-language questions related to the uploaded document.

- 🧾 **Flashcard Mode**  
  Automatically generates concept-definition pairs to help with active recall.

- 📝 **Quiz Mode**  
  Auto-generates multiple-choice questions (MCQs) to test your understanding of the content.

### 💻 Requirements

- **Python 3.10** (🚨 Must use Python 3.10 for compatibility with `unstructured[all-docs]`)
- `poppler-utils` (Poppler must be installed separately for image processing)

#### 🧰 Python Dependencies

Install required libraries with:

```bash
pip install -r requirements.txt
