import os
import re
import time
import uuid
import base64
from dotenv import load_dotenv
from IPython.display import Image, display, clear_output
from openai import RateLimitError
from base64 import b64decode
from unstructured.partition.pdf import partition_pdf
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
import io

def get_pdf_file():
    """Get PDF file through either widget or file path input"""
    try:
        get_ipython() # This will raise NameError if not in Jupyter
        from ipywidgets import FileUpload, Button, Output, VBox
        from IPython.display import display, clear_output
        print("Running in Jupyter environment with widgets")

        upload = FileUpload(accept='.pdf', multiple=False, description='Upload PDF')
        output = Output()

        def on_upload(change):
            with output:
                clear_output()
                if not upload.value:
                    print("Please upload a PDF file first.")
                    return
                file_name = next(iter(upload.value))
                content = upload.value[file_name]['content']
                processor = PDFProcessor()
                processor.file_content = content
                processor.process_pdf_content()
                processor.ask_questions_loop()

        upload.observe(on_upload, names='value')
        display(VBox([upload, output]))
        return None

    except NameError:
        print("Running in standard Python environment")
        file_path = input("Enter path to PDF file: ").strip()
        if not file_path:
            return None
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
        
def load_environment_variables():
    """Load environment variables from .env file"""
    load_dotenv()
    return {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY"),
        "LANGCHAIN_TRACING_V2_ENABLED": os.getenv("LANGCHAIN_TRACING_V2_ENABLED"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    }


def extract_pdf_content(file_path=None, file_content=None):
    """Extract content from PDF file using unstructured library"""
    if file_path:
        return partition_pdf(
            filename=file_path,
            pdf_infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=5000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=3000,
        )
    elif file_content:
        return partition_pdf(
            file=io.BytesIO(file_content),
            pdf_infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=5000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=3000,
        )
    else:
        raise ValueError("Either file_path or file_content must be provided")


def separate_content_types(chunks):
    """Separate tables from text chunks"""
    tables = []
    texts = []
    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)
        if "CompositeElement" in str(type(chunk)):
            texts.append(chunk)
    return tables, texts


def get_images_base64(chunks):
    """Extract base64 encoded images from chunks"""
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64


def display_base64_image(base64_code):
    """Display base64 encoded image"""
    image_data = base64.b64decode(base64_code)
    display(Image(data=image_data))


def create_summary_chain(api_key):
    """Create chain for summarizing text and tables"""
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additional comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant", groq_api_key=api_key)
    return {"element": lambda x: x} | prompt | model | StrOutputParser()


def safe_batch_summarize(chain, inputs, config, max_retries=5):
    """Handle rate limits when summarizing content"""
    for attempt in range(max_retries):
        try:
            return chain.batch(inputs, config)
        except RateLimitError as e:
            message = str(e)
            match = re.search(r'try again in ([\d.]+)s', message)
            wait_time = float(match.group(1)) if match else 15
            print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
    raise Exception("Rate limit error: max retries exceeded.")


def create_image_description_chain(api_key):
    """Create chain for describing images"""
    prompt_template = """Describe the image in detail. For context,
    the image is part of a research paper explaining the transformers
    architecture. Be specific about graphs, such as bar plots."""

    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key
    )
    return prompt | llm | StrOutputParser()


def describe_images(chain, images):
    """Describe images with rate limiting handling"""
    image_summaries = []
    for idx, image in enumerate(images):
        try:
            print(f"Processing image {idx + 1}/{len(images)}...")
            result = chain.invoke({"image": image})
            image_summaries.append(result)
            time.sleep(5)  # Adjust based on observed limits
        except Exception as e:
            print(f"Error processing image {idx + 1}: {e}")
            image_summaries.append("Error or rate-limited")
    return image_summaries


def setup_retriever(api_key, texts, text_summaries, tables, table_summaries, images, image_summaries):
    """Setup multi-vector retriever with all content types"""
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=api_key,
        model="models/embedding-001"
    )
    vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=embeddings)
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Add text documents
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]})
        for i, summary in enumerate(text_summaries)
        if summary.strip()
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset([
        (doc_ids[i], texts[i])
        for i, summary in enumerate(text_summaries)
        if summary.strip()
    ])

    # Add table documents
    table_ids = [str(uuid.uuid4()) for _ in tables]
    valid_docs = []
    valid_ids = []
    for i, summary in enumerate(table_summaries):
        if summary.strip():
            try:
                embedding = embeddings.embed_query(summary)
                if embedding:
                    doc = Document(page_content=summary, metadata={id_key: table_ids[i]})
                    valid_docs.append(doc)
                    valid_ids.append(table_ids[i])
            except Exception as e:
                print(f"Embedding failed for index {i}: {e}")

    if valid_docs:
        retriever.vectorstore.add_documents(valid_docs)
        retriever.docstore.mset(list(zip(valid_ids, [tables[i] for i in range(len(table_ids)) if table_ids[i] in valid_ids])))
    else:
        print("No valid table summaries to add.")

    # Add image documents
    img_ids = [str(uuid.uuid4()) for _ in images]
    summary_img = [
        Document(page_content=summary, metadata={id_key: img_ids[i]})
        for i, summary in enumerate(image_summaries)
        if summary.strip()
    ]
    retriever.vectorstore.add_documents(summary_img)
    retriever.docstore.mset([
        (img_ids[i], images[i])
        for i, summary in enumerate(image_summaries)
        if summary.strip()
    ])

    return retriever


def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}


def build_prompt(kwargs):
    """Construct prompt with context from different content types"""
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text

    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )


def create_qa_chain(retriever, api_key):
    """Create question answering chain with sources"""
    chain_with_sources = {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        response=(
            RunnableLambda(build_prompt)
            | ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
            | StrOutputParser()
        )
    )

    return chain_with_sources

