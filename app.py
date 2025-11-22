import streamlit as st
import json
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOpenAI
import asyncio
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PDF_PATH = "ukpga_20250022_en.pdf"

st.title("Universal Credit Act 2025 AI Analysis")

@st.cache_data(show_spinner=True)
def extract_full_text(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    full_text = "\n\n".join([page.page_content.strip() for page in pages])
    return full_text

@st.cache_resource(show_spinner=True)
def load_vectorstore():
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

vectorstore = None
if os.path.exists("faiss_index"):
    vectorstore = FAISS.load_local(
        "faiss_index",
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True,
    )
else:
    vectorstore = load_vectorstore()

agent = st.selectbox("Choose LLM Model", ["Gemini", "OpenAI"])

def extract_json_from_text(text):
    # Extract JSON inside markdown json code fences if present
    match = re.search(r"``````", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # fallback: try to find JSON object anywhere
    json_part = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if json_part:
        return json_part.group(1).strip()
    return text.strip()

async def async_qa_gemini(query, retriever, chat_history):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=GEMINI_API_KEY,
    )
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    context = " ".join([msg["query"] + " " + msg["response"] for msg in chat_history[-5:]])
    full_query = f"Previous context: {context}\n\nUser query: {query}"
    response = await qa.arun(full_query)
    return response

def sync_qa_openai(query, retriever, chat_history):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        api_key=OPENAI_API_KEY,
    )
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    context = " ".join([msg["query"] + " " + msg["response"] for msg in chat_history[-5:]])
    full_query = f"Previous context: {context}\n\nUser query: {query}"
    return qa.run(full_query)

def run_analysis():
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    if not hasattr(run_analysis, "chat_history"):
        run_analysis.chat_history = []
    chat_history = run_analysis.chat_history

    full_text = extract_full_text(PDF_PATH)

    summary_prompt = (
        "Summarize the Universal Credit Act 2025 into 5-10 key bullet points focusing on: "
        "Purpose, Key definitions, Eligibility, Obligations, Enforcement elements."
    )
    extraction_prompt = (
        "Extract key legislative sections from the Act. Provide JSON with keys: "
        "\"definitions\", \"obligations\", \"responsibilities\", \"eligibility\", \"payments\", \"penalties\", \"record_keeping\"."
    )
    rule_check_prompt = (
        "Check the following 6 rules about the Act and provide JSON outputs per rule: "
        "1) Act must define key terms; "
        "2) Act must specify eligibility criteria; "
        "3) Act must specify responsibilities of administering authority; "
        "4) Act must include enforcement or penalties; "
        "5) Act must include payment calculation or entitlement structure; "
        "6) Act must include record-keeping or reporting requirements."
    )

    if agent == "Gemini":
        summary = asyncio.run(async_qa_gemini(summary_prompt, retriever, chat_history))
        extraction_raw = asyncio.run(async_qa_gemini(extraction_prompt, retriever, chat_history))
        rules_raw = asyncio.run(async_qa_gemini(rule_check_prompt, retriever, chat_history))
    else:
        if not OPENAI_API_KEY:
            st.error("OPENAI_API_KEY missing in .env file")
            return None, None, None, None
        summary = sync_qa_openai(summary_prompt, retriever, chat_history)
        extraction_raw = sync_qa_openai(extraction_prompt, retriever, chat_history)
        rules_raw = sync_qa_openai(rule_check_prompt, retriever, chat_history)

    # Extract clean JSON from raw LLM responses
    extraction_clean = extract_json_from_text(extraction_raw)
    rules_clean = extract_json_from_text(rules_raw)

    try:
        extraction_json = json.loads(extraction_clean)
    except Exception as e:
        extraction_json = {"error": "Extraction output not valid JSON", "raw": extraction_raw, "exception": str(e)}

    try:
        rules_json = json.loads(rules_clean)
    except Exception as e:
        rules_json = {"error": "Rules output not valid JSON", "raw": rules_raw, "exception": str(e)}

    chat_history.extend([
        {"query": summary_prompt, "response": summary},
        {"query": extraction_prompt, "response": extraction_raw},
        {"query": rule_check_prompt, "response": rules_raw},
    ])

    # Prepare final report JSON
    final_report = {
        "full_text": full_text,
        "summary": summary,
        "key_legislative_sections": extraction_json,
        "rule_checks": rules_json
    }

    with open("universal_credit_act_2025_report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    return full_text, summary, extraction_json, rules_json

if st.button("Run Full Act Analysis"):
    with st.spinner("Running analysis..."):
        full_text, summary, extracted_sections, rule_checks = run_analysis()
        if full_text is None:
            st.stop()

        st.markdown("## Task 1: Full Extracted Text (Raw from PDF)")
        with st.expander("Show full extracted text"):
            st.text_area("", full_text, height=300)

        st.markdown("## Task 2: Summary (5-10 bullet points)")
        st.write(summary)

        st.markdown("## Task 3: Extracted Key Legislative Sections JSON")
        st.json(extracted_sections)

        st.markdown("## Task 4: Rule Checks JSON")
        st.json(rule_checks)

        st.success("Final JSON report saved as universal_credit_act_2025_report.json")
