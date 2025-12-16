# app.py
import os
from pathlib import Path
from typing import List, Dict, Any
import subprocess
import zipfile
import tempfile

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ---------- Config ----------

CHROMA_DB_DIR = "CHROMA_DB"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not set in environment.")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL_NAME = "llama-3.1-8b-instant"

CODE_EXTENSIONS = {
    ".py", ".ipynb",
    ".js", ".jsx", ".ts", ".tsx",
    ".java", ".kt", ".kts",
    ".cs", ".vb",
    ".cpp", ".cc", ".cxx", ".hpp", ".h", ".c",
    ".go", ".rs", ".swift",
    ".php", ".rb", ".pl",
    ".scala", ".hs",
    ".sh", ".bash", ".zsh",
    ".ps1",
    ".html", ".htm", ".css",
    ".json", ".yml", ".yaml", ".toml", ".ini",
    ".md", ".rst",
    ".xml",
}

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\nclass ", "\ndef ", "\n\n", "\n", " "],
)

# ---------- Core helpers ----------

def groq_chat(system_prompt: str, user_prompt: str) -> str:
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def load_code_files(repo_path: Path) -> List[Dict[str, Any]]:
    files = []
    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in CODE_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            st.warning(f"Error reading {path}: {e}")
            continue
        files.append({
            "file_path": str(path.relative_to(repo_path)),
            "ext": ext,
            "text": text,
        })
    return files


def build_chunks(files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks = []
    for f in files:
        doc_text = f["text"]
        file_path = f["file_path"]
        ext = f["ext"]
        splits = text_splitter.split_text(doc_text)
        for i, chunk in enumerate(splits):
            chunk_id = f"{file_path}::chunk-{i}"
            chunks.append({
                "id": chunk_id,
                "content": chunk,
                "metadata": {
                    "file_path": file_path,
                    "ext": ext,
                    "chunk_index": i,
                }
            })
    return chunks


def prepare_repo_path(
    source_type: str,
    uploaded_files,        # list of UploadedFile or None
    git_url: str | None,
    zip_file
) -> Path:
    """
    Normalize all ingestion modes into a local folder path.
    - File upload: save uploaded files into a temp folder.
    - GitHub repo URL: clone into work_repos.
    - ZIP upload: extract into temp folder.
    """
    work_root = Path("work_repos")
    work_root.mkdir(exist_ok=True)

    if source_type == "File upload":
        if not uploaded_files:
            raise ValueError("Upload at least one code file.")
        temp_dir = tempfile.mkdtemp(prefix="uploaded_code_")
        target_dir = Path(temp_dir)
        for uf in uploaded_files:
            # Keep original filename
            dest = target_dir / uf.name
            with open(dest, "wb") as f:
                f.write(uf.read())
        return target_dir

    elif source_type == "GitHub repo URL":
        assert git_url, "Provide a GitHub repo URL."
        repo_name = git_url.rstrip("/").split("/")[-1].replace(".git", "")
        target_dir = work_root / repo_name
        if not target_dir.exists():
            st.info(f"Cloning {git_url} ...")
            subprocess.run(["git", "clone", git_url, str(target_dir)], check=True)
        return target_dir

    elif source_type == "ZIP upload":
        assert zip_file is not None, "Upload a ZIP file."
        temp_dir = tempfile.mkdtemp(prefix="code_zip_")
        target_dir = Path(temp_dir)
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(target_dir)
        return target_dir

    else:
        raise ValueError(f"Unknown source_type: {source_type}")


@st.cache_resource(show_spinner=False)
def build_collection(repo_path: str):
    repo = Path(repo_path)
    code_files = load_code_files(repo)
    if not code_files:
        raise ValueError(f"No supported code files found under: {repo}")

    chunks = build_chunks(code_files)
    if not chunks:
        raise ValueError("No chunks were created from the code files.")

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Recreate collection fresh
    try:
        chroma_client.delete_collection("codebase")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="codebase",
        embedding_function=embedding_func,
    )

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["content"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    return collection, code_files


def retrieve_chunks(collection, query: str, k: int = 5):
    result = collection.query(
        query_texts=[query],
        n_results=k
    )
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    ids = result["ids"][0]
    return [
        {"id": _id, "content": doc, "metadata": meta}
        for doc, meta, _id in zip(docs, metas, ids)
    ]


def describe_project(code_files: List[Dict[str, Any]]) -> str:
    sample_files = code_files[: min(10, len(code_files))]
    context_parts = []
    for f in sample_files:
        snippet = f["text"][:800]
        context_parts.append(
            f"File: {f['file_path']}\n---\n{snippet}\n"
        )
    context_str = "\n\n".join(context_parts)
    system_prompt = (
        "You are a senior software engineer. "
        "Given snippets from multiple files of a codebase, "
        "explain what this project does, main components, and how they interact. "
        "Give a clear and concise explanation from the code; "
        "if the user asks about any logic, function, formulas or scripts, "
        "explain them meaningfully. "
        "Be concise but clear. Mention key files and responsibilities."
    )
    user_prompt = f"Here are code snippets:\n\n{context_str}\n\nExplain the project."
    return groq_chat(system_prompt, user_prompt)


def ask_code(collection, question: str, k: int = 6) -> str:
    retrieved = retrieve_chunks(collection, question, k=k)
    context_blocks = []
    for r in retrieved:
        fp = r["metadata"]["file_path"]
        ext = r["metadata"]["ext"]
        context_blocks.append(
            f"[File: {fp} (ext={ext})]\n{r['content']}\n"
        )
    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a highly skilled Senior Software Engineer and Code Analyst. "
        "Your answers MUST be based ONLY on the provided code snippets (CONTEXT). "
        "If the context is insufficient, say: 'The relevant code details were not found in the indexed snippets.' "
        "Explain functions, variables, and logic clearly. "
        "When citing, mention the file name like (Source: filename.py)."
    )

    user_prompt = (
        f"Question: {question}\n\n"
        "--- CONTEXT: CODE SNIPPETS ---\n"
        f"{context_str}\n"
        "-----------------------------\n\n"
        "Now provide a detailed, professional answer based strictly on the CONTEXT above."
    )
    return groq_chat(system_prompt, user_prompt)


# ---------- Streamlit UI ----------

st.set_page_config(page_title="Codebase Explainer", layout="wide")
st.title("📂 Codebase Explainer & QA")

st.sidebar.header("Ingestion Options")

source_type = st.sidebar.radio(
    "Select source type",
    options=["File upload", "GitHub repo URL", "ZIP upload"],
)

uploaded_files = None
git_repo_url = None
zip_upload = None

if source_type == "File upload":
    uploaded_files = st.sidebar.file_uploader(
        "Upload code files",
        type=None,  # allow all; CODE_EXTENSIONS will filter
        accept_multiple_files=True,
        help="Upload one or more code files (any language).",
    )

elif source_type == "GitHub repo URL":
    git_repo_url = st.sidebar.text_input(
        "GitHub repo URL",
        value="",
        help="Public HTTPS GitHub URL to clone and analyze.",
    )

elif source_type == "ZIP upload":
    zip_upload = st.sidebar.file_uploader(
        "Upload ZIP file",
        type=["zip"],
        help="Upload a ZIP containing your code/project.",
    )

if "collection" not in st.session_state:
    st.session_state.collection = None
    st.session_state.code_files = None
    st.session_state.project_description = None
    st.session_state.chat_history = []

if st.sidebar.button("Analyse"):
    try:
        repo_path = prepare_repo_path(source_type, uploaded_files, git_repo_url, zip_upload)
        with st.spinner("Indexing codebase and building Chroma collection..."):
            collection, code_files = build_collection(str(repo_path))
        st.session_state.collection = collection
        st.session_state.code_files = code_files
        st.session_state.project_description = describe_project(code_files)
        st.success("Index built successfully!")
    except Exception as e:
        st.error(f"Indexing failed: {e}")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Project Overview")
    if st.session_state.project_description:
        st.write(st.session_state.project_description)
    else:
        st.info("Upload files / select GitHub / ZIP and click Analyse to see overview.")

with col2:
    st.subheader("Chat with the Codebase")
    if st.session_state.collection is None:
        st.info("Index a repo or upload files first.")
    else:
        user_q = st.text_input("Ask a question about this codebase")
        if st.button("Ask") and user_q.strip():
            with st.spinner("Thinking..."):
                ans = ask_code(st.session_state.collection, user_q)
            # append in natural order (oldest -> newest)
            st.session_state.chat_history.append(("user", user_q))
            st.session_state.chat_history.append(("assistant", ans))

        # 🔽 render in reverse so newest appears on top
        for role, msg in reversed(st.session_state.chat_history):
            if role == "user":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"**Assistant:** {msg}")
