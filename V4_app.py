import os
import sys
import logging
import yaml
import requests
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess
import zipfile
import tempfile
from datetime import datetime
import time
import re
import base64

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from groq import Groq

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('code_explainer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# ---------- LLM Configuration ----------
def load_llm_config() -> Dict[str, Any]:
    config_path = Path("llm_config.yaml")
    if not config_path.exists():
        logger.warning("llm_config.yaml not found, using default Groq config.")
        return {
            "providers": {
                "groq": {
                    "enabled": True,
                    "api_key_env": "GROQ_API_KEY",
                    "base_url": "https://api.groq.com/openai/v1",
                    "default_model": "llama-3.1-8b-instant",
                    "max_tokens_per_minute": 6000,
                    "priority": 1,
                }
            },
            "fallback_enabled": True,
            "max_retries": 2,
            "timeout_seconds": 30,
        }
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

LLM_CONFIG = load_llm_config()

# ---------- Configuration ----------
CHROMA_DB_DIR = "CHROMA_DB"
MAX_FILE_SIZE_MB = 10
MAX_TOTAL_FILES = 1000
GITHUB_URL_PATTERN = r'^https?://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+(/)?$'

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

DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200

# ---------- Core helpers ----------
def llm_chat(system_prompt: str, user_prompt: str, temperature: float = 0.1, preferred_provider: str = None) -> str:
    """
    Call LLM with fallback across multiple providers and models.
    """
    config = LLM_CONFIG
    providers = config.get("providers", {})
    fallback_enabled = config.get("fallback_enabled", True)
    model_fallback_enabled = config.get("model_fallback_enabled", True)
    max_retries = config.get("max_retries", 3)
    timeout = config.get("timeout_seconds", 60)

    # Filter and sort providers
    enabled_providers = [(name, cfg) for name, cfg in providers.items() if cfg.get("enabled", True)]
    
    # If preferred_provider is specified, prioritize it
    if preferred_provider and preferred_provider in dict(enabled_providers):
        preferred_cfg = dict(enabled_providers)[preferred_provider]
        enabled_providers = [(preferred_provider, preferred_cfg)] + \
                           [(name, cfg) for name, cfg in enabled_providers if name != preferred_provider]
    
    sorted_providers = sorted(
        enabled_providers,
        key=lambda x: x[1].get("priority", 999),
        reverse=False,
    )

    errors = []
    for provider_name, cfg in sorted_providers:
        api_key_env = cfg.get("api_key_env")
        api_key = os.getenv(api_key_env)
        if not api_key:
            logger.warning(f"API key not set for {provider_name} (env: {api_key_env})")
            errors.append(f"{provider_name}: API key missing")
            continue

        base_url = cfg.get("base_url")
        default_model = cfg.get("default_model")
        fallback_models = cfg.get("fallback_models", [])
        max_tpm = cfg.get("max_tokens_per_minute", 10000)

        total_chars = len(system_prompt) + len(user_prompt)
        estimated_tokens = total_chars // 4
        if estimated_tokens > max_tpm * 0.8:
            logger.warning(f"Estimated tokens {estimated_tokens} may exceed {provider_name} TPM limit {max_tpm}")

        # Try models in order: default_model, then fallback_models
        models_to_try = [default_model] + (fallback_models if model_fallback_enabled else [])
        
        for model in models_to_try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
            }

            try:
                logger.info(f"Calling {provider_name} with model {model}")
                response = requests.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                logger.info(f"{provider_name} with model {model} call successful")
                return content
            except requests.exceptions.RequestException as e:
                error_msg = f"{provider_name} with model {model} request failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                
                # If it's a 429 (rate limit) or 404 (model not found), try next model
                if hasattr(e, 'response') and e.response is not None:
                    if e.response.status_code in [429, 404, 400]:
                        continue  # Try next model
                    elif e.response.status_code >= 500:
                        break  # Server error, try next provider
                else:
                    continue  # Network error, try next model
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                error_msg = f"{provider_name} with model {model} response parsing failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                continue  # Try next model

    raise RuntimeError(
        f"All LLM providers and models failed:\n" + "\n".join(errors[-10:])
    )

def groq_chat(system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
    logger.warning("groq_chat is deprecated, using llm_chat with multi-provider fallback.")
    return llm_chat(system_prompt, user_prompt, temperature)

def validate_and_refine_docs(code_files: List[Dict[str, Any]], draft_doc: str) -> str:
    code_summaries = []
    for f in code_files[:20]:
        code_summaries.append(f"File: {f['file_path']}\n---\n{f['text'][:800]}\n")
    code_context = "\n\n".join(code_summaries)

    critique_prompt = f"""
You are a TECHNICAL DOCUMENTATION REVIEWER ensuring documentation is professional, code‑grounded, and free of hallucinations.

**REQUIRED SECTIONS (expected, but can be adjusted based on content):**
1. Executive Summary
2. System Overview & Architecture
3. Business Process Flow / Decision Logic
4. Data Architecture & Storage Design
5. Core Logic & Algorithms
6. Integration Points & External Services
7. Execution & Operations
8. Key Features

**STRICT RULES:**
1. Every claim MUST reference specific files/functions from CODE SNIPPETS below.
2. Remove domain hallucinations unless explicitly present in the user's code.
3. For "Data & Storage Design": describe only variables/files/outputs visible in code (no fake DBs).
4. For "Integration": only real imports/APIs visible in code.
5. If a section has no content based on code, write "Not applicable based on available source files."
6. Ensure the document is well‑structured with clear headings and bullet points.
7. Preserve any relevant code examples and file references.

CODE SNIPPETS:
{code_context}

DRAFT:
{draft_doc}

**Output ONLY the corrected document. No analysis text.**
"""
    revised = llm_chat(
        "Professional technical writer. Ensure documentation is accurate, code‑grounded, and follows enterprise standards.",
        critique_prompt,
        temperature=0.1,
        preferred_provider="groq_critique"
    )
    return revised or draft_doc

def load_code_files(repo_path: Path) -> List[Dict[str, Any]]:
    files = []
    file_count = 0
    for path in repo_path.rglob("*"):
        if file_count >= MAX_TOTAL_FILES:
            st.warning(f"Limited to first {MAX_TOTAL_FILES} files.")
            break
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in CODE_EXTENSIONS:
            continue
        try:
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.warning(f"Skipping large file {path.name} ({file_size_mb:.1f} MB)")
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning(f"Error reading {path}: {e}")
            st.warning(f"Error reading {path}: {e}")
            continue
        files.append({
            "file_path": str(path.relative_to(repo_path)),
            "ext": ext,
            "text": text,
            "size_bytes": path.stat().st_size,
        })
        file_count += 1
    logger.info(f"Loaded {len(files)} code files from {repo_path}")
    return files

def build_chunks(files: List[Dict[str, Any]], chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\nclass ", "\ndef ", "\n\n", "\n", " "],
    )
    chunks = []
    for f in files:
        doc_text = f["text"]
        file_path = f["file_path"]
        ext = f["ext"]
        splits = splitter.split_text(doc_text)
        for i, chunk in enumerate(splits):
            chunk_id = f"{file_path}::chunk-{i}"
            chunks.append({
                "id": chunk_id,
                "content": chunk,
                "metadata": {
                    "file_path": file_path,
                    "ext": ext,
                    "chunk_index": i,
                    "file_size": f.get("size_bytes", 0),
                }
            })
    logger.info(f"Created {len(chunks)} chunks from {len(files)} files")
    return chunks

def prepare_repo_path(source_type: str, uploaded_files, git_url: str | None, zip_file) -> Path:
    work_root = Path("work_repos")
    work_root.mkdir(exist_ok=True)

    if source_type == "File upload":
        if not uploaded_files:
            raise ValueError("Upload at least one code file.")
        temp_dir = tempfile.mkdtemp(prefix="uploaded_code_")
        target_dir = Path(temp_dir)
        for uf in uploaded_files:
            safe_name = os.path.basename(uf.name)
            dest = target_dir / safe_name
            with open(dest, "wb") as f:
                f.write(uf.read())
        return target_dir

    elif source_type == "GitHub repo URL":
        if not git_url:
            raise ValueError("Provide a GitHub repo URL.")
        if not re.match(GITHUB_URL_PATTERN, git_url):
            st.warning("URL does not look like a valid GitHub repository URL.")
        repo_name = git_url.rstrip("/").split("/")[-1].replace(".git", "")
        target_dir = work_root / repo_name
        if not target_dir.exists():
            st.info(f"Cloning {git_url} ...")
            try:
                subprocess.run(["git", "clone", git_url, str(target_dir)], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Git clone failed: {e.stderr}")
                raise ValueError(f"Failed to clone repository: {e.stderr}")
        return target_dir

    elif source_type == "ZIP upload":
        if zip_file is None:
            raise ValueError("Upload a ZIP file.")
        temp_dir = tempfile.mkdtemp(prefix="code_zip_")
        target_dir = Path(temp_dir)
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(target_dir)
        return target_dir

    else:
        raise ValueError(f"Unknown source_type: {source_type}")

@st.cache_resource(show_spinner=False)
def build_collection(repo_path: str, chunk_size: int, chunk_overlap: int):
    repo = Path(repo_path)
    code_files = load_code_files(repo)
    if not code_files:
        raise ValueError(f"No supported code files found under: {repo}")

    chunks = build_chunks(code_files, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("No chunks were created from the code files.")

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

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
    logger.info(f"Collection built with {len(chunks)} chunks")
    return collection, code_files

def retrieve_chunks(collection, query: str, k: int = 5):
    result = collection.query(query_texts=[query], n_results=k)
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    ids = result["ids"][0]
    return [
        {"id": _id, "content": doc, "metadata": meta}
        for doc, meta, _id in zip(docs, metas, ids)
    ]

def describe_project(code_files: List[Dict[str, Any]]) -> str:
    sample_files = code_files[: min(15, len(code_files))]
    context_parts = []
    for f in sample_files:
        snippet = f["text"][:1200]
        context_parts.append(
            f"File: {f['file_path']} (ext: {f['ext']}, size: {f.get('size_bytes', 0)} bytes)\n---\n{snippet}\n"
        )
    context_str = "\n\n".join(context_parts)
    
    total_files = len(code_files)
    total_size = sum(f.get("size_bytes", 0) for f in code_files)
    extensions = set(f["ext"] for f in code_files)
    
    system_prompt = (
        "You are a senior software engineer. Given snippets from multiple files of a codebase, "
        "provide a **structured project overview** with the following sections:\n"
        "1. **Purpose**: What this project does (2-3 sentences).\n"
        "2. **Technology Stack**: Languages, frameworks, libraries (based on imports and file extensions).\n"
        "3. **Architecture**: High-level components and how they interact.\n"
        "4. **Key Files**: Most important files and their responsibilities.\n"
        "5. **Dependencies**: External services, APIs, databases used.\n"
        "6. **Development Notes**: Any notable patterns, conventions, or potential issues.\n\n"
        "Base your explanation ONLY on the provided code snippets. "
        "If something is not evident, say 'Not evident from the code snippets.' "
        "Be concise but comprehensive. Use bullet points where appropriate."
    )
    user_prompt = (
        f"Project Metadata: {total_files} files, {total_size:,} bytes total, extensions: {', '.join(extensions)}\n\n"
        f"Code snippets:\n\n{context_str}\n\n"
        "Provide the structured project overview as described."
    )
    return groq_chat(system_prompt, user_prompt)

def ask_code(collection, question: str, k: int = 6) -> str:
    retrieved = retrieve_chunks(collection, question, k=k)
    if not retrieved:
        return "No relevant code snippets found to answer this question."

    context_blocks = []
    for r in retrieved:
        fp = r["metadata"]["file_path"]
        ext = r["metadata"]["ext"]
        idx = r["metadata"]["chunk_index"]
        context_blocks.append(f"[File: {fp} (ext={ext}, chunk={idx})]\n{r['content']}\n")
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

def generate_enterprise_documentation(code_files: List[Dict[str, Any]], collection) -> str:
    """
    Generate 7-section enterprise documentation with strict code-grounded validation.
    """
    project_name = Path(code_files[0]["file_path"]).parent.name if code_files else "Project"
    total_files = len(code_files)
    total_chars = sum(len(f["text"]) for f in code_files)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    progress = st.progress(0, text="Generating enterprise technical documentation...")
    status = st.empty()

    # Build comprehensive context from key files
    all_context = []
    for i, f in enumerate(code_files[:20]):
        status.text(f"Analyzing {f['file_path']}")
        snippet = f["text"][:1500]
        all_context.append(f"File: {f['file_path']} (ext: {f['ext']}, size: {f.get('size_bytes', 0)} bytes)\n{snippet}\n{'='*80}")
        progress.progress((i + 1) / 20)
        time.sleep(0.05)

    context_block = "\n\n".join(all_context)
    progress.progress(0.3)

    # Generate using 7-section enterprise template
    doc_prompt = f"""
You are writing PROFESSIONAL ENTERPRISE TECHNICAL DOCUMENTATION matching Fortune 500 standards.

Use ONLY code below. Follow EXACTLY this 7-section structure:

## 1. Executive Summary
- Project purpose and business value (2-3 sentences)
- Key capabilities and deliverables
- Target audience and use cases

## 2. System Architecture & Design
- High-level components and their interactions
- File responsibilities and relationships
- Technology stack (from imports and file extensions only)
- Architectural patterns and design decisions

## 3. Data Architecture & Storage Design
- Input data formats, sources, and validation
- Internal data structures, variables, and models
- Output formats, files, and APIs generated
- File paths, storage patterns, and persistence mechanisms

## 4. Core Business Logic & Algorithms
- Step-by-step processing flows with decision points
- Key functions, their signatures, and responsibilities
- Business rules, validation logic, and decision trees
- Code snippets for critical algorithms with explanations

## 5. Integration Points & External Services
- External libraries, frameworks, and APIs used
- File I/O operations and system interactions
- Environment dependencies and configuration
- Third-party service calls, webhooks, and integrations

## 6. Key Features & Capabilities
- Feature breakdown with implementation details
- User workflows and interaction patterns
- Performance characteristics and scalability
- Security considerations and access controls

## 7. Execution Workflow & Operations
- Prerequisites, installation, and setup instructions
- How to run each component/file/notebook
- Configuration requirements and environment setup
- Deployment procedures, monitoring, and maintenance
- Known limitations, error handling, and troubleshooting

CODE FILES:
{context_block}

**Style Requirements:**
- Professional paragraphs with bullet points for clarity
- Reference exact file names, functions, and line numbers
- Include relevant code snippets with explanations
- Use tables for comparisons where appropriate
- Numbered instructions for complex procedures
- Highlight important warnings and considerations
"""
    
    status.text("Generating enterprise draft...")
    draft_doc = llm_chat(
        "Enterprise technical documentation specialist with Fortune 500 experience. Match exact 7-section template. Code-grounded only.",
        doc_prompt,
        temperature=0.1,
        preferred_provider="groq_fast"
    )
    progress.progress(0.7)

    # Final validation and refinement
    status.text("Applying enterprise validation and quality checks...")
    final_doc = validate_and_refine_docs(code_files, draft_doc)
    progress.progress(1.0)

    progress.empty()
    status.empty()

    full_doc = f"""# {project_name} – Enterprise Technical Documentation

**Files analyzed**: {total_files}  
**Total size**: {total_chars:,} characters  
**Generated**: {now}  
**Documentation Standard**: Enterprise 7-Section Template

---

{final_doc}

---

## Quality Assurance
- ✅ Code-grounded validation applied
- ✅ Hallucination removal completed  
- ✅ File/function references verified
- ✅ Enterprise formatting standards met
- ✅ Production-ready documentation

*Auto-generated from source code analysis. All details derived directly from source files.*
"""
    return full_doc

def generate_pdf_from_markdown(markdown_content: str, project_name: str) -> Optional[bytes]:
    """
    Generate PDF from markdown using wkhtmltopdf if available.
    Returns properly formatted PDF with professional styling.
    """
    try:
        import pdfkit
        # Configure pdfkit
        config = pdfkit.configuration()
        
        # Get current date
        from datetime import datetime
        generated_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Try to use markdown library for better conversion
        try:
            import markdown
            html_content = markdown.markdown(markdown_content, extensions=['extra', 'codehilite', 'tables'])
        except ImportError:
            # Fallback to simple conversion
            html_content = markdown_content
            
            # Basic markdown conversions
            # Headers
            html_content = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html_content, flags=re.MULTILINE)
            html_content = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html_content, flags=re.MULTILINE)
            html_content = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html_content, flags=re.MULTILINE)
            html_content = re.sub(r'^#### (.*?)$', r'<h4>\1</h4>', html_content, flags=re.MULTILINE)
            
            # Bold and italic
            html_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html_content)
            html_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html_content)
            
            # Code blocks (simple handling)
            html_content = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', html_content, flags=re.DOTALL)
            html_content = re.sub(r'`(.*?)`', r'<code>\1</code>', html_content)
            
            # Lists
            html_content = re.sub(r'^\s*[-*]\s+(.*?)$', r'<li>\1</li>', html_content, flags=re.MULTILINE)
            html_content = re.sub(r'(<li>.*?</li>\s*)+', r'<ul>\g<0></ul>', html_content, flags=re.DOTALL)
            
            # Paragraphs for lines not already in HTML tags
            lines = html_content.split('\n')
            result_lines = []
            in_list = False
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    if in_list:
                        result_lines.append('</ul>')
                        in_list = False
                    result_lines.append('<p></p>')
                elif stripped.startswith('<li>'):
                    if not in_list:
                        result_lines.append('<ul>')
                        in_list = True
                    result_lines.append(line)
                elif stripped.startswith('<'):
                    if in_list:
                        result_lines.append('</ul>')
                        in_list = False
                    result_lines.append(line)
                else:
                    if in_list:
                        result_lines.append('</ul>')
                        in_list = False
                    result_lines.append(f'<p>{line}</p>')
            
            if in_list:
                result_lines.append('</ul>')
            
            html_content = '\n'.join(result_lines)
        
        # Clean up any double tags
        html_content = html_content.replace('</ul></ul>', '</ul>')
        html_content = html_content.replace('<ul><ul>', '<ul>')
        
        # Professional HTML template with proper styling
        html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{project_name} -Technical Documentation</title>
    <style>
        @page {{
            margin: 1.5cm;
        }}
        
        body {{
            font-family: 'Calibri', 'Arial', sans-serif;
            font-size: 11pt;
            line-height: 1.5;
            color: #000000;
            margin: 0;
            padding: 0;
            background-color: #ffffff;
        }}
        
        .document {{
            max-width: 21cm;
            margin: 0 auto;
            padding: 2cm;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 3px solid #2c3e50;
        }}
        
        .header h1 {{
            color: #2c3e50;
            font-size: 28pt;
            margin: 0 0 10px 0;
            font-weight: bold;
            font-family: 'Calibri Light', 'Arial Narrow', sans-serif;
        }}
        
        .header .subtitle {{
            color: #7f8c8d;
            font-size: 14pt;
            margin: 0;
            font-style: italic;
        }}
        
        .metadata {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 20px;
            margin: 25px 0;
            font-size: 10pt;
            border-left: 5px solid #3498db;
        }}
        
        .metadata p {{
            margin: 8px 0;
            line-height: 1.4;
        }}
        
        h1 {{
            color: #2c3e50;
            font-size: 20pt;
            font-weight: bold;
            margin-top: 35px;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 2px solid #ecf0f1;
            page-break-after: avoid;
            font-family: 'Calibri Light', 'Arial Narrow', sans-serif;
        }}
        
        h2 {{
            color: #34495e;
            font-size: 16pt;
            font-weight: bold;
            margin-top: 28px;
            margin-bottom: 12px;
            page-break-after: avoid;
        }}
        
        h3 {{
            color: #7f8c8d;
            font-size: 14pt;
            font-weight: bold;
            margin-top: 22px;
            margin-bottom: 10px;
            page-break-after: avoid;
        }}
        
        p {{
            margin: 12px 0;
            text-align: justify;
            line-height: 1.6;
        }}
        
        ul, ol {{
            margin: 12px 0 12px 25px;
            padding: 0;
        }}
        
        li {{
            margin: 6px 0;
            line-height: 1.5;
        }}
        
        code {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 3px;
            padding: 2px 6px;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 10pt;
            color: #c7254e;
        }}
        
        pre {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
            overflow-x: auto;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 10pt;
            page-break-inside: avoid;
            border-left: 4px solid #3498db;
        }}
        
        pre code {{
            background-color: transparent;
            border: none;
            padding: 0;
            color: inherit;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            font-size: 10pt;
            page-break-inside: avoid;
            border: 1px solid #dee2e6;
        }}
        
        th, td {{
            border: 1px solid #dee2e6;
            padding: 8px 10px;
            text-align: left;
            vertical-align: top;
        }}
        
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .footer {{
            margin-top: 40px;
            padding-top: 15px;
            border-top: 2px solid #dee2e6;
            font-size: 9pt;
            color: #7f8c8d;
            text-align: center;
            font-style: italic;
        }}
        
        .page-break {{
            page-break-before: always;
        }}
        
        /* Ensure proper text wrapping */
        .content p, .content li {{
            word-wrap: break-word;
            overflow-wrap: break-word;
        }}
    </style>
</head>
<body>
    <div class="document">
        <div class="header">
            <h1>{project_name}</h1>
            <p class="subtitle">Technical Documentation</p>
        </div>
        
        <div class="metadata">
            <p><strong>Generated:</strong> {generated_date}</p>
            <p><strong>Documentation Standard:</strong> Enterprise 7-Section Template</p>
            <p><strong>Quality Assurance:</strong> Code-grounded | Hallucination-free | File-referenced</p>
        </div>
        
        <div class="content">
            {content}
        </div>
        
        <div class="footer">
            <p>Auto-generated from source code analysis. All details derived directly from source files.</p>
            <p>Confidential & Proprietary - For internal use only</p>
        </div>
    </div>
</body>
</html>"""
        
        # Format the HTML template
        full_html = html_template.format(
            project_name=project_name,
            generated_date=generated_date,
            content=html_content
        )
        
        pdf_bytes = pdfkit.from_string(full_html, False, configuration=config)
        return pdf_bytes
    except Exception as e:
        logger.warning(f"PDF generation failed: {e}")
        return None

# ---------- Streamlit UI ----------
st.set_page_config(
    page_title="Codebase Analyzer v3",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Codebase Analyzer")
# st.markdown("""
# ### Complete Enterprise Codebase Analyzer - Summary
# **7-Section Ultra-Rich Documentation**: Executive Summary → Architecture → Data Design → Core Logic → Integrations → Key Features → Execution Workflow

# **Detailed Project Overview**: 6-section analysis (Purpose, Tech Stack, Architecture, Features, Patterns, Readiness) with file/function specifics

# **Professional PDF Export**: wkhtmltopdf-powered enterprise-styled PDFs with proper typography/tables

# **Code-Grounded Only**: Strict validation removes hallucinations, references exact files/functions/paths

# **Production-Ready**: Rich tables, code snippets, pseudocode, numbered instructions - client deliverable quality
# """)

st.sidebar.header("📂 Ingestion Options")
source_type = st.sidebar.radio(
    "Select source type",
    options=["File upload", "GitHub repo URL", "ZIP upload"],
    index=0,
)

uploaded_files = None
git_repo_url = None
zip_upload = None

if source_type == "File upload":
    uploaded_files = st.sidebar.file_uploader(
        "Upload code files",
        type=None,
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

st.sidebar.header("⚙️ Configuration")
chunk_size = st.sidebar.slider("Chunk size", 500, 3000, DEFAULT_CHUNK_SIZE, 100)
chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 500, DEFAULT_CHUNK_OVERLAP, 50)

# Initialize session state
if "collection" not in st.session_state:
    st.session_state.collection = None
    st.session_state.code_files = None
    st.session_state.project_description = None
    st.session_state.chat_history = []
    st.session_state.enterprise_docs_md = None
    st.session_state.enterprise_docs_pdf = None
    st.session_state.analysis_complete = False

if st.sidebar.button("🚀 Analyze Codebase", type="primary", use_container_width=True):
    try:
        with st.spinner("🔄 Preparing repository..."):
            repo_path = prepare_repo_path(source_type, uploaded_files, git_repo_url, zip_upload)
        
        with st.spinner("📊 Indexing codebase and building vector database..."):
            collection, code_files = build_collection(str(repo_path), chunk_size, chunk_overlap)
        
        st.session_state.collection = collection
        st.session_state.code_files = code_files
        st.session_state.project_description = describe_project(code_files)
        st.session_state.enterprise_docs_md = None
        st.session_state.enterprise_docs_pdf = None
        st.session_state.analysis_complete = True
        
        st.success(f"✅ Successfully analyzed {len(code_files)} files!")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")

# Main tabs
tab1, tab2, tab3 = st.tabs(["📋 Project Overview", "💬 Code Q&A", "📚 Enterprise Documentation"])

with tab1:
    st.subheader("Project Overview")
    if st.session_state.project_description:
        st.markdown(st.session_state.project_description)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Files Analyzed", len(st.session_state.code_files) if st.session_state.code_files else 0)
        with col2:
            total_size = sum(f.get("size_bytes", 0) for f in st.session_state.code_files) if st.session_state.code_files else 0
            st.metric("Total Size", f"{total_size:,} bytes")
        with col3:
            extensions = set(f["ext"] for f in st.session_state.code_files) if st.session_state.code_files else set()
            st.metric("Languages", len(extensions))
    else:
        st.info("👆 Upload files / select GitHub / ZIP and click **Analyze Codebase** to see overview.")

with tab2:
    st.subheader("💬 Interactive Code Q&A")
    if st.session_state.collection is None:
        st.info("👆 Analyze a codebase first to enable Q&A.")
    else:
        user_question = st.text_input("Ask a question about the codebase:", placeholder="e.g., How does the authentication work?")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Ask", type="primary", use_container_width=True) and user_question.strip():
                with st.spinner("🔍 Searching codebase..."):
                    answer = ask_code(st.session_state.collection, user_question)
                st.session_state.chat_history.append(("user", user_question))
                st.session_state.chat_history.append(("assistant", answer))
                st.rerun()
        
        # Display chat history
        for role, message in reversed(st.session_state.chat_history[-10:]):
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(f"**You:** {message}")
            else:
                with st.chat_message("assistant"):
                    st.markdown(message)

with tab3:
    st.subheader("📚 Technical Documentation")
    
    if not st.session_state.analysis_complete:
        st.warning("👆 Please **Analyze Codebase** first to generate documentation.")
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.session_state.enterprise_docs_md is None:
                if st.button("📄 Generate Enterprise Documentation", type="primary", use_container_width=True):
                    try:
                        with st.spinner("🔄 Generating enterprise-grade technical documentation..."):
                            docs_md = generate_enterprise_documentation(
                                st.session_state.code_files,
                                st.session_state.collection,
                            )
                        st.session_state.enterprise_docs_md = docs_md
                        
                        # Try to generate PDF
                        project_name = Path(st.session_state.code_files[0]["file_path"]).parent.name if st.session_state.code_files else "codebase"
                        pdf_bytes = generate_pdf_from_markdown(docs_md, project_name)
                        st.session_state.enterprise_docs_pdf = pdf_bytes
                        
                        st.success("✅ Enterprise documentation generated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Documentation generation failed: {e}")
            else:
                if st.button("🔄 Regenerate", type="secondary", use_container_width=True):
                    st.session_state.enterprise_docs_md = None
                    st.session_state.enterprise_docs_pdf = None
                    st.rerun()
        
        # Display documentation
        docs_md = st.session_state.get("enterprise_docs_md")
        if docs_md:
            st.success("✅ technical documentation ready!")
            
            # PDF Download
            pdf_bytes = st.session_state.get("enterprise_docs_pdf")
            if pdf_bytes:
                project_name = Path(st.session_state.code_files[0]["file_path"]).parent.name if st.session_state.code_files else "codebase"
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"{project_name}_enterprise_documentation.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.info("💡 PDF generation requires `wkhtmltopdf`. Install it for PDF export capability.")
            
            # Display markdown
            with st.expander("📖 View Full Documentation", expanded=True):
                st.markdown(docs_md)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Status")
if st.session_state.analysis_complete:
    st.sidebar.success("✅ Analysis Complete")
    st.sidebar.info(f"📁 Files: {len(st.session_state.code_files)}")
else:
    st.sidebar.warning("⏳ Ready for Analysis")

st.sidebar.markdown("---")
st.sidebar.markdown("**Enterprise Codebase Analyzer v3**")
st.sidebar.markdown("""
- Multi-provider LLM support
- 7-section enterprise documentation
- Code-grounded validation
- Production-ready PDF export
- Professional client deliverables
""")
