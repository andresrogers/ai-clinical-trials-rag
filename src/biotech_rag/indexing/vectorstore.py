"""Chroma vectorstore helper utilities with version compatibility.

This module provides a resilient `init_chroma` that attempts the legacy
`Settings`-based initialization and falls back to the newer `chromadb.Client()`
constructor. It also provides `upsert_documents` and a helper to get/create
collections across different chroma client APIs.
"""

from pathlib import Path

try:
    import chromadb

    # Settings may not be present in newer chroma releases
    try:
        from chromadb.config import Settings  # type: ignore
    except Exception:
        Settings = None  # type: ignore
except Exception:
    chromadb = None  # type: ignore


def _get_or_create_collection(client, name: str):
    """Get or create a collection in a Chroma client across API variants.

    Args:
        client: chromadb client instance
        name: collection name

    Returns:
        collection object
    """
    if hasattr(client, "get_or_create_collection"):
        return client.get_or_create_collection(name=name)

    if hasattr(client, "get_collection") and hasattr(client, "create_collection"):
        try:
            return client.get_collection(name)
        except Exception:
            return client.create_collection(name)

    if hasattr(client, "create_collection"):
        return client.create_collection(name)

    raise RuntimeError("Cannot create or access collection on Chroma client; incompatible API")


def init_chroma(
    persist_dir: Path, collection_name: str = "clinical_trials"
) -> tuple[object, object]:
    """Initialize and return a Chroma client and collection.

    The function attempts the legacy Settings-based initialization (duckdb+parquet)
    and falls back to calling `chromadb.Client()` for newer Chroma versions.

    Args:
        persist_dir: Path to persist Chroma database files.
        collection_name: Collection name to get or create.

    Returns:
        (client, collection)
    """
    if chromadb is None:
        raise RuntimeError("chromadb is not installed")

    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = None

    # Try legacy Settings-based constructor first
    if "Settings" in globals() and Settings is not None:
        try:
            client = chromadb.Client(
                Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(persist_dir))
            )
        except Exception:
            client = None

    # Fallback to persistent client when Settings are unavailable
    if client is None:
        try:
            if hasattr(chromadb, "PersistentClient"):
                client = chromadb.PersistentClient(path=str(persist_dir))
            else:
                # Last resort: non-persistent client (may be in-memory depending on chroma version)
                client = chromadb.Client()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Chroma client: {e}")

    collection = _get_or_create_collection(client, collection_name)
    return client, collection


def upsert_documents(collection, docs: list[dict], embeddings: list[list[float]] | None = None):
    """Upsert docs into the collection.

    Args:
        collection: chroma collection object
        docs: list of {'id': str, 'text': str, 'metadata': {...}}
        embeddings: optional list of embedding vectors matching docs order
    """
    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metadatas = [d.get("metadata", {}) for d in docs]

    if embeddings is not None:
        # Using upsert instead of add to handle existing records
        collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    else:
        collection.upsert(ids=ids, documents=texts, metadatas=metadatas)


def validate_vectorstore_for_nct(persist_dir: Path, collection_name: str, nct_id: str) -> bool:
    """
    Check if the vectorstore exists and has chunks for the given NCT ID.

    Args:
        persist_dir: Path to Chroma persist directory.
        collection_name: Collection name.
        nct_id: NCT ID to check.

    Returns:
        True if chunks exist for NCT, False otherwise.
    """
    try:
        client, collection = init_chroma(persist_dir, collection_name)
        # Query for chunks with this NCT in metadata
        results = collection.get(where={"nct_id": nct_id}, limit=1)
        return len(results.get("ids", [])) > 0
    except Exception as e:
        print(f"Warning: Vectorstore validation failed for NCT {nct_id}: {e}")
        return False


def get_chroma_class():
    """Get the Chroma class with telemetry disabled.

    This function dynamically imports the Chroma class, preferring the dedicated
    langchain-chroma package to avoid LangChain deprecation warnings. It also
    sets environment variables to disable Chroma telemetry and suppresses
    chromadb INFO logs.

    Returns:
        Chroma class from langchain_chroma or langchain_community.vectorstores

    Raises:
        ImportError: If no Chroma wrapper is available
    """
    import os
    import importlib
    import logging

    # Disable Chroma telemetry
    os.environ.setdefault("CHROMA_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("CHROMADB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("CHROMA_TELEMETRY", "0")

    # Suppress chromadb INFO logs
    for name in ("chromadb", "chromadb.telemetry", "chromadb.telemetry.product", "chromadb.telemetry.product.posthog"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Prefer langchain_chroma, fall back to langchain_community.vectorstores
    ChromaClass = None
    for module_name in ("langchain_chroma", "langchain_community.vectorstores"):
        try:
            mod = importlib.import_module(module_name)
            ChromaClass = getattr(mod, "Chroma")
            break
        except Exception:
            continue

    if ChromaClass is None:
        raise ImportError("No Chroma wrapper available; install langchain_chroma or langchain_community.vectorstores")

    return ChromaClass
