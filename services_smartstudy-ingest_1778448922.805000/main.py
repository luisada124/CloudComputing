import hashlib
import logging
import os
import tempfile
from datetime import datetime, timezone

import certifi
import functions_framework
import vertexai
from google.cloud import storage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient, UpdateOne
from pypdf import PdfReader
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel


logging.basicConfig(level=logging.INFO)


# MongoDB config
DB_NAME = os.environ.get("DB_NAME", "smartstudy")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "lecture_chunks")

# Vertex AI config
VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID", "cloud-project-495915")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-005")


def make_chunk_id(file_name, page, chunk_index, text):
    raw = f"{file_name}:{page}:{chunk_index}:{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def generate_embeddings(texts):
    """
    Generates one embedding per text chunk using Vertex AI.
    For document chunks, we use RETRIEVAL_DOCUMENT.
    Later, for user questions, we will use RETRIEVAL_QUERY.
    """

    if not texts:
        return []

    vertexai.init(
        project=VERTEX_PROJECT_ID,
        location=VERTEX_LOCATION,
    )

    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)

    embeddings = []
    batch_size = 32

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        inputs = [
            TextEmbeddingInput(text, "RETRIEVAL_DOCUMENT")
            for text in batch
        ]

        logging.info(
            f"Generating embeddings for batch "
            f"{start + 1}-{start + len(batch)} of {len(texts)}"
        )

        batch_embeddings = model.get_embeddings(inputs)

        embeddings.extend([
            embedding.values
            for embedding in batch_embeddings
        ])

    return embeddings


@functions_framework.cloud_event
def hello_gcs(cloud_event):
    data = cloud_event.data

    bucket_name = data.get("bucket")
    file_name = data.get("name")
    content_type = data.get("contentType")

    logging.info("=== SmartStudy ingestion started ===")
    logging.info(f"New file detected: gs://{bucket_name}/{file_name}")
    logging.info(f"Content type: {content_type}")

    if not file_name or not file_name.lower().endswith(".pdf"):
        logging.info("File is not a PDF. Skipping.")
        return

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_path = temp_file.name

    mongo_client = None

    try:
        # Step 1: Download PDF
        logging.info(f"Downloading PDF: {file_name}")
        blob.download_to_filename(temp_path)

        # Step 2: Extract text per page
        reader = PdfReader(temp_path)
        logging.info(f"PDF has {len(reader.pages)} pages")

        extracted_pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()

            extracted_pages.append({
                "page": page_number,
                "text": text,
            })

        total_chars = sum(len(page["text"]) for page in extracted_pages)

        logging.info(
            f"Extracted {total_chars} characters "
            f"from {len(extracted_pages)} pages"
        )

        # Step 3: Create chunks with page metadata
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        all_chunks = []

        for page_data in extracted_pages:
            if not page_data["text"]:
                continue

            page_chunks = splitter.create_documents(
                [page_data["text"]],
                metadatas=[{
                    "source": file_name,
                    "page": page_data["page"],
                }],
            )

            all_chunks.extend(page_chunks)

        logging.info(f"Split into {len(all_chunks)} chunks")

        for i, chunk in enumerate(all_chunks[:3]):
            preview = chunk.page_content[:200].replace("\n", " ")
            logging.info(
                f"Chunk {i + 1} "
                f"(page {chunk.metadata['page']}): {preview}"
            )

        if not all_chunks:
            logging.warning("No chunks were created. Stopping pipeline.")
            return

        # Step 4: Generate embeddings
        texts = [chunk.page_content for chunk in all_chunks]

        logging.info(f"Generating embeddings for {len(texts)} chunks")

        embeddings = generate_embeddings(texts)

        logging.info(f"Generated {len(embeddings)} embeddings")

        if len(embeddings) != len(all_chunks):
            raise ValueError(
                f"Embedding count mismatch: "
                f"{len(embeddings)} embeddings for {len(all_chunks)} chunks"
            )

        if embeddings:
            logging.info(f"Embedding dimension: {len(embeddings[0])}")

        # Step 5: Connect to MongoDB
        atlas_uri = os.environ["ATLAS_URI"]

        mongo_client = MongoClient(
            atlas_uri,
            serverSelectionTimeoutMS=30000,
            tls=True,
            tlsCAFile=certifi.where(),
        )

        mongo_client.admin.command("ping")
        logging.info("MongoDB connection successful.")

        db = mongo_client[DB_NAME]
        collection = db[COLLECTION_NAME]

        # Step 6: Store chunks + embeddings in MongoDB
        operations = []
        now = datetime.now(timezone.utc)

        for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
            page = chunk.metadata.get("page")
            source = chunk.metadata.get("source")

            chunk_id = make_chunk_id(
                file_name=source,
                page=page,
                chunk_index=i,
                text=chunk.page_content,
            )

            operations.append(
                UpdateOne(
                    {"_id": chunk_id},
                    {
                        "$set": {
                            "text": chunk.page_content,
                            "embedding": embedding,
                            "source": source,
                            "page": page,
                            "bucket": bucket_name,
                            "chunk_index": i,
                            "content_type": content_type,
                            "embedding_model": EMBEDDING_MODEL,
                            "vertex_location": VERTEX_LOCATION,
                            "updated_at": now,
                        },
                        "$setOnInsert": {
                            "created_at": now,
                        },
                    },
                    upsert=True,
                )
            )

        if operations:
            result = collection.bulk_write(operations, ordered=False)

            logging.info(
                f"MongoDB write complete. "
                f"Upserted: {result.upserted_count}, "
                f"Modified: {result.modified_count}, "
                f"Matched: {result.matched_count}"
            )
        else:
            logging.warning("No MongoDB operations to execute.")

        logging.info(
            f"Pipeline complete for {file_name} — "
            f"chunks and embeddings stored in MongoDB"
        )
        logging.info("=== SmartStudy ingestion finished ===")

    except Exception as e:
        logging.exception(f"Error processing PDF {file_name}: {e}")
        raise

    finally:
        if mongo_client:
            mongo_client.close()
            logging.info("MongoDB connection closed.")

        if os.path.exists(temp_path):
            os.remove(temp_path)
            logging.info("Temporary PDF deleted.")