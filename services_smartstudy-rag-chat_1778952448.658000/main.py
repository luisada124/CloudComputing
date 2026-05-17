import json
import logging
import os
import re

import certifi
import functions_framework
import vertexai
from pymongo import MongoClient
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_vertexai import ChatVertexAI


logging.basicConfig(level=logging.INFO)

# MongoDB config
ATLAS_URI = os.environ["ATLAS_URI"]
DB_NAME = os.environ.get("DB_NAME", "smartstudy")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "lecture_chunks")
VECTOR_INDEX_NAME = os.environ.get("VECTOR_INDEX_NAME", "vector_index")

# Embedding config
VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID", "cloud-project-495915")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-005")

# Gemini config
GEMINI_PROJECT_ID = os.environ.get("GEMINI_PROJECT_ID", VERTEX_PROJECT_ID)
GEMINI_LOCATION = os.environ.get("GEMINI_LOCATION", "global")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Retrieval safety config
MIN_VECTOR_SCORE = float(os.environ.get("MIN_VECTOR_SCORE", "0.0"))
MAX_CONTEXT_CHARS_PER_CHUNK = int(os.environ.get("MAX_CONTEXT_CHARS_PER_CHUNK", "1800"))


def json_response(data, status=200):
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    }

    return json.dumps(data, ensure_ascii=False, indent=2), status, headers


def text_response(text, status=200):
    headers = {
        "Content-Type": "text/plain; charset=utf-8",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    }

    return text, status, headers


def wants_text_response(request, body):
    query_format = request.args.get("format", "").lower()
    body_format = str(body.get("format", "")).lower()
    accept_header = request.headers.get("Accept", "").lower()

    return (
        query_format in {"text", "plain", "md", "markdown"}
        or body_format in {"text", "plain", "md", "markdown"}
        or "text/plain" in accept_header
    )


def get_mongo_collection():
    client = MongoClient(
        ATLAS_URI,
        serverSelectionTimeoutMS=30000,
        tls=True,
        tlsCAFile=certifi.where(),
    )

    client.admin.command("ping")
    collection = client[DB_NAME][COLLECTION_NAME]

    return client, collection


def list_available_sources():
    mongo_client, collection = get_mongo_collection()

    try:
        sources = collection.distinct("source")
        sources = [source for source in sources if source]
        return sorted(sources)

    finally:
        mongo_client.close()


def embed_query(question: str):
    """
    Convert the user question/topic into an embedding.
    The query uses RETRIEVAL_QUERY.
    The PDF chunks used RETRIEVAL_DOCUMENT in the ingestion service.
    """

    vertexai.init(
        project=VERTEX_PROJECT_ID,
        location=VERTEX_LOCATION,
    )

    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)

    inputs = [
        TextEmbeddingInput(question, "RETRIEVAL_QUERY")
    ]

    embeddings = model.get_embeddings(inputs)

    return embeddings[0].values


def is_valid_chunk(chunk):
    text = str(chunk.get("text", "")).strip()
    source = str(chunk.get("source", "")).strip()
    score = float(chunk.get("score", 0) or 0)

    if not text:
        return False

    if not source:
        return False

    if score < MIN_VECTOR_SCORE:
        return False

    return True


def retrieve_top_chunks(query_vector, limit=3):
    mongo_client, collection = get_mongo_collection()

    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": 100,
                    "limit": 30,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "source": 1,
                    "page": 1,
                    "chunk_index": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        results = list(collection.aggregate(pipeline))

        valid_results = [
            chunk for chunk in results
            if is_valid_chunk(chunk)
        ]

        return valid_results[:limit]

    finally:
        mongo_client.close()


def build_context(chunks):
    """
    Builds context using stable source labels.
    Gemini must cite using [S1], [S2], etc.
    """

    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown source")
        page = chunk.get("page", "unknown page")
        chunk_index = chunk.get("chunk_index", "unknown chunk")
        text = str(chunk.get("text", "")).strip()

        if MAX_CONTEXT_CHARS_PER_CHUNK > 0:
            text = text[:MAX_CONTEXT_CHARS_PER_CHUNK]

        context_parts.append(
            f"[S{i}] File: {source} | Page: {page} | Chunk: {chunk_index}\n"
            f"{text}"
        )

    return "\n\n".join(context_parts)


def build_sources(chunks):
    sources = []

    for i, chunk in enumerate(chunks, start=1):
        sources.append(
            {
                "label": f"S{i}",
                "source": chunk.get("source"),
                "page": chunk.get("page"),
                "chunk_index": chunk.get("chunk_index"),
                "score": round(float(chunk.get("score", 0) or 0), 4),
                "text_preview": str(chunk.get("text", ""))[:300],
            }
        )

    return sources


def format_sources_text(sources):
    if not sources:
        return "Sources:\n- No sources found."

    lines = ["Sources:"]

    for source in sources:
        lines.append(
            f"- [{source['label']}] "
            f"{source['source']} | page {source['page']} | "
            f"chunk {source['chunk_index']} | score {source['score']}"
        )

    return "\n".join(lines)


def no_sources_message(mode, requested_text):
    available_sources = list_available_sources()

    return {
        "mode": mode,
        "requested": requested_text,
        "answer": (
            "I could not find enough relevant source material in MongoDB to answer this request. "
            "This can happen if the PDF was not uploaded yet, the ingestion pipeline has not finished, "
            "the document has no embeddings, or the Vector Search index is not ready."
        ),
        "available_sources": available_sources,
        "sources": [],
    }


def format_no_sources_text(payload):
    lines = [
        "SmartStudy could not find relevant sources.",
        "=" * 80,
        "",
        f"Mode: {payload.get('mode')}",
        f"Requested: {payload.get('requested')}",
        "",
        payload.get("answer", ""),
        "",
        "Available uploaded sources:",
    ]

    available_sources = payload.get("available_sources") or []

    if available_sources:
        for source in available_sources:
            lines.append(f"- {source}")
    else:
        lines.append("- No uploaded sources found in the configured MongoDB collection.")

    lines.extend(
        [
            "",
            "Checks:",
            "- Confirm the PDF was uploaded to the Cloud Storage bucket.",
            "- Confirm the ingestion Cloud Function finished successfully.",
            "- Confirm MongoDB documents contain the fields: text, embedding, source, page.",
            "- Confirm the Vector Search index is READY.",
        ]
    )

    return "\n".join(lines)


def extract_relevance_keywords(text):
    """
    Extracts meaningful keywords from the user question/topic.
    Used as a guardrail to avoid answering with unrelated retrieved chunks.
    """

    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())

    stopwords = {
        "what", "when", "where", "which", "who", "why", "how",
        "the", "and", "for", "with", "from", "that", "this",
        "about", "into", "onto", "your", "does", "have", "has",
        "give", "make", "create", "generate", "explain", "describe",
        "quiz", "question", "questions", "answer", "answers",
        "type", "types", "kind", "kinds", "topic", "project",
        "please", "based", "using", "material", "document",
    }

    keywords = []

    for token in tokens:
        if token in stopwords:
            continue

        if len(token) < 4:
            continue

        # simple plural normalization: dogs -> dog, embeddings -> embedding
        if token.endswith("s") and len(token) > 4:
            token = token[:-1]

        keywords.append(token)

    return sorted(set(keywords))


def chunks_are_relevant_to_request(request_text, chunks):
    """
    Checks if retrieved chunks are at least minimally related to the request.

    This prevents cases like:
    /quiz dogs types
    -> retrieved Paillier cryptography chunks
    -> Gemini generates unrelated quiz
    """

    keywords = extract_relevance_keywords(request_text)

    # If no useful keywords are extractable, do not block.
    # Example: very short questions.
    if not keywords:
        return True, [], keywords

    combined_context = " ".join(
        (
            str(chunk.get("source", "")) + " " +
            str(chunk.get("text", ""))
        ).lower()
        for chunk in chunks
    )

    matched_keywords = []

    for keyword in keywords:
        if keyword in combined_context:
            matched_keywords.append(keyword)

    # Require at least one meaningful keyword to appear in retrieved material.
    is_relevant = len(matched_keywords) > 0

    return is_relevant, matched_keywords, keywords


def irrelevant_context_message(mode, requested_text, checked_keywords):
    return {
        "mode": mode,
        "requested": requested_text,
        "answer": (
            "I could not find relevant information about this request in the uploaded material. "
            "The retrieved chunks did not match the topic closely enough, so I will not generate "
            "an answer or quiz from unrelated sources."
        ),
        "checked_keywords": checked_keywords,
        "sources": [],
    }


def format_irrelevant_context_text(payload):
    lines = [
        "SmartStudy could not find relevant material.",
        "=" * 80,
        "",
        f"Mode: {payload.get('mode')}",
        f"Requested: {payload.get('requested')}",
        "",
        payload.get("answer", ""),
    ]

    checked_keywords = payload.get("checked_keywords") or []

    if checked_keywords:
        lines.extend(
            [
                "",
                "Keywords checked:",
                "- " + ", ".join(checked_keywords),
            ]
        )

    lines.extend(
        [
            "",
            "No answer was generated because the retrieved context was not relevant enough.",
        ]
    )

    return "\n".join(lines)

def get_gemini_model(temperature=0.2, max_tokens=1800):
    """
    LangChain wrapper around Gemini through Vertex AI.
    This is the model component used inside the LCEL chains.
    """

    return ChatVertexAI(
        model=GEMINI_MODEL,
        project=GEMINI_PROJECT_ID,
        location=GEMINI_LOCATION,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# -------------------------
# LCEL CHAINS
# -------------------------

def create_retrieval_chain(limit=3):
    """
    LCEL retrieval chain:

    input text
        ↓
    embed_query
        ↓
    MongoDB Atlas Vector Search
        ↓
    valid retrieved chunks
    """

    return (
        RunnableLambda(lambda text: embed_query(text))
        | RunnableLambda(lambda query_vector: retrieve_top_chunks(query_vector, limit=limit))
    )


def create_answer_chain():
    """
    LCEL answer generation chain:

    context + question
        ↓
    ChatPromptTemplate
        ↓
    ChatVertexAI Gemini model
        ↓
    StrOutputParser
        ↓
    final answer string
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are SmartStudy, a formal academic tutor.

Use the provided context to answer the student's question.

Strict rules:
1. Answer only using the provided context.
2. If the context is insufficient, say that the uploaded material does not contain enough information.
3. Cite sources only using labels like [S1], [S2], [S3].
4. Do not invent file names, pages, chunks, citations, or facts.
5. Explain clearly and academically.
6. Include a short "Key idea" section.
7. End with one useful follow-up study question.

Output format:

Answer:
...

Key idea:
...

Follow-up study question:
...
"""
            ),
            (
                "human",
                """
Context:
{context}

Student question:
{question}
"""
            ),
        ]
    )

    model = get_gemini_model(
        temperature=0.2,
        max_tokens=1800,
    )

    return prompt | model | StrOutputParser()


def create_quiz_chain():
    """
    LCEL quiz generation chain:

    context + topic
        ↓
    ChatPromptTemplate
        ↓
    ChatVertexAI Gemini model
        ↓
    StrOutputParser
        ↓
    quiz string
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are SmartStudy, a formal academic tutor.

Create a quiz based only on the provided context.

Strict rules:
1. Use only the context.
2. Create exactly 5 questions.
3. Mix question types:
   - 2 short-answer questions
   - 2 conceptual explanation questions
   - 1 multiple-choice question
4. After the quiz, provide an answer key.
5. Every answer in the answer key must cite sources using labels like [S1], [S2].
6. Do not invent facts outside the context.
7. Keep the wording clear and useful for exam preparation.

Output format:

Quiz: <topic>

Question 1 - Short answer:
...

Question 2 - Short answer:
...

Question 3 - Conceptual explanation:
...

Question 4 - Conceptual explanation:
...

Question 5 - Multiple choice:
A. ...
B. ...
C. ...
D. ...

Answer key:
1. ...
2. ...
3. ...
4. ...
5. ...
"""
            ),
            (
                "human",
                """
Context:
{context}

Quiz topic:
{topic}
"""
            ),
        ]
    )

    model = get_gemini_model(
        temperature=0.3,
        max_tokens=2600,
    )

    return prompt | model | StrOutputParser()


def generate_answer_with_lcel(question: str, chunks):
    context = build_context(chunks)

    if not context.strip():
        return "The uploaded material does not contain enough information to answer this question."

    answer_chain = create_answer_chain()

    return answer_chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )


def generate_quiz_with_lcel(topic: str, chunks):
    context = build_context(chunks)

    if not context.strip():
        return "The uploaded material does not contain enough information to generate a quiz."

    quiz_chain = create_quiz_chain()

    return quiz_chain.invoke(
        {
            "topic": topic,
            "context": context,
        }
    )


def format_answer_text(question, answer, sources):
    return (
        f"SmartStudy Answer\n"
        f"{'=' * 80}\n\n"
        f"Question:\n{question}\n\n"
        f"{answer}\n\n"
        f"{'=' * 80}\n"
        f"{format_sources_text(sources)}\n"
    )


def format_quiz_text(topic, quiz, sources):
    return (
        f"SmartStudy Quiz Mode\n"
        f"{'=' * 80}\n\n"
        f"Topic:\n{topic}\n\n"
        f"{quiz}\n\n"
        f"{'=' * 80}\n"
        f"{format_sources_text(sources)}\n"
    )


@functions_framework.http
def ask(request):
    if request.method == "OPTIONS":
        return json_response({}, status=204)

    if request.method == "GET":
        if request.args.get("sources", "").lower() == "true":
            return json_response(
                {
                    "database": DB_NAME,
                    "collection": COLLECTION_NAME,
                    "available_sources": list_available_sources(),
                }
            )

        return json_response(
            {
                "service": "SmartStudy RAG Chat",
                "status": "running",
                "architecture": "MongoDB Atlas Vector Search + LangChain LCEL + Gemini 2.5 Flash",
                "usage": {
                    "normal_question": {
                        "method": "POST",
                        "body": {
                            "question": "What is cloud computing?"
                        },
                    },
                    "quiz_mode": {
                        "method": "POST",
                        "body": {
                            "question": "/quiz Paillier cryptosystem"
                        },
                    },
                    "readable_terminal_output": {
                        "url_suffix": "?format=text"
                    },
                    "list_sources": {
                        "method": "GET",
                        "url_suffix": "?sources=true"
                    },
                },
            }
        )

    if request.method != "POST":
        return json_response(
            {
                "error": "Use POST with JSON body: {\"question\": \"...\"}"
            },
            status=405,
        )

    try:
        body = request.get_json(silent=True) or {}
        text_mode = wants_text_response(request, body)

        question = body.get("question", "").strip()

        if not question:
            return json_response(
                {
                    "error": "Missing field: question"
                },
                status=400,
            )

        is_quiz_mode = question.lower().startswith("/quiz")

        if is_quiz_mode:
            topic = question[5:].strip()

            if not topic:
                return json_response(
                    {
                        "error": "Use /quiz followed by a topic, for example: /quiz Paillier cryptosystem"
                    },
                    status=400,
                )

            logging.info(f"Received quiz request for topic: {topic}")

            retrieval_chain = create_retrieval_chain(limit=5)
            chunks = retrieval_chain.invoke(topic)

            logging.info(f"Retrieved {len(chunks)} chunks for quiz mode using LCEL")

            if not chunks:
                payload = no_sources_message(
                    mode="quiz",
                    requested_text=topic,
                )

                if text_mode:
                    return text_response(format_no_sources_text(payload))

                return json_response(payload, status=200)

            is_relevant, matched_keywords, checked_keywords = chunks_are_relevant_to_request(
                topic,
                chunks,
            )

            if not is_relevant:
                payload = irrelevant_context_message(
                    mode="quiz",
                    requested_text=topic,
                    checked_keywords=checked_keywords,
                )

                if text_mode:
                    return text_response(format_irrelevant_context_text(payload))

                return json_response(payload, status=200)

            quiz = generate_quiz_with_lcel(topic, chunks)
            sources = build_sources(chunks)

            if not sources:
                payload = no_sources_message(
                    mode="quiz",
                    requested_text=topic,
                )

                if text_mode:
                    return text_response(format_no_sources_text(payload))

                return json_response(payload, status=200)

            if text_mode:
                return text_response(format_quiz_text(topic, quiz, sources))

            return json_response(
                {
                    "mode": "quiz",
                    "topic": topic,
                    "quiz": quiz,
                    "sources": sources,
                },
                status=200,
            )

        logging.info(f"Received question: {question}")

        retrieval_chain = create_retrieval_chain(limit=3)
        chunks = retrieval_chain.invoke(question)

        logging.info(f"Retrieved {len(chunks)} chunks from MongoDB using LCEL")

        if not chunks:
            payload = no_sources_message(
                mode="question",
                requested_text=question,
            )

            if text_mode:
                return text_response(format_no_sources_text(payload))

            return json_response(payload, status=200)

        is_relevant, matched_keywords, checked_keywords = chunks_are_relevant_to_request(
            question,
            chunks,
        )

        if not is_relevant:
            payload = irrelevant_context_message(
                mode="question",
                requested_text=question,
                checked_keywords=checked_keywords,
            )

            if text_mode:
                return text_response(format_irrelevant_context_text(payload))

            return json_response(payload, status=200)

        answer = generate_answer_with_lcel(question, chunks)
        sources = build_sources(chunks)

        if not sources:
            payload = no_sources_message(
                mode="question",
                requested_text=question,
            )

            if text_mode:
                return text_response(format_no_sources_text(payload))

            return json_response(payload, status=200)

        if text_mode:
            return text_response(format_answer_text(question, answer, sources))

        return json_response(
            {
                "question": question,
                "answer": answer,
                "sources": sources,
            },
            status=200,
        )

    except Exception as e:
        logging.exception("Error while answering question")

        return json_response(
            {
                "error": str(e),
                "type": type(e).__name__,
            },
            status=500,
        )