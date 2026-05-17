## Environment Variables

The following environment variables must be configured in Cloud Run.

### Ingestion service: `smartstudy-ingest`

| Variable | Description |
|---|---|
| `ATLAS_URI` | MongoDB Atlas connection string. Should be stored as a secret. |
| `DB_NAME` | MongoDB database name, e.g. `smartstudy`. |
| `COLLECTION_NAME` | MongoDB collection name, e.g. `lecture_chunks`. |
| `VERTEX_PROJECT_ID` | Google Cloud project used for Vertex AI embeddings. |
| `VERTEX_LOCATION` | Vertex AI region, e.g. `us-central1`. |
| `EMBEDDING_MODEL` | Embedding model, e.g. `text-embedding-005`. |

### Chat service: `smartstudy-rag-chat`

| Variable | Description |
|---|---|
| `ATLAS_URI` | MongoDB Atlas connection string. Should be stored as a secret. |
| `DB_NAME` | MongoDB database name. |
| `COLLECTION_NAME` | MongoDB collection where chunks and embeddings are stored. |
| `VECTOR_INDEX_NAME` | MongoDB Atlas Vector Search index name. |
| `VERTEX_PROJECT_ID` | Google Cloud project used for query embeddings. |
| `VERTEX_LOCATION` | Vertex AI region for embeddings. |
| `EMBEDDING_MODEL` | Same embedding model used during ingestion. |
| `GEMINI_PROJECT_ID` | Google Cloud project used for Gemini. |
| `GEMINI_LOCATION` | Gemini location, e.g. `global`. |
| `GEMINI_MODEL` | Gemini model, e.g. `gemini-2.5-flash`. |


## Using the Chat Service

After deploying the `smartstudy-rag-chat` service, you can ask questions from the terminal using `curl`. The service receives a question, retrieves the most relevant PDF chunks from MongoDB Atlas Vector Search, and uses Gemini to generate an answer with citations.

Example:

curl -s -X POST "https://smartstudy-rag-chat-806559489994.europe-west1.run.app?format=text" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the class in Paillier cryptosystem?"}'

The service also supports a quiz mode. To use it, start the question with /quiz followed by the topic. It will retrieve relevant chunks and generate a 5-question quiz with an answer key.

Example:

curl -s -X POST "https://smartstudy-rag-chat-806559489994.europe-west1.run.app?format=text" \
  -H "Content-Type: application/json" \
  -d '{"question":"/quiz Paillier cryptosystem"}'
