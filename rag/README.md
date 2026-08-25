# Real-retriever inputs

The paper evaluates top-5 BM25 and Contriever retrieval over `wikimedia/wikipedia` (`20231101.en`) without reranking. The corpus and indexing pipeline are not bundled in this repository.

`inference/generator/.sh/rag_test.sh` accepts pre-retrieved records through `RAG_INPUT_FILE`. Each record must contain:

```json
{
  "id": "s0",
  "question": "...",
  "gt_answer": ["..."],
  "bm25-facts": ["passage 1", "passage 2"],
  "Contriever-facts": ["passage 1", "passage 2"]
}
```

The reported setup retrieves five passages. Contriever uses `facebook/contriever`, a maximum embedding input length of 256 tokens, and 100 KNN candidates.
