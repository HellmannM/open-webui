## HNSW Index

- owui creates IVFFLAT Index, need to manually drop it and create hnsw.
- Drop the IVFFLAT index
  ```SQL
  DROP INDEX idx_document_chunk_vector;
  ```
- Create your own HNSW index, e.g.
  ```SQL
  CREATE INDEX idx_document_chunk_vector_hnsw
  ON document_chunk USING hnsw (vector vector_cosine_ops) WITH (m = 16, ef_construction = 64);
  ```
- tune ef_search per session (SET ivfflat.probes = … for IVFFLAT or SET hnsw.ef_search = … for HNSW)
- make persistent / survive restarts:
  strip/patch the Open WebUI initializer so it doesn’t recreate the IVFFLAT index on boot.

## GIN Index
- A plain B-tree on collection_name is already set up by the same initializer
  `CREATE INDEX idx_document_chunk_collection_name ON document_chunk (collection_name);`
- Metadata lives in the JSONB column vmetadata. Nothing in core creates indexes for it, so every
  `WHERE vmetadata->>'doc_type' = 'review'` turns into a sequential scan.
  If you know which keys you filter on, add expression indexes yourself, e.g.
  ```SQL
  CREATE INDEX idx_dc_doc_type ON document_chunk ((vmetadata->>'doc_type'));
  CREATE INDEX idx_dc_author   ON document_chunk ((vmetadata->>'author'));
  -- or for ad‑hoc JSON predicates
  CREATE INDEX idx_dc_meta_gin ON document_chunk USING gin (vmetadata jsonb_path_ops);
  ```
- After adding them, `ANALYZE document_chunk;` so the planner notices.
- To benefit from those indexes you must push the predicates into SQL.
  When you call `get_sources_from_items` today it asks pgvector to do an ANN search and then
  we filter chunks in Python (which happens after the DB work, so indexes don’t help).
  If you migrate a filter into the database — e.g. by calling
  `VECTOR_DB_CLIENT.query(..., filter={'doc_type': 'review'})`
  before the vector search or by extending pgvector’s query to join on metadata — you’ll actually
  use the B-tree/GIN indexes you created.
