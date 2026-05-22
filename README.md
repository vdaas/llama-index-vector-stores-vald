# LlamaIndex Vector_Stores Integration: Vald

Integration of [Vald](https://vald.vdaas.org/) - a highly scalable distributed
fast approximate nearest-neighbor dense vector search engine - as a LlamaIndex
`VectorStore`.

## Install

```bash
pip install llama-index-vector-stores-vald vald-client-python
```

## Quick start

```python
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.vald import ValdVectorStore

# Connecto to a Vald gateway. Defaults: localhost:8081, insecure channel.
vector_store = ValdVectorStore(host="localhost", port=8081)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

nodes = [
  TextNode(text="hello world", embedding=[0.1, 0.2, 0.3]),
  TextNode(text="goodbye world", embedding=[0.4, 0.5, 0.6]),
]

index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)

retriever = index.as_retriever(similarity_top_k=2)
results = retriever.retrieve("hello")
```

## Behaviour and limitations

Vald is primarily an ANN index keyed by string ID.
This integration does not yet persist node text or metadata.

- `stores_text = False` - node text is not currently persisted by this integration.
  Pair this store with a separate document store (e.g. `SimpleDocumentStore`) to recover node content.
- `flat_metadata = False` - metadata persistence is not yet implemented.
- `VectorStoreQuery.filters` is currently ignored with a warning.
- Only `VectorStoreQueryMode.DEFAULT` is supported.
  Sparse / hybrid / text-search modes raise `ValueError`.
- Vald's Search returns a `distance` (smaller = closer).
  This integration surfaces it as-is in `VectorStoreQueryResult.similarities`.

## Constructor arguments

| Argument | Default | Description |
|---|---|---|
| `host` | `localhost` | Vald gateway host. |
| `port` | `8081` | Vald gateway port. |
| `secure` | `False` | When `True`, open a TLS gRPC channel. |
| `grpc_options` | `None` | Extra gRPC channel options, e.g. `[("grpc.max_send_message_length", -1)]`. |
| `channel` | `None` | Pre-build `grpc.Channel`. When supplied, `host` / `port` / `secure` are ignored and the caller owns the channel lifecycle (not closed by `store.close()`). |
| `insert_mode` | `False` | When `True`, `add()` uses Vald's Insert RPC (errors on an existing id). When `False` (default), uses Upsert(inserts or updates depending on whether the id exists). |
| `skip_strict_exist_check` | `True` | Forwarded to Insert / Upsert / Update / Remove configs. |
| `search_radius` | `-1.0` | Default NGT radius parameter ( `-1` = no limit). |
| `search_epsilon` | `0.01` | Default NGT epsilon parameter. |
| `search_timeout_ns` | `3_000_000_000` | Default per-RPC timeout in nanoseconds. |

### Per-query overrides

`query` accepts a `vald_search_config` dict to override the init-time defaults on a per-call basis:

```python
retriever.retrieve(
  "my_query",
  vald_search_config={
    "radius": 0.5,
    "epsilon": 0.05,
    "timeout_ns": 5_000_000_000,
  },
)
```

Recognised keys: `radius`, `epsilon`, `timeuot_ns`.
`similarity_top_k` is taken from `VectorStoreQuery.similarity_top_k` (the standard LlamaIndex field) and falls back to `10`.

## Bulk ingestion

For large ingestion batches, use `bulk_add()` instead of `add()`:

```python
store.bulk_add(node_generator())
```

`bulk_add()` accepts any `Iterable[BaseNode]` (lists, generators, streams loaded from disk)
and uses Vald's bidirectional streaming RPC (`StreamInsert` / `StreamUpsert` depending on `insert_mode`).
Errors are aggregated: the stream runs to completion and a `RuntimeError` is raised at the end if any entries failed.

`add()` remains available for small batches and interactive use;
it sends one RPC per node and is simpler to reason about.

## Deletion

- `delete(ref_doc_id)` is not yet supported.
  Use `delete_nodes(node_ids=[...])` instead.
- `clear()` issues a Vald `Flush` RPC, which removes all vectors from the index.

## License

MIT. See [LICENSE](./LICENSE)
