"""Vald vector store."""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple

import grpc
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)

from vald.v1.payload import payload_pb2
from vald.v1.vald import (
    flush_pb2_grpc,
    insert_pb2_grpc,
    remove_pb2_grpc,
    search_pb2_grpc,
    update_pb2_grpc,
    upsert_pb2_grpc,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8081
DEFAULT_SIMILARITY_TOP_K = 10
DEFAULT_SEARCH_RADIUS = -1.0
DEFAULT_SEARCH_EPSILON = 0.01
DEFAULT_SEARCH_TIMEOUT_NS = 3_000_000_000

VALD_SEARCH_CONFIG_KWARGS = "vald_saerch_config"


class ValdVectorStore(BasePydanticVectorStore):
    """
    Vald Vector Store.

    Vald (https://vald.vdaas.org/) is a distributed fast approximate
    nearest-neighbor dense vector search engine.

    Args:
        host: Vald gateway host.
        port: Vald gateway port.
        secure: When True, open a TLS gRPC channel.
        grpc_options: Extra gRPC channel options.
        channel: Pre-built `grpc.Channel`. When supplied, `host` /
            `port` / `secure` are ignored and the caller owns the
            channel lifecycle.
        insert_mode: When True, `add()` uses Vald's Insert RPC (errors
            on an existing id). When False(defalut), uses Upsert
            (inserts or updates depending on whether the id exists).
        skip_strict_exist_check: Forwarded to Insert / Upsert / Update /
            Remove configs.
        search_radius: Default NGT radius parameter (`-1` = no limit).
        search_epsilon: Default NGT epsilon parameter.
        search_timeout_ns: Default per-RPC search timeout in nanoseconds.

    Examples:
        `pip install llama-index-vector-stores-vald vald-client-python`

        ```
        from llama_index.vector_stores.vald import ValdVectorStore

        store = ValdVectorStore(host="localhost", port=8081)
        ```

    """

    stores_text: bool = False
    flat_metadata: bool = False
    is_embedding_query: bool = True

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    secure: bool = False
    grpc_options: Optional[List[Tuple[str, Any]]] = None
    insert_mode: bool = False
    skip_strict_exist_check: bool = True
    search_radius: float = DEFAULT_SEARCH_RADIUS
    search_epsilon: float = DEFAULT_SEARCH_EPSILON
    search_timeout_ns: int = DEFAULT_SEARCH_TIMEOUT_NS

    _channel: Any = PrivateAttr(default=None)
    _owns_channel: bool = PrivateAttr(default=False)
    _insert_stub: Any = PrivateAttr(default=None)
    _search_stub: Any = PrivateAttr(default=None)
    _update_stub: Any = PrivateAttr(default=None)
    _upsert_stub: Any = PrivateAttr(default=None)
    _remove_stub: Any = PrivateAttr(default=None)
    _flush_stub: Any = PrivateAttr(default=None)

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        secure: bool = False,
        grpc_options: Optional[List[Tuple[str, Any]]] = None,
        channel: Any = None,
        insert_mode: bool = False,
        skip_strict_exist_check: bool = True,
        search_radius: float = DEFAULT_SEARCH_RADIUS,
        search_epsilon: float = DEFAULT_SEARCH_EPSILON,
        search_timeout_ns: int = DEFAULT_SEARCH_TIMEOUT_NS,
    ) -> None:
        super().__init__(
            stores_text=False,
            host=host,
            port=port,
            secure=secure,
            grpc_options=grpc_options,
            insert_mode=insert_mode,
            skip_strict_exist_check=skip_strict_exist_check,
            search_radius=search_radius,
            search_epsilon=search_epsilon,
            search_timeout_ns=search_timeout_ns,
        )

        if channel is None:
            target = f"{host}:{port}"
            if secure:
                creds = grpc.ssl_channel_credentials()
                self._channel = grpc.secure_channel(
                    target,
                    creds,
                    options=grpc_options,
                )
            else:
                self._channel = grpc.insecure_channel(
                    target,
                    options=grpc_options,
                )
            self._owns_channel = True
        else:
            self._channel = channel
            self._owns_channel = False

        self._insert_stub = insert_pb2_grpc.InsertStub(self._channel)
        self._search_stub = search_pb2_grpc.SearchStub(self._channel)
        self._update_stub = update_pb2_grpc.UpdateStub(self._channel)
        self._upsert_stub = upsert_pb2_grpc.UpsertStub(self._channel)
        self._remove_stub = remove_pb2_grpc.RemoveStub(self._channel)
        self._flush_stub = flush_pb2_grpc.FlushStub(self._channel)

    @classmethod
    def class_name(cls) -> str:
        return "ValdVectorStore"

    @property
    def client(self) -> Any:
        return self._channel

    @property
    def insert_stub(self) -> Any:
        return self._insert_stub

    @property
    def search_stub(self) -> Any:
        return self._search_stub

    @property
    def update_stub(self) -> Any:
        return self._update_stub

    @property
    def upsert_stub(self) -> Any:
        return self._upsert_stub

    @property
    def remove_stub(self) -> Any:
        return self._remove_stub

    @property
    def flush_stub(self) -> Any:
        return self._flush_stub

    def close(self) -> None:
        if self._owns_channel and self._channel is not None:
            self._channel.close()
            self._channel = None

    def add(
        self,
        nodes: Sequence[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        if not nodes:
            return []

        if self.insert_mode:
            cfg = payload_pb2.Insert.Config(
                skip_strict_exist_check=self.skip_strict_exist_check,
            )
            stub_call = self._insert_stub.Insert
            request_cls = payload_pb2.Insert.Request
        else:
            cfg = payload_pb2.Upsert.Config(
                skip_strict_exist_check=self.skip_strict_exist_check,
            )
            stub_call = self._upsert_stub.Upsert
            request_cls = payload_pb2.Upsert.Request

        ids: List[str] = []
        for node in nodes:
            embedding = node.get_embedding()
            if embedding is None:
                raise ValueError(
                    f"Node {node.node_id} has no embedding; "
                    "an embedding is required to add a node to Vald."
                )
            vec = payload_pb2.Object.Vector(
                id=node.node_id,
                vector=list(embedding),
            )
            stub_call(request_cls(vector=vec, config=cfg))
            ids.append(node.node_id)

        return ids

    def bulk_add(
        self,
        nodes: Iterable[BaseNode],
        **bulk_kwargs: Any,
    ) -> List[str]:
        if self.insert_mode:
            cfg = payload_pb2.Insert.Config(
                skip_strict_exist_check=self.skip_strict_exist_check,
            )
            request_cls = payload_pb2.Insert.Request
            stream_call = self._insert_stub.StreamInsert
        else:
            cfg = payload_pb2.Upsert.Config(
                skip_strict_exist_check=self.skip_strict_exist_check,
            )
            request_cls = payload_pb2.Upsert.Request
            stream_call = self._upsert_stub.StreamUpsert

        sent: Set[str] = set()
        succeeded: List[str] = []

        def request_iter() -> Iterable[Any]:
            for node in nodes:
                embedding = node.get_embedding()
                if embedding is None:
                    raise ValueError(
                        f"Node {node.node_id} has no embedding; "
                        "an embedding is required to add a node to Vald."
                    )
                sent.add(node.node_id)
                yield request_cls(
                    vector=payload_pb2.Object.Vector(
                        id=node.node_id,
                        vector=list(embedding),
                    ),
                    config=cfg,
                )

        for response in stream_call(request_iter()):
            if response.HasField("location"):
                succeeded.append(response.location.uuid)

        failed = sent - set(succeeded)
        if failed:
            raise RuntimeError(
                f"bulk_add: {len(failed)} entries failed out of {len(sent)}. "
                f"Failed ids: {sorted(failed)}"
            )

        return succeeded

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Not yet supported: deleting by ref_doc_id is not implemented in
        ValdVectorStore. Use `delete_nodes(node_ids=[...])` instead.
        """
        raise NotImplementedError(
            "ValdVectorStore.delete(ref_doc_id) is not yet supported. "
            "Use delete_nodes(node_ids=[...]) instead."
        )

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[Any] = None,
        **delete_kwargs: Any,
    ) -> None:
        if filters is not None:
            raise NotImplementedError(
                "Filter-based deletion is not yet supported in ValdVectorStore. "
                "Pass node_ids explicitly for now."
            )
        if not node_ids:
            return
        cfg = payload_pb2.Remove.Config(
            skip_strict_exist_check=self.skip_strict_exist_check,
        )
        for node_id in node_ids:
            rid = payload_pb2.Object.ID(id=node_id)
            self._remove_stub.Remove(payload_pb2.Remove.Request(id=rid, config=cfg))

    def clear(self) -> None:
        """Issue Vald's Flush RPC, removing all vectors from the index."""
        self._flush_stub.Flush(payload_pb2.Flush.Request())

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Query Vald for the top-k approximate nearest neighbors of
        `queyr.query_embedding`.

        Per-call overrides for Vald's NGT search parameters can be passed
        via the `vald_saerch_config` kwarg:

        ```
        retriever.retrieve(
            "query",
            vald_saerch_config={"radius": 0.5, "epsilon": 0.05, "timeout_ns": 5_000_000_000},
        )
        ```

        Recognised keys: `radius`, `epsilon`, `timeout_ns`.
        Defaults come from the constructor.
        """
        if query.filters is not None:
            logger.warning(
                "ValdVectorStore does not yet support VectorStoreQuery.filters; "
                "they will be ignored for this query."
            )
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise ValueError(
                f"ValdVectorStore only supports DEFAULT query mode, got {query.mode}."
            )
        if query.query_embedding is None:
            raise ValueError("ValdVectorStore requires query.query_embedding.")

        overrides = kwargs.get(VALD_SEARCH_CONFIG_KWARGS) or {}
        if not isinstance(overrides, dict):
            raise TypeError(
                f"{VALD_SEARCH_CONFIG_KWARGS} must be a dict, got {type(overrides).__name__}."
            )

        num = query.similarity_top_k or DEFAULT_SIMILARITY_TOP_K
        cfg = payload_pb2.Search.Config(
            num=num,
            radius=overrides.get("radius", self.search_radius),
            epsilon=overrides.get("epsilon", self.search_epsilon),
            timeout=overrides.get("timeout_ns", self.search_timeout_ns),
        )
        request = payload_pb2.Search.Request(
            vector=list(query.query_embedding),
            config=cfg,
        )
        response = self._search_stub.Search(request)

        ids = [r.id for r in response.results]
        # Vald return distance (smaller = closer). We pass it through unchanged.
        # Strictly speaking this means smaller = more similar in the returned
        # `similarities` which inverts LlamaIndex's "higher = better" convention.
        similarities = [float(r.distance) for r in response.results]
        # Alternative: min-max normalize into [0, 1] so that closer = larger.
        # Disabled by default because the meaning of the absolute value changes
        # per query (it depends on the result set's min/max) and degenerates
        # when the result set has sewer tahn two items.
        #
        # dists = [float(r.distance) for r in response.results]
        # if dists:
        #     lo, hi = min(dists), max(dists)
        #     span = hi - lo
        #     if span == 0:
        #         similarities = [1.0 for _ in dists]
        #     else:
        #         similarities = [1.0 - (d - lo) / span for d in dists]
        # else:
        #     similarities = []
        return VectorStoreQueryResult(ids=ids, similarities=similarities, nodes=None)
