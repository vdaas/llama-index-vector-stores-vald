"""Tests for ValdVectorStore. Vald gRPC stubs are mocked - no network access. """

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from llama_index.vector_stores.vald import ValdVectorStore
from llama_index.vector_stores.vald.base import VALD_SEARCH_CONFIG_KWARGS


def _make_store(monkeypatch: pytest.MonkeyPatch):
    import grpc

    monkeypatch.setattr(grpc, "insecure_channel", lambda *a, **kw: MagicMock(name="channel"))
    return ValdVectorStore(host="localhost", port=8081)


def test_is_base_pydantic_vector_store_subclass() -> None:
    assert issubclass(ValdVectorStore, BasePydanticVectorStore)


def test_class_name() -> None:
    assert ValdVectorStore.class_name() == "ValdVectorStore"


def test_default_flags() -> None:
    fields = ValdVectorStore.model_fields
    assert fields["stores_text"].default is False
    assert fields["flat_metadata"].default is False


def test_add_uses_upsert_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store(monkeypatch)
    store._upsert_stub = MagicMock()

    nodes = [
        TextNode(id_="n1", text="a", embedding=[0.1, 0.2, 0.3]),
        TextNode(id_="n2", text="b", embedding=[0.4, 0.5, 0.6]),
    ]
    ids = store.add(nodes)

    assert ids == ["n1", "n2"]
    assert store._upsert_stub.Upsert.call_count == 2
    sent_req = store._upsert_stub.Upsert.call_args_list[0].args[0]
    assert sent_req.vector.id == "n1"
    assert list(sent_req.vector.vector) == pytest.approx([0.1, 0.2, 0.3])


def test_add_uses_insert_when_insert_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    import grpc

    monkeypatch.setattr(grpc, "insecure_channel", lambda *a, **kw: MagicMock())
    store = ValdVectorStore(insert_mode=True)
    store._insert_stub = MagicMock()
    store._upsert_stub = MagicMock()

    nodes = [
        TextNode(id_="n1", text="a", embedding=[0.1, 0.2, 0.3]),
    ]
    _ = store.add(nodes)

    assert store._insert_stub.Insert.call_count == 1
    assert store._upsert_stub.Upsert.call_count == 0


def _location_response(uuid: str) -> SimpleNamespace:
    return SimpleNamespace(
        HasField=lambda field, _uuid=uuid: field == "location",
        location=SimpleNamespace(uuid=uuid),
    )


def _status_response() -> SimpleNamespace:
    return SimpleNamespace(
        HasField=lambda field: field == "status",
        status=SimpleNamespace(code=13, message="oops"),
    )


def test_bulk_add_uses_stream_upsert_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store(monkeypatch)

    received_requests: list[Any] = []

    def fake_stream_upsert(request_iter: Any) -> Any:
        for req in request_iter:
            received_requests.append(req)
            yield _location_response(req.vector.id)

    store._upsert_stub = MagicMock()
    store._upsert_stub.StreamUpsert.side_effect = fake_stream_upsert

    nodes = [
        TextNode(id_="n1", text="a", embedding=[0.1, 0.2, 0.3]),
        TextNode(id_="n2", text="b", embedding=[0.4, 0.5, 0.6]),
    ]
    ids = store.bulk_add(nodes)

    assert sorted(ids) == ["n1", "n2"]
    assert len(received_requests) == 2
    assert received_requests[0].vector.id == "n1"
    assert list(received_requests[0].vector.vector) == pytest.approx([0.1, 0.2, 0.3])


def test_bulk_add_uses_stream_insert_when_insert_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    import grpc

    monkeypatch.setattr(grpc, "insecure_channel", lambda *a, **kw: MagicMock())
    store = ValdVectorStore(insert_mode=True)

    def fake_stream_upsert(request_iter: Any) -> Any:
        for req in request_iter:
            yield _location_response(req.vector.id)

    store._insert_stub = MagicMock()
    store._upsert_stub = MagicMock()
    store._upsert_stub.StreamUpsert.side_effect = fake_stream_upsert

    nodes = [
        TextNode(id_="n1", text="a", embedding=[0.1, 0.2, 0.3]),
    ]
    ids = store.bulk_add(nodes)

    assert store._insert_stub.StreamInsert.call_count == 1
    assert store._upsert_stub.StreamUpsert.call_count == 0


def test_bulk_add_aggregates_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store(monkeypatch)

    def fake_stream_upsert(request_iter: Any) -> Any:
        requests = list(request_iter)
        # n1 succeeds, n2 fails (Status, no uuid info)
        yield _location_response(requests[0].vector.id)
        yield _status_response()

    store._upsert_stub = MagicMock()
    store._upsert_stub.StreamUpsert.side_effect = fake_stream_upsert

    nodes = [
        TextNode(id_="n1", text="a", embedding=[0.1, 0.2, 0.3]),
        TextNode(id_="n2", text="b", embedding=[0.4, 0.5, 0.6]),
    ]
    with pytest.raises(RuntimeError, match=r"1 entries failed out of 2"):
        _ = store.bulk_add(nodes)


def test_bulk_add_accepts_iterable(monkeypatch: pytest.MonkeyPatch) -> None:
    """bulk_add must accept a generator (not just a list)."""
    store = _make_store(monkeypatch)

    def fake_stream_upsert(request_iter: Any) -> Any:
        for req in request_iter:
            yield _location_response(req.vector.id)

    store._upsert_stub = MagicMock()
    store._upsert_stub.StreamUpsert.side_effect = fake_stream_upsert

    def node_gen() -> Any:
        for i in range(3):
            yield TextNode(id_=f"n{i}", text=str(i), embedding=[0.1, 0.2])

    ids = store.bulk_add(node_gen())
    assert sorted(ids) == ["n0", "n1", "n2"]


def test_add_rejects_node_without_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store(monkeypatch)

    store._upsert_stub = MagicMock()

    with pytest.raises(ValueError, match="embedding not set."):
        _ = store.add([TextNode(id_="n1", text="a")])


def test_bulk_add_rejects_node_without_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store(monkeypatch)

    def fake_stream_upsert(request_iter: Any) -> Any:
        for req in request_iter:
            yield _location_response(req.vector.id)

    store._upsert_stub = MagicMock()
    store._upsert_stub.StreamUpsert.side_effect = fake_stream_upsert

    with pytest.raises(ValueError, match="embedding not set."):
        _ = store.bulk_add([TextNode(id_="n1", text="a")])


def test_query_returns_ids_and_similarity(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store(monkeypatch)

    fake_response = SimpleNamespace(
        results=[
            SimpleNamespace(id="n1", distance=0.1),
            SimpleNamespace(id="n2", distance=0.2),
        ]
    )
    store._search_stub = MagicMock()
    store._search_stub.Search.return_value = fake_response

    result = store.query(
        VectorStoreQuery(query_embedding=[0.1, 0.2, 0.3], similarity_top_k=2)
    )

    print(result)
    assert result.ids == ["n1", "n2"]
    assert result.similarities == pytest.approx([0.1, 0.2])
    assert result.nodes is None

    # default radius/epsilon/timeout were used
    sent_cfg = store._search_stub.Search.call_args.args[0].config
    assert sent_cfg.radius == pytest.approx(-1.0)
    assert sent_cfg.epsilon == pytest.approx(0.01)
    assert sent_cfg.timeout == 3_000_000_000
    assert sent_cfg.num == 2


def test_query_per_call_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store(monkeypatch)

    store._search_stub = MagicMock()
    store._search_stub.Search.return_value = SimpleNamespace(results=[])

    result = store.query(
        VectorStoreQuery(query_embedding=[0.1, 0.2, 0.3], similarity_top_k=5),
        vald_search_config={"radius": 0.5, "epsilon": 0.2, "timeout_ns": 1_000_000_000},
    )

    sent_cfg = store._search_stub.Search.call_args.args[0].config
    assert sent_cfg.radius == pytest.approx(0.5)
    assert sent_cfg.epsilon == pytest.approx(0.2)
    assert sent_cfg.timeout == 1_000_000_000
    assert sent_cfg.num == 5

    # store-level defaults must remain untouched
    assert store.search_radius == pytest.approx(-1.0)
    assert store.search_epsilon == pytest.approx(0.01)
    assert store.search_timeout_ns == 3_000_000_000


def test_query_overrides_must_be_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store(monkeypatch)

    store._search_stub = MagicMock()

    with pytest.raises(TypeError, match="vald_search_config"):
        _ = store.query(
            VectorStoreQuery(query_embedding=[0.1, 0.2, 0.3]),
            vald_search_config="not-a-dict",
        )


def test_query_rejects_non_default_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store(monkeypatch)

    store._search_stub = MagicMock()

    bad_query = VectorStoreQuery(
        query_embedding=[0.1, 0.2, 0.3],
        mode=VectorStoreQueryMode.HYBRID,
    )

    with pytest.raises(ValueError, match="DEFAULT query mode"):
        _ = store.query(bad_query)


def test_query_requires_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store(monkeypatch)

    store._search_stub = MagicMock()

    with pytest.raises(ValueError, match="query_embedding"):
        _ = store.query(VectorStoreQuery(query_embedding=None))


def test_query_with_filters_warns_but_runs(
        monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    from llama_index.core.vector_stores.types import (
        FilterOperator,
        MetadataFilter,
        MetadataFilters,
    )

    store = _make_store(monkeypatch)

    store._search_stub = MagicMock()
    store._search_stub.Search.return_value = SimpleNamespace(results=[])

    filters = MetadataFilters(
        filters=[MetadataFilter(key="k", value="v", operator=FilterOperator.EQ)],
    )
    with caplog.at_level("WARNING"):
        store.query(
            VectorStoreQuery(query_embedding=[0.1, 0.2, 0.3], filters=filters),
        )
    assert any(
        "does not yet support VectorStoreQuery.filters" in r.message for r in caplog.records
    )


def test_delete_raises_not_impplemented(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store(monkeypatch)

    store._remove_stub = MagicMock()

    with pytest.raises(NotImplementedError, match="delete_nodes"):
        store.delete("ref-1")
    assert store._remove_stub.Remove.call_count == 0


def test_delete_nodes_iterates(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store(monkeypatch)

    store._remove_stub = MagicMock()

    store.delete_nodes(node_ids=["a", "b", "c"])
    assert store._remove_stub.Remove.call_count == 3


def test_delete_nodes_rejects_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store(monkeypatch)

    store._remove_stub = MagicMock()

    with pytest.raises(NotImplementedError):
        store.delete_nodes(node_ids=["a"], filters="anything-truthy")


def test_clear_calls_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _make_store(monkeypatch)

    store._flush_stub = MagicMock()

    store.clear()
    assert store._flush_stub.Flush.call_count == 1


def test_caller_owned_channel_not_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    import grpc

    # ensure no channel construction happens when caller passes one in
    monkeypatch.setattr(
        grpc,
        "insecure_channel",
        lambda *a, **kw: pytest.fail("should not construct channel"),
    )
    user_channel = MagicMock(name="user-owned")
    store = ValdVectorStore(channel=user_channel)
    assert store.client is user_channel

    store.close()
    user_channel.close.assert_not_called()


def test_self_owned_channel_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    import grpc

    fake_channel = MagicMock(name="self-owned")
    monkeypatch.setattr(grpc, "insecure_channel", lambda *a, **kw: fake_channel)

    store = ValdVectorStore()
    store.close()
    fake_channel.close.assert_called_once()
