"""Stateful sync tests: backfill, retry, duplicates and unchanged email."""

import copy
from types import SimpleNamespace

import pytest
from omegaconf import open_dict

from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily.analysis import NOTE_TAG
from zotero_arxiv_daily.construct_email import render_email
from zotero_arxiv_daily.executor import Executor


class Library:
    def __init__(self):
        self.store = {}
        self.writes = []
        self.fail_note = False
        self.ambiguous_note = False

    def everything(self, items):
        return copy.deepcopy(items)

    def collections(self):
        return [{"key": "DAILY", "data": {"name": "DailyArxiv", "parentCollection": False}}]

    def collection_items_top(self, key):
        return [i for i in self.store.values() if key in i["data"].get("collections", [])]

    def items(self, **kwargs):
        return [i for i in self.store.values() if i["data"]["itemType"] != "note"]

    def children(self, key):
        return [i for i in self.store.values() if i["data"].get("parentItem") == key]

    def item(self, key):
        return copy.deepcopy(self.store[key])

    def item_template(self, item_type):
        return {"itemType": item_type, "tags": [], "collections": []}

    def addto_collection(self, key, item):
        self.store[item["key"]]["data"]["collections"].append(key)
        return SimpleNamespace(raise_for_status=lambda: None)

    def create_items(self, items):
        successful, unchanged, failed = {}, {}, {}
        for n, item in enumerate(items):
            if 'key' in item and 'version' not in item:
                failed[str(n)] = {"code": 428, "message": "Version required for key-based writes"}
                continue
            if item["itemType"] == "note" and self.fail_note:
                failed[str(n)] = {"code": 403, "message": "Write denied"}
                continue
            key = item["key"]
            if key in self.store:
                unchanged[str(n)] = key
                continue
            stored = {"key": key, "version": 1, "data": copy.deepcopy(item)}
            self.store[key] = stored
            self.writes.append(copy.deepcopy(item))
            successful[str(n)] = stored
            if item["itemType"] == "note" and self.ambiguous_note:
                raise TimeoutError("POST saved, response lost")
        return {"successful": successful, "unchanged": unchanged, "failed": failed}

    def parent(self, url="https://arxiv.org/abs/2610.08268v1", collection="DAILY"):
        item = {"key": "EXISTING", "version": 1, "data": {
            "itemType": "preprint", "url": url, "collections": [collection],
            "title": "User's title", "extra": "User's metadata", "tags": [],
        }}
        self.store[item["key"]] = item
        return item


@pytest.fixture
def sync(config, monkeypatch):
    with open_dict(config.zotero):
        config.zotero.save_collection = "DailyArxiv"
    library = Library()
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: library)
    executor = Executor.__new__(Executor)
    executor.config = config
    executor.openai_client = object()
    analysed = []
    def generate(client, paper, params):
        analysed.append(paper.url)
        return "<h1>中文详细解读</h1><p>带证据的解读</p>"
    monkeypatch.setattr("zotero_arxiv_daily.executor.generate_analysis", generate)
    return executor, library, analysed


def papers(count):
    return [make_sample_paper(url=f"https://arxiv.org/abs/2610.{i:05d}v1", score=10 - i / 100) for i in range(count)]


def notes(library):
    return [i for i in library.store.values() if i["data"]["itemType"] == "note"]


def test_only_same_top_ten_are_synced_and_analysed(sync):
    executor, library, analysed = sync
    candidates = papers(15)
    before = render_email(candidates)
    executor.save_to_zotero(candidates)
    assert analysed == [p.url for p in candidates[:10]]
    assert len(notes(library)) == 10
    assert len(library.store) == 20
    assert render_email(candidates) == before
    for note in notes(library):
        assert note["data"]["parentItem"] in library.store
        assert note["data"]["tags"] == [{"tag": NOTE_TAG}]


def test_repeat_run_does_not_add_parents_or_notes_or_generate_again(sync):
    executor, library, analysed = sync
    candidates = papers(3)
    executor.save_to_zotero(candidates)
    executor.save_to_zotero(candidates)
    assert len(library.store) == 6 and len(analysed) == 3


def test_existing_parent_receives_note_without_overwriting_metadata(sync):
    executor, library, analysed = sync
    original = copy.deepcopy(library.parent())
    executor.save_to_zotero([make_sample_paper(url=original["data"]["url"])])
    assert library.store["EXISTING"] == original
    assert len(library.store) == 2
    assert notes(library)[0]["data"]["parentItem"] == "EXISTING"
    assert len(analysed) == 1


def test_library_parent_outside_daily_collection_is_reused(sync):
    executor, library, analysed = sync
    library.parent(url="http://arxiv.org/pdf/2610.08268v1.pdf", collection="USERCOLL")
    executor.save_to_zotero([make_sample_paper(url="https://arxiv.org/abs/2610.08268v2")])
    assert len(library.store) == 2
    assert library.store["EXISTING"]["data"]["collections"] == ["USERCOLL", "DAILY"]
    assert len(analysed) == 1


def test_user_notes_are_preserved_and_do_not_block_backfill(sync):
    executor, library, analysed = sync
    parent = library.parent()
    library.store["USERNOTE"] = {"key": "USERNOTE", "data": {
        "itemType": "note", "parentItem": parent["key"], "note": "My notes", "tags": [],
    }}
    executor.save_to_zotero([make_sample_paper(url=parent["data"]["url"])])
    assert len(notes(library)) == 2
    assert library.store["USERNOTE"]["data"]["note"] == "My notes"


def test_generation_failure_isolated_and_retried(sync, monkeypatch):
    executor, library, analysed = sync
    candidates = papers(2)
    def generate(client, paper, params):
        if paper.url == candidates[0].url:
            raise RuntimeError("LLM unavailable")
        return "<p>详细解读</p>"
    monkeypatch.setattr("zotero_arxiv_daily.executor.generate_analysis", generate)
    executor.save_to_zotero(candidates)
    assert len(library.store) == 3 and len(notes(library)) == 1
    monkeypatch.setattr("zotero_arxiv_daily.executor.generate_analysis", lambda *a: "<p>补齐解读</p>")
    executor.save_to_zotero(candidates)
    assert len(library.store) == 4 and len(notes(library)) == 2


def test_failed_note_write_is_not_marked_complete(sync):
    executor, library, analysed = sync
    library.fail_note = True
    executor.save_to_zotero(papers(1))
    assert len(library.store) == 1 and not notes(library)
    library.fail_note = False
    executor.save_to_zotero(papers(1))
    assert len(library.store) == 2 and len(analysed) == 2


def test_lost_write_response_is_safe_to_retry(sync):
    executor, library, analysed = sync
    library.ambiguous_note = True
    executor.save_to_zotero(papers(1))
    executor.save_to_zotero(papers(1))
    assert len(library.store) == 2 and len(analysed) == 1


def test_duplicate_candidates_reuse_parent(sync):
    executor, library, analysed = sync
    executor.save_to_zotero([
        make_sample_paper(url="https://arxiv.org/abs/2610.08268v1"),
        make_sample_paper(url="https://arxiv.org/abs/2610.08268v2"),
    ])
    assert len(library.store) == 2 and len(analysed) == 1


@pytest.mark.parametrize("disabled", ["collection", "count"])
def test_disabled_settings_do_not_generate(sync, disabled):
    executor, library, analysed = sync
    if disabled == "collection":
        executor.config.zotero.save_collection = None
    elif disabled == "count":
        with open_dict(executor.config.zotero):
            executor.config.zotero.save_top_k = 0
    executor.save_to_zotero(papers(1))
    assert not analysed and not notes(library)
    assert not library.store


def test_concurrent_note_created_during_generation_is_not_duplicated(sync, monkeypatch):
    executor, library, analysed = sync
    parent = library.parent()
    def generate(*args):
        library.store["OTHERNOTE"] = {"key": "OTHERNOTE", "data": {
            "itemType": "note", "parentItem": parent["key"], "note": "Other run's analysis",
            "tags": [{"tag": NOTE_TAG}],
        }}
        return "<p>Our analysis</p>"
    monkeypatch.setattr("zotero_arxiv_daily.executor.generate_analysis", generate)
    executor.save_to_zotero([make_sample_paper(url=parent["data"]["url"])])
    assert len(notes(library)) == 1


def test_email_is_sent_before_sync(config, monkeypatch):
    candidates = papers(2)
    events = []
    executor = Executor.__new__(Executor)
    executor.config = config
    executor.openai_client = object()
    executor.fetch_zotero_corpus = lambda: [object()]
    executor.filter_corpus = lambda corpus: corpus
    executor.retrievers = {"arxiv": SimpleNamespace(retrieve_papers=lambda: candidates)}
    executor.reranker = SimpleNamespace(rerank=lambda papers, corpus: papers)
    monkeypatch.setattr(type(candidates[0]), "generate_tldr", lambda *a: None)
    monkeypatch.setattr(type(candidates[0]), "generate_affiliations", lambda *a: None)
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", lambda cfg, content: events.append(("email", content)))
    executor.save_to_zotero = lambda papers: events.append(("sync", papers))
    executor.run()
    assert [event[0] for event in events] == ["email", "sync"]
    assert events[0][1] == render_email(candidates)


def test_real_clients_with_mock_http_create_child_and_skip_on_second_run(config, monkeypatch):
    """Exercise real Pyzotero/OpenAI serialization against in-memory HTTP servers."""
    import json
    import httpx
    from openai import OpenAI
    from pyzotero.zotero import Zotero
    from zotero_arxiv_daily.analysis import SECTIONS

    stored, writes, llm_calls = {}, [], []

    def zotero_server(request):
        path = request.url.path
        if request.method == "GET":
            if path.endswith("/collections"):
                data = [{"key": "DAILY", "data": {"name": "DailyArxiv", "parentCollection": False}}]
            elif path == "/items/new":
                data = {"itemType": request.url.params["itemType"], "tags": [], "collections": []}
            elif path.endswith("/children"):
                parent = path.split("/")[-2]
                data = [i for i in stored.values() if i["data"].get("parentItem") == parent]
            elif path.rsplit("/", 1)[-1] in stored:
                data = stored[path.rsplit("/", 1)[-1]]
            else:
                data = [i for i in stored.values() if i["data"]["itemType"] == "preprint"]
            return httpx.Response(200, json=data)
        assert request.method == "POST" and path.endswith("/items")
        assert request.headers.get("Zotero-Write-Token")
        successful = {}
        for n, data in enumerate(json.loads(request.content)):
            if 'key' in data and 'version' not in data:
                return httpx.Response(200, json={"successful": {}, "unchanged": {}, "failed": {
                    str(n): {"code": 428, "message": "Version required for key-based writes"},
                }})
            assert data['version'] == 0, "New objects must declare version zero"
            assert data["key"] not in stored, "Duplicate HTTP write"
            key = data["key"]
            item = {"key": key, "version": 1, "data": data}
            stored[key] = item
            writes.append(data)
            successful[str(n)] = item
        return httpx.Response(200, json={"successful": successful, "unchanged": {}, "failed": {}})

    def llm_server(request):
        payload = json.loads(request.content)
        llm_calls.append(payload)
        data = {
            "sections": [{"heading": h, "body": "中文研究解读 [S1]。"} for h in SECTIONS],
            "evidence": [{"block": 1, "quote": "Throughput was 100 tokens/s.", "claim": "吞吐量结果"}],
        }
        return httpx.Response(200, json={
            "id": "chatcmpl-test", "created": 1, "model": payload["model"], "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(data, ensure_ascii=False)}}],
        })

    with open_dict(config.zotero):
        config.zotero.save_collection = "DailyArxiv"
    zot = Zotero("123", "user", "fake-key")
    zot.client.close()
    zot.client = httpx.Client(transport=httpx.MockTransport(zotero_server))
    llm = OpenAI(api_key="fake-key", base_url="https://llm.invalid/v1", http_client=httpx.Client(transport=httpx.MockTransport(llm_server)))
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: zot)
    executor = Executor.__new__(Executor)
    executor.config, executor.openai_client = config, llm
    paper = make_sample_paper(url="https://arxiv.org/abs/2610.08268v1", full_text="Throughput was 100 tokens/s.")
    try:
        executor.save_to_zotero([paper])
        executor.save_to_zotero([paper])
    finally:
        llm.close()
        zot.client.close()
    assert len(writes) == 2 and len(llm_calls) == 1
    assert all(data['version'] == 0 for data in writes)
    assert writes[1]["itemType"] == "note"
    assert writes[1]["parentItem"] == writes[0]["key"]
    assert "中文详细解读" in writes[1]["note"]
