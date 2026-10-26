"""Stateful report sync coverage; no real model, WebDAV or Zotero calls."""

import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import threading
import subprocess

import pytest

from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily import executor as executor_module, reports
from zotero_arxiv_daily.construct_email import render_email
from zotero_arxiv_daily.executor import Executor


class Library:
    def __init__(self):
        self.store = {}
        self.lose_response = False

    def everything(self, items):
        return copy.deepcopy(items)

    def collections(self):
        return [{"key": "DAILY", "data": {"name": "DailyArxiv", "parentCollection": False}}]

    def collection_items_top(self, key):
        return [item for item in self.store.values() if key in item["data"].get("collections", [])]

    def items(self, **kwargs):
        return [item for item in self.store.values() if item["data"]["itemType"] == "preprint"]

    def children(self, key):
        return [item for item in self.store.values() if item["data"].get("parentItem") == key]

    def item(self, key):
        return copy.deepcopy(self.store[key])

    def item_template(self, item_type, linkmode=None):
        return {"itemType": item_type, "tags": [], "collections": []}

    def create_items(self, items):
        successful, unchanged = {}, {}
        for index, data in enumerate(items):
            assert data["version"] == 0  # Zotero rejects key-based creates without this.
            key = data["key"]
            if key in self.store:
                unchanged[str(index)] = key
                continue
            item = {"key": key, "version": 1, "data": copy.deepcopy(data)}
            self.store[key] = item
            successful[str(index)] = item
            if data["itemType"] == "attachment" and self.lose_response:
                raise TimeoutError("Saved, but response lost")
        return {"successful": successful, "unchanged": unchanged, "failed": {}}

    def addto_collection(self, key, item):
        self.store[item["key"]]["data"]["collections"].append(key)
        return SimpleNamespace(raise_for_status=lambda: None)


class WebDAV:
    def __init__(self):
        self.files = {}
        self.fail = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def put(self, url, data, **kwargs):
        if self.fail:
            raise RuntimeError("WebDAV unavailable")
        self.files[url] = data
        return SimpleNamespace(raise_for_status=lambda: None)


@pytest.fixture
def sync(config, monkeypatch, tmp_path):
    config.zotero.save_collection = "DailyArxiv"
    config.webdav = {"url": "https://dav.invalid/user/", "username": "user", "password": "fake"}
    library, dav, generated = Library(), WebDAV(), []
    monkeypatch.setattr("zotero_arxiv_daily.executor.zotero.Zotero", lambda *a, **kw: library)
    monkeypatch.setattr(reports.requests, "Session", lambda: dav)
    def generate(paper, llm, output_dir):
        generated.append(paper.url)
        path = tmp_path / output_dir.name / "report.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<html><body>中文图文解读</body></html>")
        return path
    monkeypatch.setattr("zotero_arxiv_daily.executor.generate_report", generate)
    executor = Executor.__new__(Executor)
    executor.config = config
    return executor, library, dav, generated


def papers(count):
    return [make_sample_paper(url=f"https://arxiv.org/abs/2610.{i:05d}v1") for i in range(count)]


def attachments(library):
    return [item for item in library.store.values() if item["data"]["itemType"] == "attachment"]


def test_top_ten_reports_and_repeat_deduplication(sync):
    executor, library, dav, generated = sync
    candidates = papers(15)
    email = render_email(candidates)
    executor.save_to_zotero(candidates)
    executor.save_to_zotero(candidates)
    assert len(generated) == len(attachments(library)) == 10
    assert len(library.store) == len(dav.files) == 20
    assert render_email(candidates) == email
    attachment = attachments(library)[0]["data"]
    assert attachment["contentType"] == "text/html"
    assert attachment["tags"] == [{"tag": reports.REPORT_TAG}]
    archive = dav.files[f'https://dav.invalid/user/zotero/{attachment["key"]}.zip']
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        assert z.namelist() == [reports.REPORT_FILENAME]
        assert "中文图文解读" in z.read(reports.REPORT_FILENAME).decode()


def test_existing_parent_and_legacy_note_are_preserved(sync):
    executor, library, dav, generated = sync
    original = {"key": "EXISTING", "version": 1, "data": {
        "itemType": "preprint", "url": "https://arxiv.org/pdf/2610.00000v1.pdf",
        "collections": ["DAILY"], "title": "User title", "extra": "User metadata",
    }}
    note = {"key": "OLDNOTE", "data": {"itemType": "note", "parentItem": "EXISTING",
            "note": "Old analysis", "tags": [{"tag": "zotero-arxiv-daily:chinese-analysis:v1"}]}}
    library.store = {"EXISTING": copy.deepcopy(original), "OLDNOTE": copy.deepcopy(note)}
    executor.save_to_zotero([make_sample_paper(url="https://arxiv.org/abs/2610.00000v2")])
    assert library.store["EXISTING"] == original
    assert library.store["OLDNOTE"] == note
    assert attachments(library)[0]["data"]["parentItem"] == "EXISTING"


def test_failed_upload_does_not_publish_attachment_and_retries(sync):
    executor, library, dav, generated = sync
    dav.fail = True
    executor.save_to_zotero(papers(1))
    assert len(library.store) == 1 and not attachments(library)
    dav.fail = False
    executor.save_to_zotero(papers(1))
    assert len(library.store) == 2 and len(attachments(library)) == 1
    assert len(generated) == 2


def test_lost_zotero_response_and_arxiv_versions_do_not_duplicate(sync):
    executor, library, dav, generated = sync
    library.lose_response = True
    candidate = papers(1)[0]
    executor.save_to_zotero([candidate])
    candidate.url = candidate.url.replace("v1", "v2")
    executor.save_to_zotero([candidate])
    assert len(library.store) == 2 and len(attachments(library)) == len(generated) == 1


def test_missing_webdav_still_syncs_parent(sync):
    executor, library, dav, generated = sync
    executor.config.webdav.password = None
    executor.save_to_zotero(papers(1))
    assert len(library.store) == 1 and not generated and not dav.files


def test_email_is_sent_before_report_sync(sync, monkeypatch):
    executor, library, dav, generated = sync
    candidates, events = papers(1), []
    executor.fetch_zotero_corpus = lambda: [object()]
    executor.filter_corpus = lambda corpus: corpus
    executor.retrievers = {"arxiv": SimpleNamespace(retrieve_papers=lambda: candidates)}
    executor.reranker = SimpleNamespace(rerank=lambda papers, corpus: papers)
    executor.openai_client = object()
    monkeypatch.setattr(type(candidates[0]), "generate_tldr", lambda *args: None)
    monkeypatch.setattr(type(candidates[0]), "generate_affiliations", lambda *args: None)
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", lambda *args: events.append("email"))
    executor.save_to_zotero = lambda *args: events.append("sync")
    executor.run()
    assert events == ["email", "sync"]


@pytest.mark.parametrize("mode,provider", [("chat_completion", "daily"), ("response", "openai")])
def test_agent_reuses_model_and_does_not_receive_other_credentials(config, monkeypatch, tmp_path, mode, provider):
    config.llm.api_mode = mode
    monkeypatch.setenv("ZOTERO_KEY", "private-zotero-key")
    monkeypatch.setenv("WEBDAV_PASSWORD", "private-webdav-password")
    def run(command, **kwargs):
        env = kwargs["env"]
        assert "ZOTERO_KEY" not in env and "WEBDAV_PASSWORD" not in env
        assert env["REPORT_API_KEY"] == config.llm.api.key
        settings = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        assert settings["model"] == f"{provider}/{config.llm.generation_kwargs.model}"
        options = settings["provider"][provider]["options"]
        assert options["baseURL"] == config.llm.api.base_url
        assert "timeout" not in options
        assert "/Users/" not in command[-1] and "无头浏览器" not in command[-1]
        assert "https://arxiv.org/html/2610.00000v1" in command[-1]
        Path(kwargs["cwd"], "report.html").write_text("<html>中文解读</html>")
    monkeypatch.setattr(reports.subprocess, "run", run)
    report = reports.generate_report(papers(1)[0], config.llm, tmp_path)
    assert report.read_text() == "<html>中文解读</html>"


def test_five_generators_overlap_and_writes_stay_on_caller(sync, monkeypatch, tmp_path):
    executor, library, dav, generated = sync
    caller = threading.get_ident()
    barrier = threading.Barrier(5, timeout=10)
    lock = threading.Lock()
    active = peak = 0
    original_create, original_put = library.create_items, dav.put
    original_children = library.children

    def create(items):
        assert threading.get_ident() == caller
        return original_create(items)

    def children(key):
        assert threading.get_ident() == caller
        return original_children(key)

    def put(*args, **kwargs):
        assert threading.get_ident() == caller
        return original_put(*args, **kwargs)

    def generate(paper, llm, output_dir):
        nonlocal active, peak
        assert threading.get_ident() != caller
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait()
        path = tmp_path / output_dir.name
        path.write_text("中文解读")
        # Hold each wave together so peak measures concurrent generations.
        barrier.wait()
        with lock:
            active -= 1
        return path

    monkeypatch.setattr(library, "create_items", create)
    monkeypatch.setattr(library, "children", children)
    monkeypatch.setattr(dav, "put", put)
    monkeypatch.setattr("zotero_arxiv_daily.executor.generate_report", generate)
    executor.save_to_zotero(papers(10))
    assert peak == 5
    assert len(attachments(library)) == 10


def test_generation_failure_does_not_stop_other_reports(sync, monkeypatch):
    executor, library, dav, generated = sync
    original = executor_module.generate_report

    def generate(paper, *args):
        if paper.url == papers(1)[0].url:
            raise RuntimeError("Model unavailable")
        return original(paper, *args)

    monkeypatch.setattr("zotero_arxiv_daily.executor.generate_report", generate)
    executor.save_to_zotero(papers(10))
    assert len(attachments(library)) == 9
    assert len(library.items()) == 10


def test_pending_arxiv_versions_generate_only_once(sync):
    executor, library, dav, generated = sync
    candidate = papers(1)[0]
    revision = copy.deepcopy(candidate)
    revision.url = candidate.url.replace("v1", "v2")
    executor.save_to_zotero([candidate, revision])
    assert len(generated) == len(attachments(library)) == 1


def test_attachment_write_survives_unavailable_template_endpoint(sync, monkeypatch):
    executor, library, dav, generated = sync
    original = library.item_template

    def template(item_type, *args):
        if item_type == "attachment":
            raise RuntimeError("Expired template cache: missing itemType query parameter")
        return original(item_type, *args)

    monkeypatch.setattr(library, "item_template", template)
    executor.save_to_zotero(papers(2))
    assert len(attachments(library)) == 2
    assert all(item["data"]["linkMode"] == "imported_file" for item in attachments(library))


@pytest.mark.parametrize("retry_after,expected", [(None, 120), ("300", 300), ("bad", 120),
    ("Thu, 01 Jan 1970 00:05:00 GMT", 300)])
def test_rate_limit_waits_and_resumes_same_session(config, monkeypatch, tmp_path, retry_after, expected):
    calls, waits = [], []
    monkeypatch.setattr(reports.time, "sleep", waits.append)
    monkeypatch.setattr(reports.time, "time", lambda: 0)
    monkeypatch.setattr(reports.random, "uniform", lambda *args: 0)

    def run(command, **kwargs):
        calls.append((command, kwargs["cwd"], kwargs["env"]))
        if len(calls) == 1:
            Path(kwargs["cwd"], "paper.txt").write_text("Already read")
            event = {"type": "error", "sessionID": "ses_existing", "error": {"data": {
                "statusCode": 429, "responseHeaders": {"retry-after": retry_after}}}}
            kwargs["stdout"].write("plain diagnostic\n" + json.dumps(event) + "\n")
            raise subprocess.CalledProcessError(1, command)
        assert command[-3:-1] == ["--session", "ses_existing"]
        assert calls[0][1:] == calls[1][1:]
        assert Path(kwargs["cwd"], "paper.txt").read_text() == "Already read"
        Path(kwargs["cwd"], "report.html").write_text("Recovered report")
        kwargs["stdout"].write(json.dumps({"type": "text", "sessionID": "ses_existing"}) + "\n")

    monkeypatch.setattr(reports.subprocess, "run", run)
    assert reports.generate_report(papers(1)[0], config.llm, tmp_path).read_text() == "Recovered report"
    assert waits == [expected]
    assert '"statusCode": 429' in (tmp_path / "agent.jsonl").read_text()
    assert '"type": "text"' in (tmp_path / "agent.jsonl").read_text()


@pytest.mark.parametrize("status,session,attempts", [(429, "ses_existing", 4),
    (401, "ses_existing", 1), (500, "ses_existing", 1), (429, None, 1)])
def test_report_retry_is_bounded_and_only_for_resumable_429(config, monkeypatch, tmp_path, status, session, attempts):
    calls, waits = [], []
    monkeypatch.setattr(reports.time, "sleep", waits.append)
    monkeypatch.setattr(reports.random, "uniform", lambda *args: 0)

    def run(command, **kwargs):
        calls.append(command)
        kwargs["stdout"].write(json.dumps({"type": "error", "sessionID": session,
            "error": {"data": {"statusCode": status}}}) + "\n")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(reports.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        reports.generate_report(papers(1)[0], config.llm, tmp_path)
    assert len(calls) == attempts
    assert waits == [120, 240, 480][:attempts - 1]
    assert not (tmp_path / reports.REPORT_FILENAME).exists()


def test_prior_429_does_not_retry_a_later_permanent_error(config, monkeypatch, tmp_path):
    calls, waits = [], []
    monkeypatch.setattr(reports.time, "sleep", waits.append)
    monkeypatch.setattr(reports.random, "uniform", lambda *args: 0)

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            kwargs["stdout"].write(json.dumps({"type": "error", "sessionID": "ses_existing",
                "error": {"data": {"statusCode": 429}}}) + "\n")
        else:
            kwargs["stdout"].write("Unstructured permanent failure\n")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(reports.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        reports.generate_report(papers(1)[0], config.llm, tmp_path)
    assert len(calls) == 2 and waits == [120]
