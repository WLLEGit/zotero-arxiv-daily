"""Local report generation, concurrency and retry boundaries; no real model calls."""
import copy
import json
from pathlib import Path
import threading
import subprocess
import pytest
from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily import reports


def papers(count):
    return [make_sample_paper(url=f"https://arxiv.org/abs/2610.{i:05d}v1") for i in range(count)]


@pytest.fixture
def generated(config, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls = []
    def generate(paper, llm, directory):
        calls.append(paper.url)
        path = directory / reports.REPORT_FILENAME
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text('<html>中文图文解读</html>')
        return path
    monkeypatch.setattr(reports, 'generate_report', generate)
    return calls


def test_top_ten_local_reports_and_repeat_deduplication(config, generated):
    candidates = papers(15)
    reports.generate_reports(candidates, config.llm)
    reports.generate_reports(candidates, config.llm)
    rows = json.loads(Path('outputs/report-results.json').read_text())
    assert len(rows) == len(generated) == 10
    assert all(r['status'] == 'existing' and Path(r['report_path']).is_file() for r in rows)
    assert not any('parent_key' in r or 'attachment_key' in r for r in rows)
    assert not Path('outputs/zotero-results.json').exists()


def test_failure_does_not_stop_other_reports_and_retries(config, generated, monkeypatch):
    original = reports.generate_report
    def fail(paper, *args):
        if paper.url == papers(1)[0].url: raise RuntimeError('unavailable')
        return original(paper, *args)
    monkeypatch.setattr(reports, 'generate_report', fail)
    reports.generate_reports(papers(10), config.llm)
    rows = json.loads(Path('outputs/report-results.json').read_text())
    assert sum(r['status'] == 'failed' for r in rows) == 1
    monkeypatch.setattr(reports, 'generate_report', original)
    reports.generate_reports(papers(10), config.llm)
    assert len(generated) == 10
    assert all(r['status'] != 'failed' for r in json.loads(Path('outputs/report-results.json').read_text()))


def test_arxiv_versions_generate_only_once(config, generated):
    paper = papers(1)[0]
    revision = copy.deepcopy(paper);revision.url = paper.url.replace('v1', 'v2')
    reports.generate_reports([paper, revision], config.llm)
    assert len(generated) == 1


def test_five_local_reports_overlap(config, generated, monkeypatch):
    original = reports.generate_report
    barrier = threading.Barrier(5, timeout=10)
    def generate(*args):
        barrier.wait()
        path = original(*args)
        barrier.wait()
        return path
    monkeypatch.setattr(reports, 'generate_report', generate)
    reports.generate_reports(papers(10), config.llm)
    assert len(generated) == 10


def test_local_reports_do_not_require_zotero_or_webdav(config, generated):
    config.zotero.save_collection = None
    reports.generate_reports(papers(1), config.llm)
    assert len(generated) == 1


def test_pipeline_reports_continue_after_metadata_failure(config, generated, monkeypatch):
    from types import SimpleNamespace
    from zotero_arxiv_daily import executor as module, zotero_sync
    executor = module.Executor.__new__(module.Executor)
    executor.config = config
    executor.openai_client = object()
    candidate = papers(1)[0]
    executor.fetch_zotero_corpus = lambda: [object()]
    executor.filter_corpus = lambda rows: rows
    executor.retrievers = {'arxiv': SimpleNamespace(retrieve_papers=lambda: [candidate])}
    executor.reranker = SimpleNamespace(rerank=lambda rows, corpus: rows)
    config.zotero.save_collection = 'DailyArxiv'
    monkeypatch.setattr(module, 'enrich_papers', lambda *args: None)
    def unavailable(*args): raise RuntimeError('Zotero unavailable')
    monkeypatch.setattr(zotero_sync.zotero, 'Zotero', unavailable)
    sent = []
    monkeypatch.setattr(module, 'send_email', lambda *args: sent.append(True))
    executor.run()
    assert sent == [True] and len(generated) == 1
    assert json.loads(Path('outputs/zotero-results.json').read_text())[0]['status'] == 'failed'
    assert json.loads(Path('outputs/report-results.json').read_text())[0]['status'] == 'saved'


@pytest.mark.parametrize("mode,provider", [("chat_completion", "daily"), ("response", "openai")])
def test_agent_reuses_model_and_does_not_receive_other_credentials(config, monkeypatch, tmp_path, mode, provider):
    config.llm.api_mode = mode
    monkeypatch.setenv("ZOTERO_KEY", "private-zotero-key")
    monkeypatch.setenv("SENDER_PASSWORD", "private-smtp-password")
    def run(command, **kwargs):
        env = kwargs["env"]
        assert "ZOTERO_KEY" not in env and "SENDER_PASSWORD" not in env
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
