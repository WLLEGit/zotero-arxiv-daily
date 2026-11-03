"""Generate local illustrated reports with OpenCode for the self-hosted reader."""

import hashlib
import json
import os
import random
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from .outputs import write_results
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from loguru import logger

REPORT_FILENAME = "daily-arxiv-report.html"


def _complete_report(path):
    if not path.is_file():
        return False
    text = path.read_text(errors='replace')
    return ('</html>' in text.lower() and 'data:image/' in text
            and len(re.findall('[\u4e00-\u9fff]', text)) >= 200)


def canonical_identifier(value: str) -> str:
    """Match arXiv abs/PDF/HTML URLs and versions to the same paper."""
    match = re.search(
        r"(?:arxiv\.org/(?:abs|pdf|html)/|arxiv:)"
        r"(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?",
        value.strip(), re.IGNORECASE,
    )
    return "arxiv:" + match.group(1).lower() if match else value.strip().rstrip("/")


def item_identifiers(item: dict) -> list[str]:
    data = item.get("data", item)
    return [canonical_identifier(data[field]) for field in ("archiveID", "url") if data.get(field)]


def stable_item_key(identity: str) -> str:
    alphabet = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    value = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:5], "big")
    return "".join(alphabet[(value >> (5 * i)) & 31] for i in range(8))


def _run_agent(command: list[str], work: str, env: dict, log_path: Path) -> None:
    """Resume transient failures in place; retain each attempt and completed files."""
    base_command = command[:-1]
    with log_path.open("a+", encoding="utf-8") as log:
        log_path.chmod(0o600)
        for attempt in range(4):
            log.seek(0, os.SEEK_END)
            start = log.tell()
            failure = None
            try:
                subprocess.run(command, cwd=work, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
                if not (Path(work) / 'report.html').is_file() or not (Path(work) / 'report.html').stat().st_size:
                    raise FileNotFoundError('Agent exited without report.html')
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                failure = exc
            finally:
                log.seek(start)
                segment = log.read()
                history = log_path.with_name(f'agent-{time.time_ns()}-attempt-{attempt + 1}.jsonl')
                history.write_text(segment); history.chmod(0o600)
                log.seek(0, os.SEEK_END)
            if failure is None:
                return
            if _complete_report(Path(work) / 'report.html'):
                logger.warning('Agent ended with an error after completing report.html; retaining the report')
                return
            error = None
            session = None
            for line in segment.splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                session = event.get('sessionID') or session
                if event.get("type") == "error":
                    error = event
            data = (error or {}).get("error", {}).get("data", {})
            status = data.get('statusCode')
            message = str(data.get('message', '')).lower()
            transient = (status in {429, 500, 502, 503, 504}
                         or isinstance(failure, FileNotFoundError)
                         or any(s in message for s in ('upstream_stream_read_error',
                             'upstream_http2_stream_error', 'socket connection', 'cannot connect to api')))
            if not transient or not session or attempt == 3 or status in {400, 401, 403, 404}:
                raise failure
            delay = (15, 45, 90)[attempt]
            if status == 429:
                delay = 120 * 2 ** attempt + random.uniform(0, 30)
                headers = data.get("responseHeaders") or {}
                for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1)):
                    value = headers.get(name)
                    if value is None:
                        continue
                    try:
                        wait = float(value) * scale
                    except (TypeError, ValueError):
                        try:
                            wait = parsedate_to_datetime(value).timestamp() - time.time()
                        except (TypeError, ValueError, OverflowError):
                            continue
                    delay = max(delay, wait)
            logger.warning(f"Report interrupted; waiting {delay:.0f}s before session retry {attempt + 1}/3")
            time.sleep(delay)
            command = [*base_command, "--session", session,
                       "继续当前论文的图文解读，复用已有阅读结果和文件，完成 report.html 写入和图片内嵌后直接交付。"]


def generate_report(paper, llm_params, output_dir: Path) -> Path:
    """Reuse the configured model and resume sessions after sustained rate limits."""
    model = str(llm_params.generation_kwargs.model)
    responses = llm_params.get("api_mode", "chat_completion") == "response"
    provider = "openai" if responses else "daily"
    selected_model = f"{provider}/{model}"
    config = {
        "model": selected_model,
        "small_model": selected_model,
        "enabled_providers": [provider],
        "provider": {provider: {
            "npm": "@ai-sdk/openai" if responses else "@ai-sdk/openai-compatible",
            "options": {"baseURL": str(llm_params.api.base_url), "apiKey": "{env:REPORT_API_KEY}"},
            "models": {model: {"name": model, "modalities": {"input": ["text", "image"], "output": ["text"]}}},
        }},
        "permission": {"task": "deny", "question": "deny", "external_directory": "deny"},
    }
    url = paper.url.replace("/abs/", "/html/") if paper.source == "arxiv" else paper.url
    prompt = Path(__file__).with_name("report_prompt.txt").read_text(encoding="utf-8")
    prompt = prompt.replace("{{paper_url}}", url)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Do not pass SMTP, Zotero or GitHub credentials to the paper agent.
    inherited = {"PATH", "LANG", "LC_ALL", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "SSL_CERT_FILE", "NODE_EXTRA_CA_CERTS"}
    # outputs lives on the data disk in self-hosted runs, not the shared 1 GB /tmp.
    with TemporaryDirectory(prefix=".work-", dir=output_dir.resolve()) as work:
        home = Path(work) / ".home"
        home.mkdir()
        env = {key: value for key, value in os.environ.items() if key in inherited}
        env.update({
            "PATH": str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', os.defpath),
            "TMPDIR": work,
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_DATA_HOME": str(home / ".local/share"),
            "XDG_STATE_HOME": str(home / ".local/state"),
            "REPORT_API_KEY": str(llm_params.api.key),
            "OPENCODE_CONFIG_CONTENT": json.dumps(config),
            "OPENCODE_DISABLE_AUTOUPDATE": "1",
        })
        try:
            _run_agent(
                ["opencode", "run", "--model", selected_model, "--format", "json", "--dir", work, prompt],
                work, env, output_dir / "agent.jsonl",
            )
        finally:
            diagnostics = output_dir / 'diagnostics'
            for path in sorted(home.glob('.local/**/log/*.log'))[-10:]:
                diagnostics.mkdir(exist_ok=True, mode=0o700)
                text = path.read_text(errors='replace')[-2_000_000:]
                target = diagnostics / f'{time.time_ns()}-{path.name}'
                target.write_text(text.replace(str(llm_params.api.key), '[redacted]'))
                target.chmod(0o600)
            draft = Path(work) / 'report.html'
            if draft.is_file():
                diagnostics.mkdir(exist_ok=True, mode=0o700)
                shutil.copyfile(draft, diagnostics / f'report-draft-{time.time_ns()}.html')
        report = output_dir / REPORT_FILENAME
        shutil.copyfile(Path(work) / "report.html", report)
        if not report.stat().st_size:
            raise ValueError("OpenCode produced an empty report")
    return report



def generate_reports(papers, llm_params, top_k=10):
    """Generate the selected local reports independently of Zotero availability."""
    if type(top_k) is not int or top_k < 0:
        raise ValueError("zotero.save_top_k must be a non-negative integer")
    selected = {}
    for paper in papers[:top_k]:
        selected.setdefault(canonical_identifier(paper.url), paper)
    receipts = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        pending = {}
        for identity, paper in selected.items():
            directory = Path('outputs/reports') / stable_item_key(identity)
            report = directory / REPORT_FILENAME
            receipt = {'url': paper.url}
            receipts.append(receipt)
            if report.is_file() and report.stat().st_size:
                receipt.update(status='existing', report_path=str(report))
            else:
                pending[pool.submit(generate_report, paper, llm_params, directory)] = receipt
        for future in as_completed(pending):
            receipt = pending[future]
            try:
                report = future.result()
                receipt.update(status='saved', report_path=str(report))
                logger.info(f"Saved local illustrated report for {receipt['url']}")
            except Exception as exc:
                receipt.update(status='failed', error=type(exc).__name__)
                logger.warning(f"Report failed for {receipt['url']}: {type(exc).__name__}")
    write_results('report-results.json', receipts)
