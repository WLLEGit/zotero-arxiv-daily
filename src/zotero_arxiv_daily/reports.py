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
from tempfile import TemporaryDirectory

from loguru import logger

REPORT_FILENAME = "daily-arxiv-report.html"


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
    """After native retries are exhausted, resume rate-limited sessions in place."""
    base_command = command[:-1]
    with log_path.open("w+", encoding="utf-8") as log:
        for attempt in range(4):
            start = log.tell()
            try:
                subprocess.run(command, cwd=work, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
                return
            except subprocess.CalledProcessError:
                log.seek(start)
                error = None
                for line in log:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict) and event.get("type") == "error":
                        error = event
                log.seek(0, os.SEEK_END)
                data = (error or {}).get("error", {}).get("data", {})
                session = (error or {}).get("sessionID")
                if data.get("statusCode") != 429 or not session or attempt == 3:
                    raise
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
                logger.warning(f"Report rate limited; waiting {delay:.0f}s before session retry {attempt + 1}/3")
                time.sleep(delay)
                # Keep the original HOME, working files and persisted session database.
                command = [*base_command, "--session", session,
                           "继续当前论文的图文解读，复用已有阅读结果和文件，完成 report.html。"]


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
    with TemporaryDirectory(prefix="daily-arxiv-report-") as work:
        home = Path(work) / ".home"
        home.mkdir()
        env = {key: value for key, value in os.environ.items() if key in inherited}
        env.update({
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_DATA_HOME": str(home / ".local/share"),
            "XDG_STATE_HOME": str(home / ".local/state"),
            "REPORT_API_KEY": str(llm_params.api.key),
            "OPENCODE_CONFIG_CONTENT": json.dumps(config),
            "OPENCODE_DISABLE_AUTOUPDATE": "1",
        })
        _run_agent(
            ["opencode", "run", "--model", selected_model, "--format", "json", "--dir", work, prompt],
            work, env, output_dir / "agent.jsonl",
        )
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
