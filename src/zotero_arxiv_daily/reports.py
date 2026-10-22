"""Generate illustrated reports with OpenCode and publish them through WebDAV."""

import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
import zipfile

import requests

REPORT_TAG = "zotero-arxiv-daily:illustrated-report:v1"
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


def has_report(children: list[dict]) -> bool:
    return any(
        child.get("data", {}).get("itemType") == "attachment"
        and not child["data"].get("deleted")
        and any(tag.get("tag") == REPORT_TAG for tag in child["data"].get("tags", []))
        for child in children
    )


def generate_report(paper, llm_params, output_dir: Path) -> Path:
    """Reuse the configured model; keep OpenCode's generation/retry defaults."""
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
    # Do not pass SMTP, Zotero, WebDAV or GitHub credentials to the paper agent.
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
        with (output_dir / "agent.jsonl").open("w", encoding="utf-8") as log:
            subprocess.run(
                ["opencode", "run", "--model", selected_model, "--format", "json", "--dir", work, prompt],
                cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
            )
        report = output_dir / REPORT_FILENAME
        shutil.copyfile(Path(work) / "report.html", report)
        if not report.stat().st_size:
            raise ValueError("OpenCode produced an empty report")
    return report


def upload_report(zot, parent_key: str, report: Path, webdav, user_id: str) -> str:
    """Publish bytes first, then the attachment. Stable keys make retries idempotent."""
    key = stable_item_key(f"{user_id}:{parent_key}:{REPORT_TAG}")
    content = report.read_bytes()
    md5 = hashlib.md5(content).hexdigest()
    mtime = report.stat().st_mtime_ns // 1_000_000
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(report, arcname=REPORT_FILENAME)
    prop = f'<properties version="1"><mtime>{mtime}</mtime><hash>{md5}</hash></properties>'
    base = str(webdav.url).rstrip("/")
    if not base.endswith("/zotero"):
        base += "/zotero"
    with requests.Session() as session:
        session.auth = (str(webdav.username), str(webdav.password))
        for suffix, data in (("zip", archive.getvalue()), ("prop", prop.encode())):
            response = session.put(f"{base}/{key}.{suffix}", data=data, timeout=120)
            response.raise_for_status()
    attachment = dict(zot.item_template("attachment", "imported_file"))
    attachment.update({
        "key": key, "version": 0, "parentItem": parent_key,
        "title": "中文图文解读", "filename": REPORT_FILENAME,
        "contentType": "text/html", "md5": md5, "mtime": mtime,
        "tags": [{"tag": REPORT_TAG}],
    })
    response = zot.create_items([attachment])
    if not any(response.get(field, {}).get("0") for field in ("successful", "unchanged", "success")):
        raise RuntimeError(f"Zotero report attachment creation failed: {response.get('failed', {})}")
    return key
