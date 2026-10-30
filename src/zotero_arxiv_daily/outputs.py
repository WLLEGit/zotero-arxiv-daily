"""Local result exports consumed by the self-hosted reader."""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from loguru import logger


def write_results(filename, rows):
    output = Path("outputs") / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output)


def export_paper_index(papers: list, sources: list[str], *, ranked=None, save_top_k=10) -> None:
    """Export public candidates and optional reading order; no private corpus or model calls."""
    try:
        output = Path("outputs/paper-index.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "sources": sources,
            "status": "complete",
            "papers": [
                {field: getattr(paper, field, None) for field in
                 ("source", "title", "authors", "abstract", "url", "pdf_url")}
                for paper in papers
            ],
        }
        if ranked is not None:
            payload["recommendation_date"] = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
            payload["recommendations"] = [
                {"url": paper.url, "rank": rank, "selected_for_zotero": rank <= save_top_k}
                for rank, paper in enumerate(ranked, 1)
            ]
        if ranked is not None:
            payload["reading"] = [
                {"url": paper.url, **{field: getattr(paper, field, None)
                 for field in ("score", "tldr", "affiliations")}}
                for paper in ranked
            ]
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(output)
        logger.info(f"Exported {len(papers)} public paper candidates")
    except Exception as exc:
        logger.warning(f"Paper index export failed: {type(exc).__name__}")
