"""Source coverage, evidence validation and safe note rendering (no live API)."""

import json
from types import SimpleNamespace

import pytest
import tiktoken

from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily.analysis import (
    SECTIONS, _source_blocks, canonical_identifier, generate_analysis, stable_item_key,
)


def result(quote="Throughput was 100 tokens/s.", block=1):
    return {
        "sections": [{"heading": h, "body": "这是基于原文的中文解读 [S1]。"} for h in SECTIONS],
        "evidence": [{"block": block, "quote": quote, "claim": "原文报告的吞吐量"}],
    }


def client_for(data, calls, finish="stop"):
    def create(**kwargs):
        calls.append(kwargs)
        content = data(kwargs) if callable(data) else json.dumps(data, ensure_ascii=False)
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason=finish, message=SimpleNamespace(content=content),
        )])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def params():
    return {"language": "English", "generation_kwargs": {"model": "configured-model"}}


@pytest.mark.parametrize("value", [
    "https://arxiv.org/abs/2610.08268v1", "http://arxiv.org/pdf/2610.08268v2.pdf",
    "arXiv:2610.08268v3", "https://arxiv.org/html/2610.08268",
])
def test_arxiv_parent_identity_ignores_url_form_and_version(value):
    assert canonical_identifier(value) == "arxiv:2610.08268"


def test_old_arxiv_identity():
    assert canonical_identifier("https://arxiv.org/abs/hep-th/9901001v2") == "arxiv:hep-th/9901001"


def test_stable_keys_use_zotero_alphabet():
    key = stable_item_key("user:paper")
    assert len(key) == 8 and set(key) <= set("23456789ABCDEFGHJKLMNPQRSTUVWXYZ")
    assert key == stable_item_key("user:paper")
    assert key != stable_item_key("another-user:paper")


def test_split_preserves_unicode_and_tail():
    text = ("方法🦆 中文实验与局限\n" * 120) + "FINAL EXPERIMENT"
    enc = tiktoken.get_encoding("o200k_base")
    blocks = _source_blocks(text, enc, 100)
    assert len(blocks) > 1
    assert "".join(blocks) == text
    assert all(len(enc.encode(b)) <= 100 for b in blocks)
    assert blocks[-1].endswith("FINAL EXPERIMENT")


def test_analysis_uses_configured_model_and_chinese_without_changing_tldr():
    paper = make_sample_paper(full_text="Throughput was 100 tokens/s.", tldr="Original English TLDR")
    calls = []
    note = generate_analysis(client_for(result(), calls), paper, params())
    assert "中文详细解读" in note
    assert "[S1]" in note and "Throughput was 100 tokens/s." in note
    assert calls[0]["model"] == "configured-model"
    assert "始终用中文" in calls[0]["messages"][0]["content"]
    assert paper.tldr == "Original English TLDR"


def test_safe_html_escaping_and_link_scheme():
    paper = make_sample_paper(full_text="<script>alert(1)</script>", title="<img src=x>", url="javascript:alert(2)")
    data = result(quote=paper.full_text)
    data["sections"][0]["body"] = "原文：<script>alert(1)</script>"
    note = generate_analysis(client_for(data, []), paper, params())
    assert "<script>" not in note and "<img" not in note
    assert "&lt;script&gt;" in note
    assert "javascript:" not in note


@pytest.mark.parametrize("finish", ["length", "content_filter", "tool_calls"])
def test_incomplete_response_is_not_saved(finish):
    with pytest.raises(ValueError, match="did not finish"):
        generate_analysis(client_for(result(), [], finish), make_sample_paper(), params())


@pytest.mark.parametrize("text", [None, "", "   "])
def test_missing_full_text_does_not_fabricate_detailed_note(text):
    calls = []
    with pytest.raises(ValueError, match="Full text unavailable"):
        generate_analysis(client_for(result(), calls), make_sample_paper(full_text=text), params())
    assert not calls


@pytest.mark.parametrize("data", [
    {"sections": []}, result(quote="Fabricated benchmark"), result(block=999),
    {**result(), "evidence": []},
    {**result(), "sections": [{"heading": h, "body": ""} for h in SECTIONS]},
    {**result(), "sections": [{"heading": h, "body": "中文 [S999]"} for h in SECTIONS]},
])
def test_invalid_analysis_is_rejected(data):
    with pytest.raises(ValueError):
        generate_analysis(client_for(data, []), make_sample_paper(full_text="Throughput was 100 tokens/s."), params())


def test_long_paper_reads_every_block_including_experiments_at_end(monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.analysis.MAX_INPUT_TOKENS", 3000)
    monkeypatch.setattr("zotero_arxiv_daily.analysis.CHUNK_TOKENS", 1000)
    paper = make_sample_paper(full_text=("Background text. " * 4000) + "FINAL EXPERIMENT")
    calls = []
    def reply(kwargs):
        user = kwargs["messages"][1]["content"]
        if "供后续解读使用的事实" in user:
            return json.dumps({"facts": "本段事实已提取。", "evidence": []}, ensure_ascii=False)
        # Evidence for the final block must still be checked against original text.
        final_block = len(calls) - 1
        return json.dumps(result(quote="FINAL EXPERIMENT", block=final_block), ensure_ascii=False)
    note = generate_analysis(client_for(reply, calls), paper, params())
    assert len(calls) > 2
    assert "FINAL EXPERIMENT" in calls[-2]["messages"][1]["content"]
    assert "全部提取文本已分块读取" in note


def test_input_over_budget_fails_explicitly_instead_of_truncating(monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.analysis.MAX_INPUT_TOKENS", 2000)
    paper = make_sample_paper(full_text="Throughput was 100 tokens/s.", abstract="word " * 10000)
    calls = []
    with pytest.raises(ValueError, match="exceeds budget"):
        generate_analysis(client_for(result(), calls), paper, params())
    assert calls == []


def test_reuses_existing_model_and_generation_options_without_mutating_config():
    calls = []
    config = params()
    config["generation_kwargs"].update(max_completion_tokens=99, max_tokens=10, n=3, stream=True, temperature=0.2)
    generate_analysis(client_for(result(), calls), make_sample_paper(full_text="Throughput was 100 tokens/s."), config)
    assert calls[0]["model"] == "configured-model"
    assert calls[0]["max_completion_tokens"] == 8000
    assert calls[0]["temperature"] == 0.2
    assert config["generation_kwargs"]["max_completion_tokens"] == 99
    assert config["generation_kwargs"]["max_tokens"] == 10
    assert "max_tokens" not in calls[0] and "stream" not in calls[0] and "n" not in calls[0]
