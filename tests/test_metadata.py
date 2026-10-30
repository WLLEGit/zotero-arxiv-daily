"""One-request metadata enrichment and bounded concurrency without model calls."""
import json
import threading
from types import SimpleNamespace

import pytest
import tiktoken

from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily import metadata


@pytest.mark.parametrize('mode', ['chat_completion', 'response'])
def test_one_request_returns_both_fields_and_preserves_full_abstract(config, mode):
    config.llm.api_mode = mode
    config.llm.language = 'Chinese'
    paper = make_sample_paper(abstract='abstract ' * 6000,
                              full_text='first-page ' * 6000 + 'EXCLUDED_END')
    received = []
    result = json.dumps({'translation': '中文完整译文', 'affiliations': ['B', 'A', 'B']})
    def create(**kwargs):
        received.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=result))], output_text=result)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                             responses=SimpleNamespace(create=create))
    metadata.enrich_paper(paper, client, config.llm)
    assert len(received) == 1
    kwargs = received[0]
    assert kwargs['model'] == config.llm.generation_kwargs.model
    messages = kwargs['messages' if mode == 'chat_completion' else 'input']
    assert messages[0]['content'].startswith('请将以下论文摘要翻译成中文。')
    payload = json.loads(messages[1]['content'])
    assert payload['abstract'] == paper.abstract
    encoder = tiktoken.encoding_for_model('gpt-4o')
    assert payload['paper_beginning'] == encoder.decode(encoder.encode(paper.full_text)[:2000])
    assert paper.tldr == '中文完整译文' and paper.affiliations == ['B', 'A']


@pytest.mark.parametrize('answer', ['', 'invalid json', '[]'])
def test_bad_response_keeps_original_abstract(config, monkeypatch, answer):
    monkeypatch.setattr(metadata, '_request_llm', lambda *args: answer)
    paper = make_sample_paper()
    metadata.enrich_paper(paper, None, config.llm)
    assert paper.tldr == paper.abstract and paper.affiliations is None


def test_malformed_affiliations_do_not_discard_translation(config, monkeypatch):
    monkeypatch.setattr(metadata, '_request_llm', lambda *args:
        '```json\n' + json.dumps({'translation': '译文', 'affiliations': 'invalid'}) + '\n```')
    paper = make_sample_paper()
    metadata.enrich_paper(paper, None, config.llm)
    assert paper.tldr == '译文' and paper.affiliations is None


def test_missing_translation_keeps_valid_affiliations(config, monkeypatch):
    monkeypatch.setattr(metadata, '_request_llm', lambda *args: json.dumps({'affiliations': ['A']}))
    paper = make_sample_paper()
    metadata.enrich_paper(paper, None, config.llm)
    assert paper.tldr == paper.abstract and paper.affiliations == ['A']


def test_empty_paper_makes_no_request(config, monkeypatch):
    def forbidden(*args): pytest.fail('No content should not call the model')
    monkeypatch.setattr(metadata, '_request_llm', forbidden)
    paper = make_sample_paper(abstract='', full_text=None)
    metadata.enrich_paper(paper, None, config.llm)
    assert paper.tldr == '' and paper.affiliations is None


def test_five_requests_overlap_preserve_order_and_isolate_failure(config, monkeypatch):
    papers = [make_sample_paper(title=str(i), abstract=str(i)) for i in range(10)]
    barrier, lock = threading.Barrier(5, timeout=10), threading.Lock()
    active = peak = calls = 0
    def request(client, params, messages):
        nonlocal active, peak, calls
        identity = json.loads(messages[1]['content'])['abstract']
        with lock:
            active += 1; calls += 1; peak = max(peak, active)
        barrier.wait()
        with lock: active -= 1
        barrier.wait()
        if identity == '3': raise RuntimeError('one failed request')
        return json.dumps({'translation': '译文' + identity, 'affiliations': ['A']})
    monkeypatch.setattr(metadata, '_request_llm', request)
    metadata.enrich_papers(papers, None, config.llm)
    assert peak == 5 and calls == 10
    assert [p.title for p in papers] == [str(i) for i in range(10)]
    assert papers[3].tldr == papers[3].abstract and papers[3].affiliations is None
    assert all(p.tldr == '译文' + str(i) for i, p in enumerate(papers) if i != 3)
