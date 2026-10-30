"""Reading-priority ranking, strict ID validation and delivery-order coverage."""
import copy
import json
from types import SimpleNamespace

import pytest

from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily.executor import Executor
from zotero_arxiv_daily.reranker.llm import rerank_with_llm


def candidates(count):
    return [make_sample_paper(title=f"Paper {i}", abstract=f"Abstract {i}",
                              score=9 - i / 100, url=f"https://arxiv.org/abs/2610.{i:05d}")
            for i in range(count)]


def test_promotes_twenty_preserves_tail_scores_and_model_config(config, monkeypatch):
    papers = candidates(25)
    before = copy.deepcopy(papers)
    config.llm.generation_kwargs.model = 'paratera/DeepSeek-V4.1-Flash-1017'
    model_config = copy.deepcopy(config.llm)
    client = object()

    def request(actual_client, params, messages):
        assert actual_client is client
        assert params['generation_kwargs']['model'] == model_config.generation_kwargs.model
        assert params['generation_kwargs']['extra_body']['thinking']['type'] == 'disabled'
        data = json.loads(messages[1]['content'])
        assert data['reading_preferences'] == 'Study new architectures for systems opportunities'
        rows = data['papers']
        assert all(set(row) == {'id', 'title', 'abstract'} for row in rows)
        assert all('arxiv' not in row['id'] for row in rows)
        ids = {row['title']: row['id'] for row in rows}
        return json.dumps({'ranking': [ids[f'Paper {i}'] for i in range(24, 4, -1)]})

    monkeypatch.setattr('zotero_arxiv_daily.reranker.llm._request_llm', request)
    ranked = rerank_with_llm(papers, client, config.llm, 'Study new architectures for systems opportunities')
    assert [p.title for p in ranked] == [f'Paper {i}' for i in range(24, 4, -1)] + [f'Paper {i}' for i in range(5)]
    assert len({id(p) for p in ranked}) == 25
    assert papers == before and config.llm == model_config


@pytest.mark.parametrize('kind', ['invalid_json', 'missing', 'duplicate', 'unknown', 'wrong_type', 'api_error', 'empty'])
def test_invalid_response_keeps_original_order(config, monkeypatch, kind):
    papers = candidates(3)

    def request(*args):
        rows = json.loads(args[2][1]['content'])['papers']
        ids = [r['id'] for r in rows]
        if kind == 'api_error':
            raise RuntimeError('Rate limited')
        if kind == 'invalid_json':
            return 'not json'
        if kind == 'empty':
            return ''
        selected = {'missing': ids[:2], 'duplicate': [ids[0]] * 3,
                    'unknown': ['invented'] + ids[1:], 'wrong_type': {'ranking': ids}}[kind]
        return json.dumps({'ranking': selected})

    monkeypatch.setattr('zotero_arxiv_daily.reranker.llm._request_llm', request)
    assert rerank_with_llm(papers, object(), config.llm, 'Preferences') is papers


@pytest.mark.parametrize('prompt,count', [(None, 3), ('', 3), ('  ', 3), ('Preferences', 0), ('Preferences', 1)])
def test_disabled_or_trivial_pool_makes_no_request(config, monkeypatch, prompt, count):
    def request(*args):
        pytest.fail('No model call should be made')
    monkeypatch.setattr('zotero_arxiv_daily.reranker.llm._request_llm', request)
    papers = candidates(count)
    assert rerank_with_llm(papers, object(), config.llm, prompt) is papers


def test_response_protocol_reuses_existing_client(config):
    config.llm.api_mode = 'response'
    seen = []

    def create(**kwargs):
        seen.append(kwargs)
        rows = json.loads(kwargs['input'][1]['content'])['papers']
        ids = {row['title']: row['id'] for row in rows}
        return SimpleNamespace(output_text='```json\n' + json.dumps({'ranking': [ids['Paper 1'], ids['Paper 0']]}) + '\n```')

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    ranked = rerank_with_llm(candidates(2), client, config.llm, 'Preferences')
    assert [p.title for p in ranked] == ['Paper 1', 'Paper 0']
    assert seen[0]['model'] == config.llm.generation_kwargs.model
    assert seen[0]['max_output_tokens'] == 4096 and 'max_tokens' not in seen[0]


def test_email_and_zotero_share_final_order_after_shortlisting(config, monkeypatch):
    config.reranker.prompt = 'Architectures and systems inspiration'
    config.executor.max_paper_num = 3
    all_papers = candidates(25)
    executor = Executor.__new__(Executor)
    executor.config = config
    executor.openai_client = object()
    executor.fetch_zotero_corpus = lambda: [object()]
    executor.filter_corpus = lambda x: x
    executor.retrievers = {'arxiv': SimpleNamespace(retrieve_papers=lambda: all_papers)}
    executor.reranker = SimpleNamespace(rerank=lambda papers, corpus: papers)
    events = []

    def request(client, params, messages):
        events.append('rerank')
        rows = json.loads(messages[1]['content'])['papers']
        assert {r['title'] for r in rows} == {'Paper 0', 'Paper 1', 'Paper 2'}
        ids = {r['title']: r['id'] for r in rows}
        return json.dumps({'ranking': [ids['Paper 2'], ids['Paper 0'], ids['Paper 1']]})

    monkeypatch.setattr('zotero_arxiv_daily.reranker.llm._request_llm', request)
    monkeypatch.setattr('zotero_arxiv_daily.executor.enrich_papers', lambda rows, *args: events.extend('metadata:' + p.title for p in rows))
    monkeypatch.setattr('zotero_arxiv_daily.executor.render_email', lambda papers: [p.title for p in papers])
    monkeypatch.setattr('zotero_arxiv_daily.executor.send_email', lambda config, body: events.append(('email', body)))
    monkeypatch.setattr('zotero_arxiv_daily.executor.save_papers', lambda papers, config: events.append(('zotero', [p.title for p in papers])))
    monkeypatch.setattr('zotero_arxiv_daily.executor.generate_reports', lambda *args: None)
    executor.run()
    expected = ['Paper 2', 'Paper 0', 'Paper 1']
    assert events == ['rerank', *['metadata:' + title for title in expected], ('email', expected), ('zotero', expected)]
