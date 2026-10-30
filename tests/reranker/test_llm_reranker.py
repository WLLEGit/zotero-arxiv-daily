"""Exercise the real RankLLM grouping/permutation pipeline with offline model replies."""
import copy
import re
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily.executor import Executor
from zotero_arxiv_daily.reranker.llm import ReaderRanker, rerank_with_llm


def candidates(count):
    return [make_sample_paper(title=f'Paper {i}',abstract=f'Abstract {i}',
            score=9-i/100,url=f'https://arxiv.org/abs/2610.{i:05d}') for i in range(count)]


@pytest.fixture(autouse=True)
def offline_token_budget(monkeypatch):
    # Tokenizer downloads are unrelated to permutation and protocol behavior.
    monkeypatch.setattr(ReaderRanker,'get_num_tokens',lambda *args:1000)
    monkeypatch.setattr(ReaderRanker,'num_output_tokens',lambda *args:1000)


def passage_numbers(messages):
    return [int(re.match(r'\[(\d+)\]',m['content']).group(1)) for m in messages
            if m['role']=='user' and re.match(r'\[\d+\]',m['content'])]


def test_rankllm_keeps_valid_order_deduplicates_and_ignores_unknown_ids(config,monkeypatch):
    papers=candidates(5);before=copy.deepcopy(papers);params=copy.deepcopy(config.llm)
    def request(client,p,messages):
        assert p['generation_kwargs']['temperature']==0
        assert 'Preferences' in messages[1]['content']
        assert 'Paper 4' in str(messages) and 'Abstract 4' in str(messages)
        return '[5] > [3] > [5] > [999] > [1]'
    monkeypatch.setattr('zotero_arxiv_daily.reranker.llm._request_llm',request)
    ranked=rerank_with_llm(papers,object(),config.llm,'Preferences')
    assert [p.title for p in ranked]==['Paper 4','Paper 2','Paper 0','Paper 1','Paper 3']
    assert papers==before and config.llm==params
    assert all(ranked[i] is papers[j] for i,j in enumerate([4,2,0,1,3]))


def test_feedback_is_used_even_without_a_manually_written_prompt(config,monkeypatch):
    examples=OmegaConf.create([{'title':'TileBench','preference':'感兴趣','reason':'诊断GPU执行瓶颈'}])
    def request(client,params,messages):
        assert 'TileBench' in messages[1]['content'] and '诊断GPU执行瓶颈' in messages[1]['content']
        return '[2] > [1]'
    monkeypatch.setattr('zotero_arxiv_daily.reranker.llm._request_llm',request)
    assert rerank_with_llm(candidates(2),object(),config.llm,None,examples)[0].title=='Paper 1'


@pytest.mark.parametrize('text',['','not a ranking','[999]'])
def test_unusable_response_retries_and_never_silently_falls_back(config,monkeypatch,text):
    calls=[]
    def request(*args):calls.append(1);return text
    monkeypatch.setattr('zotero_arxiv_daily.reranker.llm._request_llm',request)
    with pytest.raises(RuntimeError,match='no usable'):
        rerank_with_llm(candidates(3),object(),config.llm,'Preferences')
    assert len(calls)==2


def test_transient_bad_format_can_recover(config,monkeypatch):
    replies=iter(['not a ranking','[3] > [2] > [1]'])
    monkeypatch.setattr('zotero_arxiv_daily.reranker.llm._request_llm',lambda *args:next(replies))
    assert rerank_with_llm(candidates(3),object(),config.llm,'Preferences')[0].title=='Paper 2'


@pytest.mark.parametrize('mode',['chat_completion','response'])
def test_existing_client_protocol_and_model_are_reused(config,mode):
    config.llm.api_mode=mode;config.llm.generation_kwargs.model='deepseek-v4.1-flash-1017'
    seen=[]
    def create(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(output_text='[2] > [1]',choices=[SimpleNamespace(message=SimpleNamespace(content='[2] > [1]'))])
    client=SimpleNamespace(responses=SimpleNamespace(create=create),chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert rerank_with_llm(candidates(2),client,config.llm,'Preferences')[0].title=='Paper 1'
    assert seen[0]['model']=='deepseek-v4.1-flash-1017'
    if mode=='response':assert seen[0]['max_output_tokens']==4096 and 'max_tokens' not in seen[0]
    else:assert seen[0]['extra_body']['thinking']['type']=='disabled'


def test_every_candidate_reaches_group_ranking_and_finalists_are_compared(config,monkeypatch):
    seen=[]
    def request(client,params,messages):
        seen.append(messages)
        return ' > '.join(f'[{i}]' for i in reversed(passage_numbers(messages)))
    monkeypatch.setattr('zotero_arxiv_daily.reranker.llm._request_llm',request)
    papers=candidates(250)
    ranked=rerank_with_llm(papers,object(),config.llm,'Preferences')
    groups=seen[:-1]
    assert sorted(len(passage_numbers(m)) for m in groups)==[50,100,100]
    titles=[re.search(r'Paper \d+',m['content']).group() for group in groups for m in group
            if m['role']=='user' and re.match(r'\[\d+\]',m['content'])]
    assert set(titles)=={p.title for p in papers} and len(titles)==250
    assert len(passage_numbers(seen[-1]))==150
    assert len({id(p) for p in ranked})==250


@pytest.mark.parametrize('prompt,count',[(None,3),('',3),(' ',3),('Preferences',0),('Preferences',1)])
def test_disabled_or_trivial_pool_makes_no_request(config,monkeypatch,prompt,count):
    monkeypatch.setattr('zotero_arxiv_daily.reranker.llm._request_llm',lambda *args:pytest.fail('Unexpected call'))
    papers=candidates(count)
    assert rerank_with_llm(papers,object(),config.llm,prompt) is papers


def test_delivery_uses_final_order_and_preserves_feedback(config,monkeypatch):
    config.reranker.prompt='Preferences';config.reranker.feedback_examples=[{'title':'TileBench','preference':'感兴趣','reason':'GPU机制'}]
    config.executor.max_paper_num=3
    executor=Executor.__new__(Executor);executor.config=config;executor.openai_client=object()
    papers=candidates(25)
    executor.fetch_zotero_corpus=lambda:[object()];executor.filter_corpus=lambda x:x
    executor.retrievers={'arxiv':SimpleNamespace(retrieve_papers=lambda:papers)}
    executor.reranker=SimpleNamespace(rerank=lambda papers,corpus:papers)
    events=[]
    def request(client,params,messages):
        assert 'TileBench' in str(messages)
        assert len(passage_numbers(messages))==25
        return '[25] > [24] > [23] > [25]'
    monkeypatch.setattr('zotero_arxiv_daily.reranker.llm._request_llm',request)
    monkeypatch.setattr('zotero_arxiv_daily.executor.enrich_papers',lambda rows,*args:events.append(('metadata',[p.title for p in rows])))
    monkeypatch.setattr('zotero_arxiv_daily.executor.render_email',lambda papers:[p.title for p in papers])
    monkeypatch.setattr('zotero_arxiv_daily.executor.send_email',lambda config,body:events.append(('email',body)))
    monkeypatch.setattr('zotero_arxiv_daily.executor.save_papers',lambda papers,config:events.append(('zotero',[p.title for p in papers])))
    monkeypatch.setattr('zotero_arxiv_daily.executor.generate_reports',lambda papers,*args:events.append(('reports',[p.title for p in papers])))
    executor.run()
    expected=['Paper 24','Paper 23','Paper 22']
    assert events==[(stage,expected) for stage in ['metadata','email','zotero','reports']]
