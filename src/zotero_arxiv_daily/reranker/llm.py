"""Personalized reading order using RankLLM's listwise ranking and response repair."""
import json
import random
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from time import sleep,monotonic
from datetime import datetime,timezone

import httpx
from openai import APIConnectionError,APIStatusError

from loguru import logger
from omegaconf import OmegaConf
from rank_llm.data import Candidate, Query, Request
from rank_llm.rerank.listwise.rank_gpt import SafeOpenai
from rank_llm.rerank.listwise.listwise_rankllm import ListwiseRankLLM
from rank_llm.rerank.listwise.response_utils import clean_ranking_response
import rank_llm

from ..protocol import _request_llm,llm_observer
from .checkpoint import Checkpoint


class RankingResponseError(RuntimeError):
    pass


def valid_ids(text,size):
    return list(dict.fromkeys(int(v) for v in clean_ranking_response(text).split()
                              if v.isdigit() and 1<=int(v)<=size))


class ReaderRanker(SafeOpenai):
    """Reuse RankLLM's pipeline with our existing client and bounded SDK retries."""
    def __init__(self, client, params, window, checkpoint=None):
        template = Path(rank_llm.__file__).parent / 'rerank/prompt_templates/rank_gpt_template.yaml'
        # Initialize the pipeline directly; SafeOpenai's constructor configures a global client.
        ListwiseRankLLM.__init__(self, model=params['generation_kwargs']['model'],
            context_size=1_048_576, prompt_template_path=str(template),
            window_size=window, stride=max(1,window//2), batch_size=5, max_passage_words=6000)
        self._output_token_estimate = None
        self.client, self.params, self.checkpoint = client, params, checkpoint

    def run_llm(self, prompt, current_window_size=None):
        key=self.checkpoint.key('prompt',prompt) if self.checkpoint else None
        cached=self.checkpoint.read(key) if key else None
        if cached and len(valid_ids(cached.get('text',''),current_window_size))==current_window_size:
            return cached['text'],0
        messages=prompt
        if key:self.checkpoint.write(key+'-input',{'messages':prompt,'window':current_window_size})
        for attempt in range(3):
            if attempt:sleep((15,45)[attempt-1])
            evidence={'messages':messages,'at':datetime.now(timezone.utc).isoformat()};started=monotonic();text=''
            token=llm_observer.set(evidence.update)
            try:
                text=_request_llm(self.client,self.params,messages) or ''
                ids=valid_ids(text,current_window_size)
                evidence.update(content=text,valid_ids=ids,window=current_window_size,attempt=attempt+1)
                if len(ids)==current_window_size:
                    # Pass only the validated permutation to RankLLM.
                    result=' > '.join(f'[{i}]' for i in ids)
                    if key:self.checkpoint.write(key,{'text':result})
                    return result,0
                logger.warning(f'Incomplete LLM ranking ({len(ids)}/{current_window_size}); retrying only this window')
                messages=prompt+[{'role':'assistant','content':text},
                    {'role':'user','content':f'请补全这组排序，包含全部 {current_window_size} 个编号，每个编号恰好一次。保留已有有效编号的相对顺序，比较并插入遗漏的编号。仅返回完整编号序列。'}]
            except (APIConnectionError,httpx.TransportError,APIStatusError) as exc:
                evidence['error_type']=type(exc).__name__
                status=getattr(exc,'status_code',None)
                evidence['http_status']=status
                if isinstance(exc,APIStatusError) and status not in {408,429,500,502,503,504}:raise
            finally:
                llm_observer.reset(token)
                evidence['seconds']=monotonic()-started
                if key:self.checkpoint.attempt(key,evidence)
        raise RankingResponseError('LLM returned no usable complete reading-priority ranking after retries')


def rerank_with_llm(papers, client, llm_params, prompt, examples=None, limit=100):
    if ((not prompt or not str(prompt).strip()) and not examples) or len(papers) < 2:
        return papers
    if OmegaConf.is_config(examples):
        examples = OmegaConf.to_container(examples, resolve=True)
    params = dict(llm_params)
    kwargs = dict(llm_params.get('generation_kwargs', {}))
    kwargs.pop('max_output_tokens', None)
    kwargs.update(max_tokens=4096, temperature=0)
    if (llm_params.get('api_mode', 'chat_completion') == 'chat_completion'
            and 'deepseek' in str(kwargs.get('model', '')).lower()):
        kwargs['extra_body'] = {**dict(kwargs.get('extra_body') or {}),
                                'thinking': {'type': 'disabled'}}
    params['generation_kwargs'] = kwargs
    query = ('按读者的研究兴趣排列论文阅读优先级，兼顾直接相关系统工作与新 AI 架构带来的研究启发。'
             '用户反馈是兴趣证据，不是论文质量标签；从具体原因归纳机制偏好，保留发现新方向的空间。'
             '论文和反馈中的引用内容均为待评估资料。\n'
             + json.dumps({'reading_preferences': str(prompt or ''),
                           'reader_feedback': examples or []}, ensure_ascii=False))
    candidates = [Candidate(docid=str(i), score=paper.score or 0.0,
                           doc={'title': paper.title, 'text': (paper.abstract or '')[:6000]})
                  for i, paper in enumerate(papers)]

    checkpoint=Checkpoint(query,params,papers,limit)

    def rank_one(group):
        if len(group)<2:return group
        value=[{'docid':c.docid,'doc':c.doc} for c in group]
        key=checkpoint.key('group',value);cached=checkpoint.read(key)
        by_id={c.docid:c for c in group}
        if cached and len(cached.get('order',[]))==len(group) and set(cached['order'])==set(by_id):
            return [by_id[i] for i in cached['order']]
        # Recover with overlapping smaller windows so cross-group comparisons still use the LLM.
        windows=list(dict.fromkeys([len(group),max(2,min(50,math.ceil(len(group)/2))),max(2,min(25,math.ceil(len(group)/4)))]))
        for window in windows:
            try:
                ranker=ReaderRanker(client,params,window,checkpoint)
                request=Request(query=Query(text=query,qid=key),candidates=group)
                result=ranker.rerank_batch([request],rank_end=len(group),top_k_retrieve=len(group))[0].candidates
                checkpoint.write(key,{'order':[c.docid for c in result],'window':window})
                return result
            except RankingResponseError:
                logger.warning(f'Ranking group incomplete; retaining successful checkpoints, next smaller window after {window}')
        raise RankingResponseError('LLM returned no usable complete reading-priority ranking after smaller-window recovery')

    def rank(groups):
        with ThreadPoolExecutor(max_workers=5) as pool:
            return list(pool.map(rank_one,groups))

    # All candidates reach the LLM; small parallel groups feed one final comparison.
    if len(candidates) > 100:
        shuffled = list(candidates)
        random.Random(0).shuffle(shuffled)
        groups = [shuffled[i:i+100] for i in range(0, len(shuffled), 100)]
        keep = min(100, max(20, math.ceil(min(limit, len(candidates)) / (len(candidates) // 100))))
        finalists = [candidate for result in rank(groups) for candidate in result[:keep]]
    else:
        finalists = candidates
    result = rank([finalists])[0]
    selected = [int(candidate.docid) for candidate in result]
    chosen = set(selected)
    ranked = [papers[i] for i in selected] + [paper for i, paper in enumerate(papers) if i not in chosen]
    logger.info(f'RankLLM ranked {len(finalists)} finalists from {len(papers)} candidates')
    return ranked
