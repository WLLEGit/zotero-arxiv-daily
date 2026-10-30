"""Personalized reading order using RankLLM's listwise ranking and response repair."""
import json
import random
import math
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf
from rank_llm.data import Candidate, Query, Request
from rank_llm.rerank.listwise.rank_gpt import SafeOpenai
from rank_llm.rerank.listwise.listwise_rankllm import ListwiseRankLLM
from rank_llm.rerank.listwise.response_utils import clean_ranking_response
import rank_llm

from ..protocol import _request_llm


class ReaderRanker(SafeOpenai):
    """Reuse RankLLM's pipeline with our existing client and bounded SDK retries."""
    def __init__(self, client, params, window):
        template = Path(rank_llm.__file__).parent / 'rerank/prompt_templates/rank_gpt_template.yaml'
        # Initialize the pipeline directly; SafeOpenai's constructor configures a global client.
        ListwiseRankLLM.__init__(self, model=params['generation_kwargs']['model'],
            context_size=1_048_576, prompt_template_path=str(template),
            window_size=window, stride=window, batch_size=5, max_passage_words=6000)
        self._output_token_estimate = None
        self.client, self.params = client, params

    def run_llm(self, prompt, current_window_size=None):
        for attempt in range(2):
            text = _request_llm(self.client, self.params, prompt) or ''
            numbers = clean_ranking_response(text).split()
            if any(value.isdigit() and 1 <= int(value) <= current_window_size for value in numbers):
                # RankLLM removes duplicates/unknown IDs and preserves valid selections.
                return text, 0
            logger.warning('No usable LLM ranking; retrying the current group')
        raise RuntimeError('LLM returned no usable reading-priority ranking')


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

    def rank(groups, window):
        ranker = ReaderRanker(client, params, window)
        requests = [Request(query=Query(text=query, qid=str(i)), candidates=group)
                    for i, group in enumerate(groups)]
        return ranker.rerank_batch(requests, rank_end=window, top_k_retrieve=window)

    # All candidates reach the LLM; small parallel groups feed one final comparison.
    if len(candidates) > 100:
        shuffled = list(candidates)
        random.Random(0).shuffle(shuffled)
        groups = [shuffled[i:i+100] for i in range(0, len(shuffled), 100)]
        keep = min(100, max(20, math.ceil(min(limit, len(candidates)) / (len(candidates) // 100))))
        finalists = [candidate for result in rank(groups, 100) for candidate in result.candidates[:keep]]
    else:
        finalists = candidates
    result = rank([finalists], len(finalists))[0]
    selected = [int(candidate.docid) for candidate in result.candidates]
    chosen = set(selected)
    ranked = [papers[i] for i in selected] + [paper for i, paper in enumerate(papers) if i not in chosen]
    logger.info(f'RankLLM ranked {len(finalists)} finalists from {len(papers)} candidates')
    return ranked
