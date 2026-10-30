"""Translate abstracts and extract affiliations in one request per paper."""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import tiktoken
from loguru import logger
from tqdm import tqdm

from .protocol import _request_llm


def enrich_paper(paper, client, llm_params):
    paper.tldr, paper.affiliations = paper.abstract or '', None
    if not paper.abstract and not paper.full_text:
        return
    try:
        language = llm_params.get('language', 'English')
        if str(language).lower() in {'chinese', 'zh', 'zh-cn', '中文'}:
            instruction = '请将以下论文摘要翻译成中文。'
        else:
            instruction = f'Translate the following paper abstract into {language}.'
        instruction += (
            ' 同时从论文开头提取作者机构，按作者顺序保留顶层机构并去重，未找到时返回空列表。'
            ' 返回 JSON 对象，translation 字段为摘要译文，affiliations 字段为机构名称字符串列表。'
        )
        encoder = tiktoken.encoding_for_model('gpt-4o')
        beginning = encoder.decode(encoder.encode(paper.full_text or '', disallowed_special=())[:2000])
        text = _request_llm(client, llm_params, [
            {'role': 'system', 'content': instruction},
            {'role': 'user', 'content': json.dumps(
                {'abstract': paper.abstract or '', 'paper_beginning': beginning}, ensure_ascii=False)},
        ]).strip()
        text = text.removeprefix('```json').removeprefix('```').removesuffix('```').strip()
        result = json.loads(text)
        translation = result.get('translation')
        if isinstance(translation, str) and translation.strip():
            paper.tldr = translation
        elif paper.abstract:
            logger.warning(f'Missing translation for {paper.url}; keeping original abstract')
        affiliations = result.get('affiliations')
        if isinstance(affiliations, list) and all(isinstance(a, str) for a in affiliations):
            paper.affiliations = list(dict.fromkeys(a.strip() for a in affiliations if a.strip()))
        else:
            logger.warning(f'Invalid affiliations for {paper.url}; keeping available translation')
    except Exception as exc:
        logger.warning(f'Paper metadata failed for {paper.url}: {type(exc).__name__}')


def enrich_papers(papers, client, llm_params):
    with ThreadPoolExecutor(max_workers=5) as pool:
        pending = [pool.submit(enrich_paper, paper, client, llm_params) for paper in papers]
        for future in tqdm(as_completed(pending), total=len(pending)):
            future.result()
