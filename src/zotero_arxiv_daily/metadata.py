"""Translate abstracts and extract affiliations in one request per paper."""
import json
import hashlib
import time
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import tiktoken
from loguru import logger
from tqdm import tqdm

from .protocol import _request_llm, llm_observer
from .outputs import write_results


def enrich_paper(paper, client, llm_params):
    paper.tldr, paper.affiliations = paper.abstract or '', None
    if not paper.abstract and not paper.full_text:
        return
    language = llm_params.get('language', 'English')
    chinese = str(language).lower() in {'chinese', 'zh', 'zh-cn', '中文'}
    instruction = '请将以下论文摘要翻译成中文。' if chinese else f'Translate the following paper abstract into {language}.'
    combined = instruction + (
        ' 同时从论文开头提取作者机构，按作者顺序保留顶层机构并去重，未找到时返回空列表。'
        ' 返回 JSON 对象，translation 字段为摘要译文，affiliations 字段为机构名称字符串列表。'
    )
    encoder = tiktoken.encoding_for_model('gpt-4o')
    beginning = encoder.decode(encoder.encode(paper.full_text or '', disallowed_special=())[:2000])
    attempts = []
    state = {'url': paper.url, 'status': 'failed'}
    for attempt in range(3):
        if attempt: time.sleep((15, 45)[attempt - 1])
        info = {}; token = llm_observer.set(info.update); text = ''
        try:
            # Last resort translates directly, without another structured-output dependency.
            messages = [{'role': 'system', 'content': instruction if attempt == 2 else combined},
                        {'role': 'user', 'content': paper.abstract if attempt == 2 else json.dumps(
                            {'abstract': paper.abstract or '', 'paper_beginning': beginning}, ensure_ascii=False)}]
            text = (_request_llm(client, llm_params, messages) or '').strip()
            if info.get('finish_reason') not in {None, 'stop', 'completed'}:
                raise ValueError('Incomplete model response')
            if attempt == 2:
                if text.startswith(('{', '[', '```')):
                    raise ValueError('Expected a plain translation')
                translation = text
            else:
                # Decode a JSON object even when it is surrounded by a code fence or prose.
                start = text.find('{')
                if start < 0: raise ValueError('Missing JSON object')
                result, _ = json.JSONDecoder().raw_decode(text[start:])
                translation = result.get('translation')
                affiliations = result.get('affiliations')
                if isinstance(affiliations, list) and all(isinstance(a, str) for a in affiliations):
                    paper.affiliations = list(dict.fromkeys(a.strip() for a in affiliations if a.strip()))
            if not isinstance(translation, str) or not translation.strip():
                raise ValueError('Missing translation')
            if chinese and not re.search('[\u4e00-\u9fff]', translation):
                raise ValueError('Missing Chinese translation')
            paper.tldr = translation.strip(); state['status'] = 'translated'
            break
        except Exception as exc:
            attempts.append({'error_type': type(exc).__name__, 'response': text[:64000],
                             **{k: info.get(k) for k in ('request_id', 'finish_reason', 'usage')}})
            status = getattr(exc, 'status_code', None)
            retryable = (isinstance(exc, (ValueError, TypeError, AttributeError)) or
                         status in {429, 500, 502, 503, 504} or
                         type(exc).__name__ in {'APIConnectionError', 'APITimeoutError', 'RemoteProtocolError'})
            if status in {400, 401, 403, 404} or not retryable: break
        finally:
            llm_observer.reset(token)
    if attempts:
        folder = Path('outputs/metadata'); folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        identity = hashlib.sha256(paper.url.encode()).hexdigest()[:16]
        path = folder / f'{identity}-{time.time_ns()}.json'
        path.write_text(json.dumps({**state, 'attempts': attempts}, ensure_ascii=False)); path.chmod(0o600)
    if state['status'] == 'failed':
        logger.warning(f'Abstract translation failed for {paper.url} after {len(attempts)} attempt(s)')
    return {**state, 'attempts': len(attempts) + (state['status'] == 'translated')}


def enrich_papers(papers, client, llm_params):
    results = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        pending = [pool.submit(enrich_paper, paper, client, llm_params) for paper in papers]
        for future in tqdm(as_completed(pending), total=len(pending)):
            result = future.result()
            if result: results.append(result)
            write_results("metadata-results.json", results)
