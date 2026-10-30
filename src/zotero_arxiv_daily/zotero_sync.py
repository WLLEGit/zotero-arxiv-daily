"""Sync selected paper metadata only; reports remain local to the reader."""
from pyzotero import zotero
from loguru import logger
from .reports import canonical_identifier, item_identifiers, stable_item_key
from .outputs import write_results


def _ensure_collection(zot, path: str) -> str:
    """Resolve a collection by its '/'-separated path, creating missing levels. Returns the collection key."""
    names = [p for p in path.split('/') if p]
    if not names:
        raise ValueError(f"Invalid zotero.save_collection: {path!r}")
    collections = zot.everything(zot.collections())
    by_key = {c['key']: c for c in collections}

    def full_path(c: dict) -> str:
        parent = c['data']['parentCollection']
        if parent and parent in by_key:
            return full_path(by_key[parent]) + '/' + c['data']['name']
        return c['data']['name']

    existing = {full_path(c): c['key'] for c in collections}
    parent_key = None
    cur = ''
    for name in names:
        cur = f"{cur}/{name}" if cur else name
        if cur in existing:
            parent_key = existing[cur]
            continue
        payload = {'name': name}
        if parent_key:
            payload['parentCollection'] = parent_key
        resp = zot.create_collections([payload])
        key = resp['successful']['0']['key']
        logger.info(f"Created Zotero collection '{cur}' ({key})")
        existing[cur] = key
        parent_key = key
    return parent_key

def _existing_papers(zot, collection_key: str) -> dict[str, dict]:
    """Prefer target-collection items; reuse papers elsewhere in the library."""
    items = zot.everything(zot.collection_items_top(collection_key))
    items += zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
    existing = {}
    for item in items:
        if item.get('data', {}).get('deleted') or not item.get('key'):
            continue
        for identifier in item_identifiers(item):
            existing.setdefault(identifier, item)
    return existing

def _created_key(response: dict) -> str:
    successful = response.get('successful', {}).get('0')
    if successful:
        return successful['key']
    # Zotero returns keys (not full items) for unchanged writes.
    key = response.get('unchanged', {}).get('0') or response.get('success', {}).get('0')
    if key:
        return key
    raise RuntimeError(f"Zotero item creation failed: {response.get('failed', {})}")

def _split_creator(name: str) -> dict:
    parts = name.strip().split()
    if len(parts) >= 2:
        return {'creatorType': 'author', 'firstName': ' '.join(parts[:-1]), 'lastName': parts[-1]}
    return {'creatorType': 'author', 'name': name}

def _paper_to_item(paper, template: dict, collection_key: str) -> tuple[dict, list[str]]:
    item = dict(template)
    item['title'] = paper.title
    item['abstractNote'] = paper.abstract or ''
    item['url'] = paper.url or ''
    item['creators'] = [_split_creator(a) for a in (paper.authors or [])]
    item['collections'] = [collection_key]
    identifiers = [paper.url] if paper.url else []
    if paper.source == 'arxiv' and paper.url:
        arxiv_id = paper.url.rsplit('/abs/', 1)[-1].rsplit('/', 1)[-1]
        item['repository'] = 'arXiv'
        item['archiveID'] = f'arXiv:{arxiv_id}'
        identifiers.append(item['archiveID'])
    else:
        item['repository'] = paper.source
    extra = []
    if paper.tldr:
        extra.append(f'Abstract translation: {paper.tldr}')
    if paper.score is not None:
        extra.append(f'Relevance score: {paper.score:.4f}')
    if paper.affiliations:
        extra.append('Affiliations: ' + ', '.join(paper.affiliations))
    if extra:
        item['extra'] = '\n'.join(extra)
    return item, identifiers


def save_papers(papers, config):
    collection = config.get("save_collection")
    if not collection or not papers:
        return
    top_k = config.get("save_top_k", 10)
    if type(top_k) is not int or top_k < 0:
        raise ValueError("zotero.save_top_k must be a non-negative integer")
    receipts = []
    try:
        zot = zotero.Zotero(config.user_id, 'user', config.api_key)
        key = _ensure_collection(zot, collection)
        existing = _existing_papers(zot, key)
        template = zot.item_template('preprint')
        for paper in papers[:top_k]:
            try:
                item, identifiers = _paper_to_item(paper, template, key)
                identifiers = [canonical_identifier(i) for i in identifiers]
                parent = next((existing[i] for i in identifiers if i in existing), None)
                if parent is None:
                    item['key'] = stable_item_key(f'{config.user_id}:{paper.source}:{identifiers[0]}')
                    item['version'] = 0
                    parent_key = _created_key(zot.create_items([item]))
                    parent = zot.item(parent_key)
                else:
                    parent_key = parent['key']
                    if key not in parent['data'].get('collections', []):
                        response = zot.addto_collection(key, parent)
                        response.raise_for_status()
                        parent = zot.item(parent_key)
                for identifier in identifiers:
                    existing[identifier] = parent
                receipts.append({'url': paper.url, 'parent_key': parent_key, 'status': 'saved'})
            except Exception as exc:
                receipts.append({'url': paper.url, 'status': 'failed', 'error': type(exc).__name__})
                logger.warning(f"Failed to sync paper {paper.url}: {type(exc).__name__}")
    except Exception as exc:
        receipts.append({'status': 'failed', 'error': type(exc).__name__})
        logger.error(f"Failed to sync Zotero metadata: {type(exc).__name__}")
    write_results('zotero-results.json', receipts)
