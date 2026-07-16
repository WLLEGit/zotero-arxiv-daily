from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper
import random
from datetime import datetime
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email
from openai import OpenAI
from tqdm import tqdm


def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} must contain only glob pattern strings.")

    return list(patterns)


class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)
    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        collections = {c['key']:c for c in collections}
        corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
        corpus = [c for c in corpus if c['data']['abstractNote'] != '']
        def get_collection_path(col_key:str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']
        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths
        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [CorpusPaper(
            title=c['data']['title'],
            abstract=c['data']['abstractNote'],
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths']
        ) for c in corpus]
    
    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")
            corpus = [
                c for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]
        if self.ignore_path_patterns:
            logger.info(f"Excluding zotero papers matching ignore_path: {self.ignore_path_patterns}")
            corpus = [
                c for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")
        return corpus

    @staticmethod
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

    @staticmethod
    def _existing_identifiers(zot, collection_key: str) -> set[str]:
        """Collect archiveIDs and urls already present in the collection, for de-duplication."""
        items = zot.everything(zot.collection_items_top(collection_key))
        identifiers: set[str] = set()
        for it in items:
            data = it.get('data', {})
            for field in ('archiveID', 'url'):
                if data.get(field):
                    identifiers.add(data[field])
        return identifiers

    @staticmethod
    def _split_creator(name: str) -> dict:
        parts = name.strip().split()
        if len(parts) >= 2:
            return {'creatorType': 'author', 'firstName': ' '.join(parts[:-1]), 'lastName': parts[-1]}
        return {'creatorType': 'author', 'name': name}

    def _paper_to_item(self, paper, template: dict, collection_key: str) -> tuple[dict, list[str]]:
        item = dict(template)
        item['title'] = paper.title
        item['abstractNote'] = paper.abstract or ''
        item['url'] = paper.url or ''
        item['creators'] = [self._split_creator(a) for a in (paper.authors or [])]
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
            extra.append(f'TLDR: {paper.tldr}')
        if paper.score is not None:
            extra.append(f'Relevance score: {paper.score:.4f}')
        if paper.affiliations:
            extra.append('Affiliations: ' + ', '.join(paper.affiliations))
        if extra:
            item['extra'] = '\n'.join(extra)
        return item, identifiers

    def save_to_zotero(self, papers: list) -> None:
        save_collection = self.config.zotero.get("save_collection", None)
        if not save_collection or not papers:
            return
        papers = papers[:self.config.zotero.get("save_top_k", 10)]  # papers arrive sorted by relevance, descending
        try:
            zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
            collection_key = self._ensure_collection(zot, save_collection)
            existing = self._existing_identifiers(zot, collection_key)
            template = zot.item_template('preprint')
            items = []
            for p in papers:
                item, identifiers = self._paper_to_item(p, template, collection_key)
                if any(i in existing for i in identifiers):
                    continue
                items.append(item)
                existing.update(identifiers)
            if not items:
                logger.info(f"All {len(papers)} papers already exist in '{save_collection}', nothing to add")
                return
            created = 0
            for i in range(0, len(items), 50):  # Zotero write API accepts up to 50 items per request
                resp = zot.create_items(items[i:i + 50])
                created += len(resp.get('successful', {}))
                if resp.get('failed'):
                    logger.warning(f"Failed to add {len(resp['failed'])} items to Zotero: {resp['failed']}")
            logger.info(f"Added {created} papers to Zotero collection '{save_collection}'")
        except Exception as e:
            logger.error(f"Failed to save papers to Zotero: {e}")

    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error(f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
            return
        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")
        reranked_papers = []
        if len(all_papers) > 0:
            logger.info("Reranking papers...")
            reranked_papers = self.reranker.rerank(all_papers, corpus)
            reranked_papers = reranked_papers[:self.config.executor.max_paper_num]
            logger.info("Generating TLDR and affiliations...")
            for p in tqdm(reranked_papers):
                p.generate_tldr(self.openai_client, self.config.llm)
                p.generate_affiliations(self.openai_client, self.config.llm)
            self.save_to_zotero(reranked_papers)
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return
        logger.info("Sending email...")
        email_content = render_email(reranked_papers)
        send_email(self.config, email_content)
        logger.info("Email sent successfully")
