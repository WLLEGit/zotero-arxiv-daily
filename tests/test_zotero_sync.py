"""Metadata sync keeps existing user items and never creates report attachments."""
import copy
from types import SimpleNamespace
import pytest
from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily import zotero_sync


class Library:
    def __init__(self):
        self.store = {}
        self.lose_response = False

    def everything(self, items):
        return copy.deepcopy(items)

    def collections(self):
        return [{"key": "DAILY", "data": {"name": "DailyArxiv", "parentCollection": False}}]

    def collection_items_top(self, key):
        return [item for item in self.store.values() if key in item["data"].get("collections", [])]

    def items(self, **kwargs):
        return [item for item in self.store.values() if item["data"]["itemType"] == "preprint"]

    def item(self, key):
        return copy.deepcopy(self.store[key])

    def item_template(self, item_type, linkmode=None):
        return {"itemType": item_type, "tags": [], "collections": []}

    def create_items(self, items):
        successful, unchanged = {}, {}
        for index, data in enumerate(items):
            assert data["version"] == 0  # Zotero rejects key-based creates without this.
            key = data["key"]
            if key in self.store:
                unchanged[str(index)] = key
                continue
            item = {"key": key, "version": 1, "data": copy.deepcopy(data)}
            self.store[key] = item
            successful[str(index)] = item
            if self.lose_response:
                raise TimeoutError("Saved, but response lost")
        return {"successful": successful, "unchanged": unchanged, "failed": {}}

    def addto_collection(self, key, item):
        self.store[item["key"]]["data"]["collections"].append(key)
        return SimpleNamespace(raise_for_status=lambda: None)


@pytest.fixture
def library(config, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config.zotero.save_collection = 'DailyArxiv'
    client = Library()
    monkeypatch.setattr(zotero_sync.zotero, 'Zotero', lambda *args, **kwargs: client)
    return client


def papers(count):
    return [make_sample_paper(url=f'https://arxiv.org/abs/2610.{i:05d}v1') for i in range(count)]


def test_metadata_only_top_ten_and_repeated_runs(config, library):
    zotero_sync.save_papers(papers(15), config.zotero)
    zotero_sync.save_papers(papers(15), config.zotero)
    assert len(library.store) == 10
    assert all(i['data']['itemType'] == 'preprint' for i in library.store.values())


def test_existing_parent_and_legacy_note_are_preserved(config, library):
    original = {'key':'EXISTING','version':1,'data':{
        'itemType':'preprint','url':'https://arxiv.org/pdf/2610.00000v1.pdf',
        'collections':['DAILY'],'title':'User title','extra':'User metadata'}}
    note = {'key':'OLDNOTE','data':{'itemType':'note','parentItem':'EXISTING','note':'Old analysis'}}
    library.store = {'EXISTING':copy.deepcopy(original),'OLDNOTE':copy.deepcopy(note)}
    zotero_sync.save_papers(papers(1), config.zotero)
    assert library.store == {'EXISTING':original,'OLDNOTE':note}


def test_lost_create_response_does_not_duplicate_metadata(config, library):
    library.lose_response = True
    zotero_sync.save_papers(papers(1), config.zotero)
    library.lose_response = False
    paper = papers(1)[0];paper.url = paper.url.replace('v1','v2')
    zotero_sync.save_papers([paper], config.zotero)
    assert len(library.store) == 1
