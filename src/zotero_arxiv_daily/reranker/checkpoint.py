"""Local ranking evidence and reusable results; contains no API credentials."""
import gzip
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from ..protocol import Paper


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        os.chmod(temporary,0o600)
        json.dump(value,stream,ensure_ascii=False)
    temporary.replace(path)


class Checkpoint:
    def __init__(self,query,params,papers,limit):
        self.root=Path('outputs/rerank')
        self.context={'version':1,'query':query,'api_mode':params.get('api_mode','chat_completion'),
                      'generation_kwargs':params['generation_kwargs'],'limit':limit}
        self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        # Preserve the exact vector order, scores and extracted text for a ranking-only resume.
        path=self.root/'input.json.gz';temporary=path.with_name('input.'+uuid4().hex+'.tmp')
        with temporary.open('wb') as stream:
            os.chmod(temporary,0o600)
            with gzip.GzipFile(fileobj=stream,mode='wb') as compressed:
                compressed.write(json.dumps({**self.context,'papers':[asdict(p) for p in papers]},ensure_ascii=False).encode())
        temporary.replace(path)

    def key(self,kind,value):
        return kind+'-'+fingerprint({'context':self.context,'input':value})

    def read(self,key):
        path=self.root/(key+'.json')
        try:
            value=json.loads(path.read_text()) if path.exists() else None
            return value if isinstance(value,dict) else None
        except (ValueError,OSError):return None

    def write(self,key,value):
        write_json(self.root/(key+'.json'),value)

    def attempt(self,key,value):
        self.write(key+'-attempt-'+uuid4().hex,value)


def load_papers(path='outputs/rerank/input.json.gz'):
    """Read-only input for an explicitly requested ranking resume, never delivers anything."""
    with gzip.open(path,'rt',encoding='utf-8') as stream:value=json.load(stream)
    return [Paper(**row) for row in value['papers']]
