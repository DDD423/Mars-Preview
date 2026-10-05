"""Download only official Qwen model assets; pin revision and SHA-256."""
import os, sys, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path('artifacts/comparison-runtime').resolve()))
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
from huggingface_hub import HfApi, snapshot_download

repo = 'Qwen/Qwen2.5-0.5B-Instruct'
destination = Path('artifacts/baselines/Qwen2.5-0.5B-Instruct')
info = HfApi(token=False).model_info(repo)
print(json.dumps({'repo': repo, 'revision': info.sha}), flush=True)
snapshot_download(repo, revision=info.sha, local_dir=str(destination), token=False,
                  allow_patterns=['*.json', '*.safetensors', '*.txt', '*.tiktoken', '*.model', 'README.md', 'LICENSE'], max_workers=3)
files = {}
for path in destination.iterdir():
    if path.is_file():
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''): digest.update(block)
        files[path.name] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
manifest = {'repo': repo, 'revision': info.sha, 'source': 'https://huggingface.co/'+repo, 'files': files}
(destination/'DOWNLOAD_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest), flush=True)
