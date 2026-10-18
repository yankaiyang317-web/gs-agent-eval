"""Download immutable public benchmark assets and verify every extracted file."""
import argparse,hashlib,json,shutil,tarfile,urllib.request
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify_files(root,members):
    for name,expected in members.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()):raise ValueError('UNSAFE_MANIFEST_PATH')
        if not path.is_file() or sha(path)!=expected:raise ValueError('MISSING_OR_CHANGED_DATA: '+name)

def extract_verified(archive,root,members):
    root=Path(root).resolve()
    with tarfile.open(archive,'r:gz') as tf:
        files=[m for m in tf.getmembers() if m.isfile()]
        if any(not (m.isfile() or m.isdir()) for m in tf.getmembers()):raise ValueError('SPECIAL_ARCHIVE_MEMBER_REJECTED')
        if len(files)!=len(members) or {m.name for m in files}!=set(members):raise ValueError('ARCHIVE_MEMBER_SET_MISMATCH')
        for member in tf.getmembers():
            target=(root/member.name).resolve()
            if not target.is_relative_to(root):raise ValueError('UNSAFE_ARCHIVE_PATH')
        # Refuse to overwrite changed user files; correct existing files are reused.
        for m in files:
            target=root/m.name
            if target.exists() and (not target.is_file() or sha(target)!=members[m.name]):raise ValueError('EXISTING_DATA_DIFFERS: '+str(target))
        for m in files:
            target=root/m.name
            if target.exists():continue
            target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_name(target.name+'.download-tmp')
            with tf.extractfile(m) as src,temporary.open('wb') as dst:shutil.copyfileobj(src,dst,4*1024*1024)
            if sha(temporary)!=members[m.name]:raise ValueError('EXTRACTED_HASH_MISMATCH: '+m.name)
            temporary.replace(target)
    verify_files(root,members)

def download(url,target,expected):
    if target.is_file():
        if sha(target)!=expected:raise ValueError('CACHED_ARCHIVE_HASH_MISMATCH: '+str(target))
        return
    target.parent.mkdir(parents=True,exist_ok=True);part=target.with_name(target.name+'.part')
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'gs-agent-eval/0.1.1'}),timeout=120) as src,part.open('wb') as dst:
        shutil.copyfileobj(src,dst,4*1024*1024)
    if sha(part)!=expected:raise ValueError('DOWNLOAD_HASH_MISMATCH: '+str(target))
    part.replace(target)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--inputs',action='store_true',help='Also download all160 reconstruction RGB-D inputs');parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    root=args.root.resolve();manifest=json.loads((root/'datasets/jiudian-downloads.json').read_text(encoding='utf-8'));selected=[a for a in manifest['archives'] if not a.get('optional') or args.inputs]
    for asset in selected:
        if not args.verify_only:
            print('Downloading/verifying',asset['name'],flush=True);archive=root/'.downloads'/asset['name'];download(asset['url'],archive,asset['sha256']);extract_verified(archive,root,asset['members'])
        else:verify_files(root,asset['members'])
        print('Verified',asset['name'],len(asset['members']),'files',flush=True)
    print('Ready: low-quality scenes/jiudian_degraded.json; high-quality scenes/jiudian.json. Portable mounts: datasets/jiudian/navigation/scene.json and datasets/jiudian/reference/scene.json.',flush=True)

if __name__=='__main__':main()
