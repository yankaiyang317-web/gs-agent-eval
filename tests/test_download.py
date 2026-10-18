import importlib.util,tarfile,io
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('download_benchmark',Path(__file__).parents[1]/'scripts/download_benchmark.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def archive(path,entries):
    with tarfile.open(path,'w:gz') as tf:
        for name,data in entries.items():
            item=tarfile.TarInfo(name);item.size=len(data);tf.addfile(item,io.BytesIO(data))

def test_verified_extraction_and_reuse(tmp_path):
    p=tmp_path/'source';p.write_bytes(b'fixed');tar=tmp_path/'a.tar.gz';archive(tar,{'datasets/a':b'fixed'});members={'datasets/a':m.sha(p)};root=tmp_path/'out';root.mkdir();m.extract_verified(tar,root,members);m.extract_verified(tar,root,members);assert (root/'datasets/a').read_bytes()==b'fixed'

def test_hash_mismatch_rejected(tmp_path):
    tar=tmp_path/'a.tar.gz';archive(tar,{'a':b'bad'})
    with pytest.raises(ValueError,match='HASH_MISMATCH'):m.extract_verified(tar,tmp_path/'out',{'a':'0'*64})

def test_traversal_rejected(tmp_path):
    tar=tmp_path/'a.tar.gz';archive(tar,{'../outside':b'x'})
    with pytest.raises(ValueError,match='UNSAFE'):m.extract_verified(tar,tmp_path/'out',{'../outside':'unused'})

def test_modified_existing_file_not_overwritten(tmp_path):
    root=tmp_path/'out';root.mkdir();(root/'a').write_bytes(b'user data');tar=tmp_path/'a.tar.gz';archive(tar,{'a':b'fixed'})
    with pytest.raises(ValueError,match='EXISTING_DATA_DIFFERS'):m.extract_verified(tar,root,{'a':'0'*64})
    assert (root/'a').read_bytes()==b'user data'

def test_extra_member_rejected(tmp_path):
    tar=tmp_path/'a.tar.gz';archive(tar,{'a':b'fixed','extra':b'x'})
    with pytest.raises(ValueError,match='MEMBER_SET'):m.extract_verified(tar,tmp_path/'out',{'a':'0'*64})
