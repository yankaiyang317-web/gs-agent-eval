from pathlib import Path
import json,hashlib,tarfile,shutil,xml.etree.ElementTree as ET,subprocess
root=Path('/machine25_data1/gs-agent-eval');legacy=Path('/machine25_data1/gs-agent-offline-eval-20261016')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4194304),b''):h.update(block)
    return h.hexdigest()
receipt=json.loads((root/'outputs/parity_receipt.json').read_text());assert receipt['passed'] and receipt['agent_modules_blocked']
xml=ET.parse(root/'outputs/cpu_tests.xml').getroot();suites=list(xml) if xml.tag=='testsuites' else [xml];counts={key:sum(int(s.get(key,0)) for s in suites) for key in ['tests','failures','errors']};assert counts['failures']==counts['errors']==0
baseline=json.loads((legacy/'BASELINE_SNAPSHOT.json').read_text());changed=[key for key,value in baseline['files'].items() if not (Path(baseline['source'])/key).is_file() or sha(Path(baseline['source'])/key)!=value]
image=json.loads(subprocess.check_output(['docker','image','inspect','gs-agent-offline-eval:20261016']))[0]
(root/'provenance/runtime_image.json').write_text(json.dumps({'image_id':image['Id'],'repo_tags':image['RepoTags'],'usage':'Existing tested CUDA runtime. Preserve/export/push this image separately if migrating servers; the code archive does not include the image.'},indent=2))
checks=root/'provenance/release_validation';checks.mkdir(exist_ok=True)
for name in ['parity_receipt.json','runtime.json','cpu_tests.xml','cpu_tests.log']:
    shutil.copy2(root/'outputs'/name,checks/name)
with (root/'.gitignore').open('a') as f:f.write('\nreleases/\n')
source=[]
for path in sorted(root.rglob('*')):
    rel=path.relative_to(root)
    if not path.is_file() or rel.parts[0] in ['outputs','benchmark_data','releases','dist','build'] or any(part=='__pycache__' or part.endswith('.egg-info') for part in rel.parts):continue
    if path.suffix=='.pyc' or path.name=='RELEASE_MANIFEST.json':continue
    source.append(path)
states=[p for p in (root/'benchmark_data').rglob('*') if p.is_file()]
manifest={'release':'gs-agent-eval-v0.1.0','frozen_on':'2026-10-18','benchmark':'jiudian_degraded_full_inputs_v1','source_sha256':{str(p.relative_to(root)):sha(p) for p in source},'state_sha256':{str(p.relative_to(root)):sha(p) for p in states},'original_agent_code_files_verified_unchanged':len(baseline['files'])-len(changed),'original_agent_files_changed_since_20261016_baseline':changed,'original_agent_modified_by_this_extraction':False,'validation':{'cpu':counts,'parity':receipt},'git_status':'Prepared independent project; no GitHub repository created and no push performed. Versioned here by immutable source/state hashes.','large_assets':'Source low/high/collision assets are not copied. Their hashes and locations are frozen in benchmarks JSON.','scientific_status':'Frozen reproducible pilot, not a claim of broadly validated B predictive accuracy.'}
(root/'RELEASE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));source.append(root/'RELEASE_MANIFEST.json')
release=root/'releases/v0.1.0';release.mkdir(parents=True,exist_ok=False)
for name,paths in [('gs-agent-eval-v0.1.0-code.tar.gz',source),('jiudian_degraded_full_inputs_v1-state.tar.gz',states)]:
    with tarfile.open(release/name,'w:gz') as tf:
        for p in paths:tf.add(p,arcname='gs-agent-eval/'+str(p.relative_to(root)),recursive=False)
wheel=next((root/'dist').glob('gs_agent_eval-*.whl'));shutil.copy2(wheel,release/wheel.name)
shutil.copy2(root/'RELEASE_MANIFEST.json',release/'RELEASE_MANIFEST.json')
(release/'SHA256SUMS.txt').write_text('\n'.join(sha(p)+'  '+p.name for p in sorted(release.iterdir()) if p.is_file())+'\n')
print(json.dumps({'release_dir':str(release),'cpu':counts,'parity':receipt['checks'],'archives':{p.name:p.stat().st_size for p in release.iterdir()}},ensure_ascii=False,indent=2),flush=True)
