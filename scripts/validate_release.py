from pathlib import Path
import subprocess,os,json,sys,importlib.metadata
root=Path('/eval');out=root/'outputs';out.mkdir(exist_ok=True)
def command(args,name,env=None,cwd=root):
    with (out/name).open('w') as log:
        r=subprocess.run(args,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:print((out/name).read_text(),flush=True);raise SystemExit(r.returncode)
    print('PASSED',name,flush=True)
command([sys.executable,'-m','pytest','-q','--junitxml=outputs/cpu_tests.xml'],'cpu_tests.log')
command([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','-w','dist','.'],'wheel.log')
wheel=next((root/'dist').glob('gs_agent_eval-*.whl'))
command([sys.executable,'-m','pip','install','--no-deps','--target','/tmp/eval-installed',str(wheel)],'install.log')
env={**os.environ,'PYTHONPATH':'/tmp/eval-installed','MPLCONFIGDIR':'/tmp/matplotlib'}
command([sys.executable,str(root/'scripts/check_release.py')],'parity.log',env=env,cwd=Path('/tmp'))
import torch
(out/'runtime.json').write_text(json.dumps({'python':sys.version,'torch':torch.__version__,'CUDA':torch.version.cuda,'GPU':torch.cuda.get_device_name(0),'installed_wheel':wheel.name,'gsplat':importlib.metadata.version('gsplat')},indent=2))
print('RELEASE_VALIDATION_COMPLETE',flush=True)
