from pathlib import Path
import subprocess,os,sys
root=Path('/eval');p=root/'gs_eval/reporting.py';text=p.read_text();text=text.replace('- 图像示例规则：固定每20帧及最后一帧，另取有效预测分最高/最低帧（只用于诊断，不据此选择基准）。','- 本次图像示例索引：`{summary["diagnostics"].get("example_indices", [])}`；空列表表示未导出示例图。诊断渲染不参与Agent决策。');p.write_text(text)
out=root/'outputs'
def run(args,name,env=None,cwd=root):
    with (out/name).open('w') as f:r=subprocess.run(args,env=env,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:print((out/name).read_text(),flush=True);raise SystemExit(r.returncode)
    print('PASSED',name,flush=True)
run([sys.executable,'-m','pytest','-q','--junitxml=outputs/cpu_tests.xml'],'cpu_tests.log')
run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','-w','dist','.'],'wheel.log')
wheel=next((root/'dist').glob('gs_agent_eval-*.whl'));run([sys.executable,'-m','pip','install','--no-deps','--target','/tmp/eval-final',str(wheel)],'install.log')
p=root/'scripts/check_release.py';text=p.read_text().replace("root/'outputs/parity'/", "root/'outputs/parity_final'/");p.write_text(text)
env={**os.environ,'PYTHONPATH':'/tmp/eval-final','MPLCONFIGDIR':'/tmp/matplotlib'}
run([sys.executable,str(p)],'parity_final.log',env=env,cwd=Path('/tmp'))
print('FINAL_VALIDATION_COMPLETE',flush=True)
