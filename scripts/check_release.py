from pathlib import Path
import sys,json
import numpy as np
sys.modules['gs_env']=None;sys.modules['gs_mcp']=None
from gs_agent_eval.cli import load_benchmark,score
from gs_eval.storage import read_json,atomic_json
root=Path(__file__).resolve().parents[1];b=load_benchmark(root/'benchmarks/jiudian_degraded_full_inputs_v1.json');expected=read_json(root/'fixtures/expected_full_input_scores.json');checks=[]
for i in range(3):
    out=root/'outputs/publication_parity'/f'clip_{i:03d}'
    s=score(b,root/'fixtures'/f'clip_{i:03d}',out,method_id=f'baseline_clip_{i+1}',agent_version='frozen_baseline_20261016')
    old=read_json(root/'fixtures'/f'expected_clip_{i:03d}.json');a=s['metric_families']['A_surface_coverage'];oa=old['metric_families']['A_surface_coverage'];B=s['metric_families']['B_view_value']['primary']['frozen_gaussmi_mean'];exp=expected['scores'][i]['full_input_RGBDN_B']
    assert np.isclose(a['primary']['surface_coverage_at_81'],oa['primary']['surface_coverage_at_81'],atol=1e-10,rtol=0)
    assert np.isclose(a['supporting']['surface_coverage_auc_81'],oa['supporting']['surface_coverage_auc_81'],atol=1e-10,rtol=0)
    assert np.isclose(B,exp,atol=1e-6,rtol=0)
    checks.append({'clip':i+1,'coverage':a['primary']['surface_coverage_at_81'],'B':B,'expected_B':exp,'B_abs_error':abs(B-exp)})
assert sys.modules['gs_env'] is None and sys.modules['gs_mcp'] is None
atomic_json(root/'outputs/parity_receipt.json',{'passed':True,'agent_modules_blocked':True,'checks':checks,'B_tolerance':1e-6,'coverage_tolerance':1e-10})
print('STANDALONE_PARITY_PASSED',json.dumps(checks),flush=True)
