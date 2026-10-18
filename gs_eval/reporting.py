import csv,json
from pathlib import Path
import numpy as np
from .storage import atomic_json,write_jsonl

def csv_rows(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        if rows:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def curve(path,rows,fields,ylabel):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4));x=[r['frame'] for r in rows]
    for field in fields:ax.plot(x,[np.nan if r.get(field) is None else r[field] for r in rows],label=field)
    ax.set(xlabel='Recorded frame',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)

def image_pair(path,high,low):
    from PIL import Image
    im=np.concatenate([high['rgb'],low['rgb']],axis=1);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(np.round(np.clip(im,0,1)*255).astype('uint8')).save(path)

def heatmap(path,values,mask,label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(figsize=(6,4));a=np.where(mask,values,np.nan);im=ax.imshow(a,cmap='magma');fig.colorbar(im,ax=ax);ax.set_title(label);ax.set_axis_off();fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)

def report(out,summary,rows,selected):
    out=Path(out);atomic_json(out/'summary.json',summary);csv_rows(out/'per_frame.csv',rows);write_jsonl(out/'selected_trajectory.jsonl',selected);csv_rows(out/'reference_residuals.csv',[{k:r[k] for k in ['frame','gt_rgb_residual','gt_relative_depth_residual','valid_reference_fraction']} for r in rows]);curve(out/'coverage_curves.png',rows,['cumulative_surface_coverage','cumulative_defect_coverage'],'Coverage (separate measures)');curve(out/'view_utility_curve.png',rows,['I_normalized','I_constant_p05'],'Fixed snapshot view utility');curve(out/'gt_rgb_residual_curve.png',rows,['gt_rgb_residual'],'GT RGB residual');curve(out/'gt_relative_depth_residual_curve.png',rows,['gt_relative_depth_residual'],'GT relative depth residual')
    a=summary['metric_families']['A_surface_coverage'];b=summary['metric_families']['B_view_value'];text=f'''# GSAgent 离线轨迹评价

状态：{summary['status']}。场景：{summary['scene_id']}。方法：{summary['method_id']}。
实际记录帧：{summary['evaluated_frames']} / 81；预算包含移动/旋转中间帧，重复帧不剔除。

## A 探索覆盖

| 项目 | 结果 |
| --- | --- |
| 参考表面覆盖 @81 | {a['primary']['surface_coverage_at_81']} |
| 覆盖 AUC @81 | {a['supporting']['surface_coverage_auc_81']} |

体素并集是固定 reference domain 的表面积近似，既不是完整物理真值，也不代表 Agent 已识别这些表面。

## B 视角价值

| 项目 | 结果 |
| --- | --- |
| 固定模型视角价值均值 | {b['primary']['frozen_gaussmi_mean']} |
| GT RGB 残差均值 | {b['observed_reference']['gt_rgb_residual_mean']} |
| GT 相对深度残差均值 | {b['observed_reference']['gt_relative_depth_residual_mean']} |
| 去重薄弱表面覆盖 @81 | {b['supporting']['defect_weighted_coverage_at_81']} |

该项是 paper inspired 3DGS scalar adaptation（frozen_gaussmi_3d_v1），不是完整复现 2D GauSS-MI，不继承其互信息保证。视角均分不是累计新信息，参考残差也不是实际修复收益。重复看同一缺陷可有高视角均分，但覆盖不会继续增加。

## 数据和有效性

- High PLY 是 pseudo GT；low/high 输入均保持只读。
- reference domain：`{json.dumps(summary['reference_domain'],ensure_ascii=False)}`
- 缓存/协议：`{summary['comparability']['reference_fingerprint']}`
- 可靠性状态：`{summary['diagnostics']['reliability_status']}`，适配方法仍需独立科学验证。
- 轨迹类型：`{summary['diagnostics']['trajectory_origin']}`。合成 probe 仅用于指标可行性检查，不能作为自主 Agent 成绩。
- 路径长度是相邻记录相机中心的 sampled path length，单位为 low 场景单位；没有真实任务时钟，不按视频 FPS 推算耗时。
- 传送/跳变：`{summary['trajectory_diagnostics']['jump_frames']}`。
- RGB/深度缺失不通过 high/low mask 交集隐藏；无有效 reference 的残差为 null。
- 本次图像示例索引：`{summary["diagnostics"].get("example_indices", [])}`；空列表表示未导出示例图。诊断渲染不参与Agent决策。
- 全部 A/B 项分别解释，不产生统一总分或混合排名。
'''
    (out/'report.md').write_text(text,encoding='utf-8')
