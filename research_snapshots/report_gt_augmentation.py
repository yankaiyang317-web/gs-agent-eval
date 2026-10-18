from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gs_eval.storage import read_json,atomic_json,digest
from gs_eval.evaluation import rank_corr

root=Path('/app');out=root/'outputs/evaluation/gt_augmentation_validation_20261016';s=read_json(out/'summary.json');protocol=read_json(out/'protocol.json');results=read_json(out/'all_results.json');rows=s['groups'];names={'replay_only':'仅旧图继续重建','clip_1':'第一段81帧','clip_2':'第二段81帧','clip_3':'第三段81帧','repeat_control':'首帧重复81次'}
with (out/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
scored=[r for r in rows if r['B'] is not None]
corr={'B_vs_global_RGB_net':rank_corr([r['B'] for r in scored],[r['all_psnr_db_net_vs_replay_mean'] for r in scored]),'B_vs_weak_region_RGB_net':rank_corr([r['B'] for r in scored],[r['baseline_weak_psnr_db_net_vs_replay_mean'] for r in scored]),'B_vs_weak_region_depth_net':rank_corr([r['B'] for r in scored],[r['baseline_weak_depth_error_reduction_net_vs_replay_mean'] for r in scored]),'interpretation':'4 groups from one scene; descriptive ranks only, not significance or general validity'};atomic_json(out/'B_gain_diagnostics.json',corr)
fig,axes=plt.subplots(1,2,figsize=(11,4.5));x=np.arange(len(rows));colors=['#64748b','#2563eb','#f59e0b','#10b981','#ef4444'];labels=['Replay only','Clip 1','Clip 2','Clip 3','Repeated view']
for ax,area in zip(axes,['all','baseline_weak']):
    y=[r[f'{area}_psnr_db_mean'] for r in rows];err=[r[f'{area}_psnr_db_std'] for r in rows];ax.bar(x,y,yerr=err,color=colors,capsize=4);ax.set_xticks(x,labels,rotation=20);ax.axhline(0,color='black',linewidth=.8);ax.set(ylabel='PSNR gain vs original low (dB)',title='All held-out pixels' if area=='all' else 'Fixed baseline-weak held-out pixels')
fig.suptitle('GT RGB-D supplementation + old-view replay; mean +/- SD over 2 seeds');fig.tight_layout();fig.savefig(out/'gt_augmentation_gains.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.4))
for r,label,col in zip(rows,labels,colors):
    if r['B'] is None:continue
    axes[0].scatter(r['B'],r['baseline_weak_psnr_db_net_vs_replay_mean'],color=col);axes[0].annotate(label,(r['B'],r['baseline_weak_psnr_db_net_vs_replay_mean']),xytext=(4,5),textcoords='offset points')
    axes[1].scatter(r['B'],100*r['baseline_weak_accurate_depth_fraction_gain_net_vs_replay_mean'],color=col);axes[1].annotate(label,(r['B'],100*r['baseline_weak_accurate_depth_fraction_gain_net_vs_replay_mean']),xytext=(4,5),textcoords='offset points')
for ax in axes:ax.axhline(0,color='grey',linestyle='--');ax.set_xlabel('Frozen full-input RGBD+normal B')
axes[0].set(ylabel='Weak-region PSNR net gain vs replay only (dB)',title='Appearance');axes[1].set(ylabel='Weak-region accurate-depth net gain (percentage points)',title='Depth agreement');fig.tight_layout();fig.savefig(out/'B_vs_augmentation_net_gain.png',dpi=160)
lines=['# GT 补拍视角价值：补几何与旧图回放验证','',
'目标：验证轨迹观察的薄弱区域，在实际加入新观测后能否改善；将固定B分数与实际收益作对照。GT是高质量PLY生成的pseudo-GT，不是实测物理真值。','',
f"实验完成5组×2种子，共10个独立模型。每组同一low初始化、1000步、320×240。全部组共用从128个旧观测中选取的至多20000个RGB-D种子Gaussian，补拍组再从对应81帧中补入至多20000个新Gaussian。点按固定0.04体素去重，重复帧不会重复添加相同点。训练优化全部Gaussian参数，不做后续split/prune。",'',
'新帧采用GT在原探索位姿的RGB-D，提供颜色和几何监督；补拍组50%更新回放旧图，50%更新使用新图。仅旧图组100%使用旧图，训练步数相同。所有参数在查看结果前固定，不挑最佳checkpoint。', '',
f"独立测试使用原始重建留出的32张相机位姿，并排除与任一补拍位姿近似重合的视角，最终 {s['heldout_views']} 个有效测试视角；这些视角未参加初始low重建、完整输入可靠性校准或本次补充训练。薄弱像素区域由更新前low在这些测试视角的RGB/深度误差固定定义，对全部组保持相同。",'',
'|组别|冻结B|全域测试PSNR增量 dB|薄弱区域PSNR增量 dB|薄弱区域深度误差减少|', '|---|---:|---:|---:|---:|']
for r in rows:
    b='—' if r['B'] is None else f"{r['B']:.4f}"
    lines.append(f"|{names[r['group']]}|{b}|{r['all_psnr_db_mean']:+.3f} ± {r['all_psnr_db_std']:.3f}|{r['baseline_weak_psnr_db_mean']:+.3f} ± {r['baseline_weak_psnr_db_std']:.3f}|{r['baseline_weak_depth_error_reduction_mean']:+.5f}|")
lines+=['','上表均相对未更新的low。正PSNR增量、正深度误差减少表示改善；±为两个种子的样本标准差，不是置信区间。','',
'|补拍组|全域PSNR净增益，相对仅旧图|薄弱区域PSNR净增益，相对仅旧图|薄弱区域准确深度覆盖净增益|','|---|---:|---:|---:|']
for r in scored:lines.append(f"|{names[r['group']]}|{r['all_psnr_db_net_vs_replay_mean']:+.3f} dB|{r['baseline_weak_psnr_db_net_vs_replay_mean']:+.3f} dB|{100*r['baseline_weak_accurate_depth_fraction_gain_net_vs_replay_mean']:+.2f} 个百分点|")
lines+=['',
'本轮第二段在薄弱区域的RGB与平均深度误差上提供了超过仅旧图重建的净收益；第一段、第三段以及重复组的薄弱区域PSNR净收益为负。重复组B仍较高，说明固定逐帧均分对冗余的惩罚不足。不同质量项也有取舍：第二段降低了平均深度误差，但其准确深度覆盖增量低于仅旧图；第三段则相反。因此不把任意单项改善解释为所有重建质量都改善。','',
'净增益按相同种子配对扣除仅旧图继续重建的收益。准确深度覆盖指reference有效像素中low alpha≥0.7且相对深度误差≤0.1的比例，它是渲染深度一致性，不是网格完整度或F-score。','',
f"B与薄弱区域RGB净增益的描述性Spearman={corr['B_vs_weak_region_RGB_net']}；与薄弱区域深度误差减少净增益的Spearman={corr['B_vs_weak_region_depth_net']}。只有4组且来自同一场景/连续轨迹，不能据此宣布B已通过泛化验证或进行显著性推断。",'',
'边界：这是GT RGB-D条件下的补拍验证，RGB-only补拍和几何视频大模型增强没有在本实验中验证。模型插入采用已知GT深度，并不代表仅靠视频就能获得相同几何。新点数量随有效新内容变化，不是等Gaussian数量比较，而是同一补图算法/相同最大新增点预算。仅旧图也可恢复一部分缺陷，因此轨迹价值应重点查看净增益。','',
'该实验修正了前一轮仅固定拓扑RGB微调、无旧图回放的不足，但不是完整复现2D GauSS-MI。每组一个包含全部81帧的批量训练任务，并未为每一帧单独重建，不能由此推出单帧边际价值。','',
'模型保存在 machine25 独立备份 outputs/evaluation/gt_augmentation_validation_20261016/<组别>_seed<种子>/augmented.ply。原模型、冻结可靠性状态和原代码98个基线文件哈希均未变。']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
atomic_json(out/'artifact_receipt.json',{'protocol_fingerprint':protocol['fingerprint'],'files':{p.name:digest(p) for p in out.glob('*') if p.is_file() and p.name!='artifact_receipt.json'},'models':{str(p.relative_to(out)):digest(p) for p in out.glob('*/augmented.ply')},'report_script_sha256':digest(Path(__file__))})
print('\n'.join(lines),flush=True)
