# 当前冻结内容和版本规则

代码版本：gs-agent-eval 0.1.0。基准：jiudian_degraded_full_inputs_v1。

- low/high/collision资产SHA256；同一世界坐标、Y-up及已核实identity Sim(3)。
- 相机FOV、81记录帧预算、碰撞胶囊radius=.25/body=.9/step=.02。
- A：原固定112参考视角产生的24087体素，voxel=.2、stride=2、近远范围/ROI/遮挡容差固定。该域是部分参考域，并非完整物理场景。
- B：128个实际重建训练输入，排除32留出帧；RGBD+深度法线残差，逐Gaussian合成贡献更新；lambda_evidence=.01、lambda_loss=10、normal_weight=.15、clip=8。p/g/支持量以不可变npz保存。
- 评分320×240，按整图像素归一化；B是固定快照均分，不处理帧间边际收益衰减，纯缺失区域有盲点。
- 缺陷加权覆盖仍使用原固定defect_weights；它与新的RGBDN可靠性是不同项，不能假称已重新校准为同一状态。

完整文件绑定见benchmarks JSON，哈希不符直接拒绝。路径迁移不改变协议指纹；真实资产、状态或规则变化会改变指纹。compare另检查评测实现指纹，不能混合新旧程序分数。

Agent代码变化：只更新agent_version/method_id，继续用本基准。修改场景、碰撞规则、p/g、参考域或公式：新建benchmark_id，例如full_inputs_v2；不可覆盖v1。修正程序bug：新评测器版本并做回归，必要时重算全部方法，不只重算新Agent。

本版可作为固定研究原型用于迭代对照；不把“冻结可复现”解释为“B已通过多场景科学验证”。不要把新旧校准B的数值大小直接当方法改善。
