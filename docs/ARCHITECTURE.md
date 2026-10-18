# 分离边界

建议独立Git仓库 `gs-agent-eval`，不是继续复制整个Agent项目。Agent仓库可在README中链接本项目，或以后增加只调用评测CLI的可选脚本；不用把评分器导入Agent运行进程。

Agent → trajectory.jsonl + campaign.json + run_metadata.json → 独立评测器 → summary.json + per_frame.csv + 报告。

每行轨迹包含`frame`、4×4 OpenCV `camera_to_world`、`intrinsics`的fx/fy/cx/cy/width/height，可选event和source_frame。世界竖直轴本基准为+Y。图像源文件路径不是评分必需输入，不会读取Agent帧作为GT。移动/旋转中间帧和重复帧都占预算，不重采样、不补帧。

原导航已输出trajectory/campaign，因而日常评分不需要改导航逻辑。为了以后可核实实验条件，建议只补充运行元数据：Agent版本、模型/提示词版本、随机种子、场景与碰撞资产哈希、相机胶囊参数。当前已有旧轨迹缺乏完整元数据时会明确标未核实，不伪造其来源；场景名缺失时使用父目录标签并记录该弱验证来源。

冻结数据和GT只在离线评测中可用。运行Agent时不要挂载GT、可靠性或报告。相机/PLY读取和坐标变换的最小代码已放入`gs_eval/_vendor`固定版本，不随Agent仓库更新。

核心legacy研究API仍位于gs_eval；日常使用`gs-agent-eval`/`python -m gs_agent_eval`，避免误用旧的24视角校准流程。最新完整输入状态通过profile绑定；score不会训练、更新p或调用视频生成模型。

research_snapshots是原始场景专用脚本，保留/app和machine25路径，需要原归档环境才能复建研究实验。它们已与日常通用评分入口分离，不能当作通用训练器。原完整备份`/machine25_data1/gs-agent-offline-eval-20261016`继续保留。
