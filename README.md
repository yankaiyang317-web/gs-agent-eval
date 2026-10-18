# GSAgent Eval

Standalone offline evaluation for GSAgent trajectories, with a downloadable hotel baseline scene.

独立评测器 **v0.1.1**，与 [原 gs-agent 项目](https://github.com/yankaiyang317-web/gs-agent) 配合使用。当前基准 **jiudian_degraded_full_inputs_v1**：A衡量固定参考域的探索覆盖，B衡量固定模型薄弱区域的视角价值；两项分开，不合成总分。

![酒店低质量导航场景](docs/images/jiudian_baseline.jpg)

## 下载并使用

```bash
git clone https://github.com/yankaiyang317-web/gs-agent-eval.git
cd gs-agent-eval
python scripts/download_benchmark.py
```

代码放Git；场景和固定状态由脚本从 [v0.1.1 Release](https://github.com/yankaiyang317-web/gs-agent-eval/releases/tag/v0.1.1) 下载并逐文件校验。默认下载低质量PLY、碰撞GLB、高质量参考PLY和固定状态；`--inputs`额外下载全部160张合成RGB及对应辅助RGB-D。高质量酒店的原有碰撞 GLB 与运行配置由 [酒店运行补充包](https://github.com/yankaiyang317-web/gs-agent-eval/releases/tag/jiudian-runtime-v1) 提供，当前默认下载会一并获取。详见 [数据说明](datasets/DATA_CARD.md)。

导航仅使用 `datasets/jiudian/navigation/`；GT和评测状态只给独立离线评测器。原项目可使用 `scenes/jiudian_degraded.json`，或只挂载navigation目录并使用其中的scene.json。单独测试高质量酒店时使用 `scenes/jiudian.json`，或挂载reference目录并使用其中的scene.json；它复用原酒店配置与原碰撞网格，仅将资源路径改成可下载的相对路径。

```bash
# 在CUDA环境安装；测试过的版本见requirements-tested.txt。
pip install .
python -m gs_agent_eval verify --benchmark benchmarks/jiudian_degraded_full_inputs_v1.json
python -m gs_agent_eval score \
  --benchmark benchmarks/jiudian_degraded_full_inputs_v1.json \
  --run-dir /absolute/agent-run --out outputs/agent_v2/run_001 \
  --method-id agent_v2 --agent-version YOUR_AGENT_COMMIT
```

完整步骤见 [与原Agent配合使用](docs/USAGE_WITH_AGENT.md)。评分需要CUDA/PyTorch/gsplat；CPU测试不需要GPU。`Dockerfile.machine25`沿用已有测试运行时，公共`Dockerfile`提供可重建配方（该配方的全新镜像构建未在此版本验证）。

## 以后迭代Agent

修改Agent并记录commit/模型/提示词版本 → 相同场景、起点、相机和碰撞条件下生成轨迹 → 使用同一基准评分 → 每次输出到新目录。日常不必重新校准或补图训练。81帧包含视频记录的中间位姿和重复帧，连续三段不是三次独立实验。

```bash
python -m pytest -q
python -m gs_agent_eval compare --inputs outputs/a/summary.json outputs/b/summary.json --out outputs/comparison
```

原Agent独立维护；本包内部固定了PLY读取与坐标变换，不依赖Agent运行模块。大型数据/视频/模型均已Git忽略；冻结状态、原始场景SHA256和参数由profile绑定，数据变化会拒绝评分。

## 目录

- `gs_agent_eval/`, `gs_eval/`：独立CLI和评分核心。
- `benchmarks/`, `scenes/`, `datasets/jiudian-downloads.json`：冻结协议、导航入口和数据下载清单。
- `templates/`, `tests/`, `fixtures/`：运行元数据、CPU测试和历史轨迹回归样本。
- `provenance/`, `research_snapshots/`：来源、历史实验报告和原场景专用研究脚本。
- `datasets/jiudian/`, `benchmark_data/`, `outputs/`：下载数据、冻结状态和新结果，均不进入Git。

本基准是可复现的单场景研究原型。B仍为实验性代理，不是实际修复量、概率或完整2D GauSS-MI复现；有帧间冗余和纯缺失区域盲点。法线由深度差分得到，参考域是部分域。GT补图已有小规模验证，生成视频增强尚未验证。详见 [冻结规则](docs/FROZEN_PROTOCOL.md)、[修改清单](docs/CHANGES.md)、[实验说明](docs/EXPERIMENTS.md)。
