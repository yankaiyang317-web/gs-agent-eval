# 与原 gs-agent 配合使用

本仓库：[yankaiyang317-web/gs-agent-eval](https://github.com/yankaiyang317-web/gs-agent-eval)。原导航项目：[gs-agent](https://github.com/yankaiyang317-web/gs-agent)。两个项目各自维护版本，通过轨迹文件连接。

## 1. 一次下载三个场景的高、低质量版本

```bash
git clone https://github.com/yankaiyang317-web/gs-agent-eval.git
cd gs-agent-eval
python scripts/download_benchmark.py
```

默认下载 jiudian、guju、laojie 三个场景各自的高、低质量 PLY、碰撞 GLB、运行配置，以及各自的固定 A/B 评分状态。每一对高低质量场景在同一个数据包中，默认一起下载。若要复建可靠性状态或做补图训练，加 `--inputs` 下载三个场景各 160 张合成 RGB、辅助深度、逐帧位姿和训练选帧记录。原始实拍照片及 COLMAP 未提供。大文件来自同一个 GitHub Release，代码 Git 历史不承载它们。

```bash
python scripts/download_benchmark.py --inputs
python scripts/download_benchmark.py --verify-only
```

## 2. 用原项目运行 Agent

可向原项目支持的stdio入口指定本仓库的manifest：

```bash
python -m gs_mcp.server --scene-manifest /absolute/gs-agent-eval/scenes/jiudian_degraded.json
```

如果使用持久HTTP服务，把`datasets/jiudian/navigation/`单独只读挂载进导航容器，例如`/data/jiudian`，并将`navigation/scene.json`注册为服务的`jiudian_degraded`场景manifest。不要给导航进程挂载reference、benchmark_data或reconstruction_inputs目录。具体HTTP注册方式以所用gs-agent版本为准。

场景名固定`jiudian_degraded`，world up为+Y；运行时开始配置campaign，使用同一起点、FOV和碰撞参数。原始记录文件`trajectory.jsonl`包括每个视频记录帧的C2W和内参，不能只保存MP4。运行元数据参照`templates/run_metadata.json`，记录真实Agent版本/模型/提示词/随机种子与实际环境；不要把模板当成已经核实的记录。

## 3. 离线评分

在有CUDA的评测环境安装`pip install .`及测试依赖；PyTorch/gsplat版本见`requirements-tested.txt`。已有machine25运行时可沿用Dockerfile.machine25；公共Dockerfile提供可重建的CUDA环境配方。

```bash
python -m gs_agent_eval verify --benchmark benchmarks/jiudian_degraded_full_inputs_v1.json
python -m gs_agent_eval score \
  --benchmark benchmarks/jiudian_degraded_full_inputs_v1.json \
  --run-dir /absolute/agent-run \
  --out outputs/agent_v2/run_001 \
  --method-id agent_v2 --agent-version YOUR_AGENT_COMMIT
```

下载后的默认路径可直接使用。数据放别处时用`--asset-root /absolute/datasets/jiudian`，保留navigation/reference子目录；冻结状态仍在`benchmark_data/jiudian_degraded_full_inputs_v1`。迁移路径不改变基准指纹，数据字节变化则拒绝。

每次用新输出目录。81记录帧为一段；连续三段用`--clip-index 0/1/2`分别评分。日常比较复用同一个冻结状态，不重新校准、不必每次补图训练。A/B分别报告，不合成总分。

```bash
python -m gs_agent_eval compare \
  --inputs outputs/agent_v1/run_001/summary.json outputs/agent_v2/run_001/summary.json \
  --out outputs/comparison
```

注意独立重复、同起点和同运行条件。不同评测实现/状态/预算不能直接混比。

## 选择高质量或退化版场景进行测试

两个场景默认都已下载，在原 gs-agent 的运行环境中选择配置：

```bash
# 高质量酒店：原 Agent 默认探索的场景
python -m gs_mcp.server --scene-manifest /absolute/gs-agent-eval/scenes/jiudian.json

# 低质量酒店：本仓库的低质量探索评测场景
python -m gs_mcp.server --scene-manifest /absolute/gs-agent-eval/scenes/jiudian_degraded.json
```

高质量起点 `[0,0,0]`，低质量起点 `[0,0,-0.15]`；各自的原有相机、碰撞参数与网格均保留。只挂载单个场景目录时，高质量使用 `datasets/jiudian/reference/scene.json`，低质量使用 `datasets/jiudian/navigation/scene.json`。两个都可用于运行测试；当前冻结评分协议专门对应低质量 `jiudian_degraded` 的探索轨迹，不能直接将高质量运行分数混进该基准。

故居、老街新版分别提供高质量的 `scenes/guju.json`、`scenes/laojie.json`，以及低质量的 `scenes/guju_degraded_overlap.json`、`scenes/laojie_degraded_overlap.json`。各自的 `datasets/<scene>/reference/scene.json` 与 `navigation/scene_overlap.json` 可单独挂载。低质量版默认使用各自高质量场景的固定碰撞网格，避免缺陷使 Agent 被卡住；`navigation/scene_overlap_sparse_collision.json` 可试验从新版低质量 PLY 生成的碰撞体，但路线检查发现它会额外阻挡部分视角。评分时使用对应的新版 benchmark ID；旧版 81 帧轨迹和补建结果不可用于新版模型。

```bash
python -m gs_agent_eval verify --benchmark benchmarks/guju_degraded_overlap_20261127.json
python -m gs_agent_eval verify --benchmark benchmarks/laojie_degraded_overlap_20261127.json
python -m gs_agent_eval score --benchmark benchmarks/guju_degraded_overlap_20261127.json --run-dir /absolute/guju_overlap_run --out outputs/guju_method_a
python -m gs_agent_eval score --benchmark benchmarks/laojie_degraded_overlap_20261127.json --run-dir /absolute/laojie_overlap_run --out outputs/laojie_method_a
```

两个场景的评测状态独立，不能把一个场景的轨迹送进另一个场景的基准。新版输入、缺损、碰撞和状态限制见[新版报告](OVERLAP_BASELINES_20261127.md)；[旧试验报告](NEW_SCENE_PILOT_20261125.md)只适用于历史低质量 PLY。
