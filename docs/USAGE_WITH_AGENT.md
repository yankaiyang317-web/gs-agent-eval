# 与原 gs-agent 配合使用

本仓库：[yankaiyang317-web/gs-agent-eval](https://github.com/yankaiyang317-web/gs-agent-eval)。原导航项目：[gs-agent](https://github.com/yankaiyang317-web/gs-agent)。两个项目各自维护版本，通过轨迹文件连接。

## 1. 一次下载三个场景的高、低质量版本

```bash
git clone https://github.com/yankaiyang317-web/gs-agent-eval.git
cd gs-agent-eval
python scripts/download_benchmark.py
```

默认下载 jiudian、guju、laojie 三个场景各自的高、低质量 PLY、碰撞 GLB、运行配置，以及酒店冻结评测状态。每一对高低质量场景在同一个数据包中，默认一起下载。若要重建酒店校准或做补图训练，再加 `--inputs` 下载酒店的160张合成RGB及对应辅助RGB-D。数据来自GitHub Release，代码Git历史不承载大文件。guju和laojie的高清合成重建输入尚未发布，训练选择和结果元数据已在场景包中。

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

新增的 guju、laojie 同样分别提供 `scenes/guju.json`、`scenes/guju_degraded.json`、`scenes/laojie.json`、`scenes/laojie_degraded.json`。各自的 `datasets/<scene>/reference/scene.json` 与 `navigation/scene.json` 可单独挂载。退化版默认使用原场景碰撞网格的副本，减少漂浮点导致的额外阻塞；`navigation/scene_sparse_collision.json` 可试验直接从退化 PLY 生成的碰撞体。**当前酒店的冻结 A/B 基准不能直接用于这两个新场景**，要评分需另建固定参考域和可靠性状态。
