# Mooncake 项目总结

> 本文档对 [Mooncake](https://github.com/kvcache-ai/Mooncake) 开源仓库进行整体梳理，帮助新贡献者、集成方与运维人员快速建立全局认知。  
> 官方文档：<https://kvcache-ai.github.io/Mooncake/>

---

## 1. 项目定位

**Mooncake** 是面向大语言模型（LLM）推理的 **以 KVCache 为中心（KVCache-centric）的解耦式服务架构**，由 [Moonshot AI](https://www.moonshot.cn/) 团队开发，并作为 [Kimi](https://kimi.ai/) 的生产级推理平台底座。

核心思想：

- **Prefill / Decode 解耦**：将计算密集的 Prefill 阶段与 Decode 阶段拆分到不同集群，独立扩缩容。
- **分布式 KVCache 池**：利用 GPU 集群中闲置的 CPU、DRAM、SSD 等资源，构建跨节点的 KV 缓存池，用更多存储换取更少重复计算。
- **高性能数据传输**：通过自研 **Transfer Engine（TE）** 实现跨 DRAM/VRAM/NVMe 的零拷贝、多网卡聚合传输。

Mooncake 相关论文获 **FAST 2025 最佳论文奖**，并已进入 PyTorch 生态。仓库同时开源了技术报告、评测 Trace 与核心子系统实现。

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     LLM 推理框架（集成层）                        │
│   vLLM / SGLang / TensorRT-LLM / LMDeploy / LMCache / ...       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ Mooncake Store│  │ Transfer Engine│  │  EP / PG 扩展  │
│ 分布式 KV 缓存 │  │  高性能数据传输 │  │ MoE 弹性并行   │
└───────┬───────┘  └───────┬───────┘  └───────────────┘
        │                  │
        └────────┬─────────┘
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  物理资源：DRAM / VRAM / NVMe SSD / RDMA / NVLink / TCP / ...    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 调度与过载控制（论文架构）

在完整 Mooncake 服务架构中，**KVCache 感知调度器**在最大化有效吞吐的同时满足延迟 SLO。针对高过载场景，系统采用 **基于预测的提前拒绝策略**，避免在无法满足 SLO 的请求上浪费算力。实验表明，在长上下文场景下吞吐最高可提升约 **525%**；在 Kimi 真实负载下可多处理约 **75%** 的请求。

### 2.2 开源范围说明

当前仓库主要开源：

| 组件 | 状态 | 说明 |
|------|------|------|
| Transfer Engine | ✅ 已开源 | 核心数据传输引擎 |
| Mooncake Store | ✅ 已开源 | 分布式 KVCache 存储 |
| P2P Store | ✅ 示例/早期版 | 生产级版本见 [checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine) |
| KVCache 调度器 | 部分在论文/生产中 | 完整调度逻辑未全部开源 |

---

## 3. 核心子系统

### 3.1 Transfer Engine（传输引擎）

**定位**：高性能、零拷贝的数据传输库，是 Mooncake 的技术底座。

**核心抽象**：

| 抽象 | 含义 |
|------|------|
| **Segment** | 可远程读写的连续地址空间，分为 RAM Segment（DRAM/VRAM）与 NVMe-oF Segment |
| **BatchTransfer** | 批量异步传输请求，支持 Read/Write，类似更灵活的 AllScatter/AllGather |

**关键能力**：

- 多 RDMA 网卡带宽聚合
- 拓扑感知路径选择（NUMA / PCIe / GPU Direct RDMA）
- 网络故障自动切换备用路径
- 端点池化与 SIEVE 淘汰算法
- 大请求切片（>64KB）多路径并行

**支持的传输后端**（部分）：

`TcpTransport` · `RdmaTransport` · `EfaTransport` · `NVMeoFTransport` · `NvlinkTransport` · `HipTransport` · Ascend 系列 · Kunpeng UB · Sunrise Link · CXL 等

**性能参考**（40GB 数据，LLaMA3-70B 约 128k tokens 的 KVCache 体量）：

- 4×200 Gbps RoCE：最高约 **87 GB/s**（约为 TCP 的 2.4×）
- 8×400 Gbps RoCE：最高约 **190 GB/s**（约为 TCP 的 4.6×）

**新一代运行时 TENT（Transfer Engine NEXT）**：

位于 `mooncake-transfer-engine/tent/`，面向异构互联与动态拓扑，提供：

- 运行时动态选择传输后端
- 基于遥测的细粒度切片调度
- 运行时内建故障恢复，对应用透明

---

### 3.2 Mooncake Store（分布式 KV 存储）

**定位**：专为 LLM 推理设计的 **分布式 KV Cache 引擎**，不是通用缓存（如 Redis）。对象一旦写入，在删除前不可变；`Get` 保证读到一致且完整的数据。

**架构角色**：

```
                    ┌─────────────────┐
                    │  Master Service │  ← 元数据、空间分配、副本策略
                    │  (mooncake_master)│
                    └────────┬────────┘
                             │ RPC（仅元数据，不经由数据面）
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
    ┌──────────┐       ┌──────────┐       ┌──────────┐
    │ Client A │◄─RDMA─►│ Client B │◄─RDMA─►│ Client C │
    │ (推理节点)│       │ (存储节点)│       │ (混合角色)│
    └──────────┘       └──────────┘       └──────────┘
```

- **Master Service**：管理对象到内存段的映射、副本放置、节点上下线；**不承载数据流**。
- **Client**：双重角色——既是上层应用的 Put/Get 客户端，也是贡献内存的存储节点；数据在 Client 之间经 Transfer Engine 直传。

**部署模式**：

1. **嵌入式**：与 vLLM/SGLang 同进程，以共享库形式导入
2. **Dummy-Real 模式**：每个 rank 一个 dummy client，实例级一个 real client 持有资源
3. **独立 Store 服务**：`python -m mooncake.mooncake_store_service`

**高可用**：支持单 Master（默认）或通过 **etcd** 多 Master 选主（HA 模式）。

**多层存储**：支持 DRAM → SSD/NVMe 卸载，扩大缓存容量。

**主要 API**：`Init` · `Put` · `Get` · `Remove` · `Upsert` · `CreateCopyTask` · `QueryTask` 等（C++ / Python / HTTP / Rust 绑定）。

---

### 3.3 Expert Parallelism（EP）与 Process Group（PG）

位于 `mooncake-ep/` 与 `mooncake-pg/`，为 MoE 模型推理提供：

- 弹性专家并行
- 故障 Rank 检测与自动恢复
- 与 EPLB 模块配合，将 Token 路由到健康 Rank

通过 CMake 选项 `-DWITH_EP=ON` 构建，产出 PyTorch 扩展（`mooncake.ep`、`mooncake.pg`）。

---

### 3.4 P2P Store

位于 `mooncake-p2p-store/`，基于 Transfer Engine 的 **无中心 Master** 点对点对象共享，典型场景为 Checkpoint 分发（类 BitTorrent 种子机制）。生产环境高性能版本已独立为 [checkpoint-engine](https://github.com/MoonshotAI/checkpoint-engine)，可在数千 GPU 上约 20 秒完成 Kimi-K2（1T 参数）权重更新。

---

### 3.5 Python 包与集成层

| 路径 | 职责 |
|------|------|
| `mooncake-wheel/` | PyPI 包 `mooncake-transfer-engine` 的 Python 绑定与 CLI |
| `mooncake-integration/` | pybind11 集成：Store、Transfer Engine、各厂商 Allocator |
| `mooncake-common/` | 公共依赖、etcd/Redis/K8s Lease 元数据后端 |
| `mooncake-rl/` | 强化学习相关示例 |
| `benchmarks/` | 存储与 xPyD 基准测试 |

**PyPI 包变体**：

| 包名 | 适用场景 |
|------|----------|
| `mooncake-transfer-engine` | CUDA ≤ 12.9 |
| `mooncake-transfer-engine-cuda13` | CUDA 13.0/13.1 |
| `mooncake-transfer-engine-non-cuda` | 无 CUDA 环境 |
| `mooncake-transfer-engine-npu` | NPU 环境 |

**常用 CLI 命令**（安装后可用）：

- `mooncake_master` — 启动 Store Master
- `mooncake_client` — Store 客户端工具
- `transfer_engine_bench` — 传输性能测试
- `mooncake_http_metadata_server` — HTTP 元数据服务
- `mc_store_rest_server` — Store REST 服务

---

## 4. 仓库目录结构

```
Mooncake/
├── mooncake-transfer-engine/   # 传输引擎核心（C++，含 TENT）
├── mooncake-store/             # 分布式 KV Store（C++，含 Go/Rust 绑定）
├── mooncake-ep/                # Expert Parallelism CUDA 扩展
├── mooncake-pg/                # Process Group 扩展
├── mooncake-p2p-store/         # P2P 对象共享（Go）
├── mooncake-integration/       # Python/C++ 绑定层
├── mooncake-wheel/             # Python 打包与发行
├── mooncake-common/            # 公共库、元数据后端
├── mooncake-rl/                # RL 示例
├── benchmarks/                 # 性能基准
├── docs/                       # Sphinx 文档源
├── docker/                     # Docker 构建文件
├── scripts/                    # 构建、Ascend、管理脚本
├── monitoring/                 # Grafana / Prometheus 配置
├── FAST25-release/             # 论文、Trace 数据集
├── extern/                     # pybind11、yalantinglibs 等第三方
├── CMakeLists.txt              # 顶层构建配置
├── dependencies.sh             # 依赖安装脚本
└── README.md                   # 项目主页
```

---

## 5. 构建系统

**构建工具**：CMake 3.16+，C++/C，部分 Go/Rust 组件。

**主要 CMake 选项**：

| 选项 | 默认 | 说明 |
|------|------|------|
| `WITH_TE` | ON | 构建 Transfer Engine |
| `WITH_STORE` | ON | 构建 Mooncake Store |
| `WITH_EP` | OFF | 构建 EP/PG PyTorch 扩展 |
| `WITH_P2P_STORE` | OFF | 构建 P2P Store |
| `WITH_STORE_RUST` | ON | Store Rust 绑定 |
| `USE_NOF` | OFF | NVMe-oF SSD 池 |
| `STORE_USE_ETCD` | OFF | Store 使用 etcd HA |
| `USE_CUDA` / `USE_MLU` / `USE_HYGON` 等 | 按需 | 各硬件后端 |

**快速构建**：

```bash
git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
sudo bash dependencies.sh
mkdir build && cd build
cmake ..
make -j
sudo make install   # 可选
```

**推荐环境**：Ubuntu 22.04+、Python 3.10+、RDMA 驱动（Mellanox OFED）、CUDA 12.1+（GPU 场景）。

---

## 6. 生态集成

Mooncake 已深度集成主流 LLM 推理与训练栈：

### 6.1 推理框架

| 框架 | 集成方式 | 典型场景 |
|------|----------|----------|
| **vLLM** | `MooncakeConnector` / `MooncakeStoreConnector` | PD 解耦、跨实例 KV 共享 |
| **SGLang** | Transfer Engine + HiCache L3 后端 | 分层 KV 缓存、EPD 解耦、多模态 Embedding 缓存 |
| **TensorRT-LLM** | `mooncake_utils` | PD 解耦 KV 传输 |
| **vLLM-Ascend** | Transfer Engine | 昇腾 NPU PD 解耦 |
| **LMDeploy** | PD 解耦后端 | 分布式 Prefill/Decode |
| **vLLM-Omni** | Store / TE Connector | 多模态流水线解耦 |
| **LightX2V** | Transfer Engine | 视频生成 Encoder/Transformer 解耦 |

### 6.2 KV 缓存管理

- **LMCache**：Mooncake Store 作为远程 Connector
- **FlexKV**：基于 TE 的分布式 KVCache 复用
- **SGLang HiCache**：L1 GPU → L2 Host → L3 Mooncake Store 三级缓存

### 6.3 训练与权重传输

- **checkpoint-engine**：生产级 P2P 权重分发
- **SGLang P2P RL**：大规模 RL 权重 RDMA 更新（Kimi-K2 1T 参数 53s→7.2s）
- **TorchSpec**：推测解码训练中的 Hidden States 管理
- **NIXL**：TE 作为后端插件

### 6.4 典型集成模式

1. **PD Disaggregation（Prefill-Decode 解耦）**  
   Prefill 节点计算 KV → Transfer Engine 传输 → Decode 节点消费 KV。

2. **HiCache 分层缓存**  
   本地 GPU/Host 未命中时，从 Mooncake Store（L3）并行 RDMA 预取。

3. **xPyD（多 Prefill : 多 Decode）**  
   任意 Prefill 与 Decode 节点间通过 Store 共享 KV，提升资源利用率。

---

## 7. 硬件支持

Mooncake 支持多种加速器与互联：

**GPU / 加速器**：NVIDIA、AMD、华为昇腾、寒武纪、摩尔线程、MetaX、平头哥等

**互联协议**：RDMA (RoCE/IB)、TCP、AWS EFA、NVLink、NVMe-oF、HIP、CXL、Ascend Direct 等

**建议**：生产环境强烈建议使用 RDMA；TCP 仅适合功能验证。

---

## 8. 元数据与运维

**元数据服务**（Transfer Engine / Store 共用，需独立部署）：

- etcd（推荐 HA）
- Redis
- HTTP Metadata Server（`mooncake_http_metadata_server`）
- Kubernetes Lease（`STORE_USE_K8S_LEASE`）

**监控**：`monitoring/` 提供 Grafana / Prometheus 配置模板。

**部署文档**：

- [Mooncake Store 部署指南](https://kvcache-ai.github.io/Mooncake/deployment/mooncake-store-deployment-guide.html)
- [NVMe-oF SSD 部署](https://kvcache-ai.github.io/Mooncake/deployment/nvmf-ssd-deployment-guide.html)

---

## 9. 研究与开源数据

### 9.1 论文

- FAST 2025：`Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot`
- ACM TOS 2025 期刊版
- arXiv 预印本：2407.00079
- MoE 容错：Surviving Partial Rank Failures in Wide Expert-Parallel MoE Inference (2026)

### 9.2 Trace 数据集

路径：`FAST25-release/traces/`

```json
{
    "timestamp": 27482,
    "input_length": 6955,
    "output_length": 52,
    "hash_ids": [46, 47, 48, ...]
}
```

字段说明：请求到达时间、输入/输出 token 数、重映射后的 block hash（已脱敏）。数据集约 **50%** 缓存命中率，可用于调度与缓存策略仿真。

---

## 10. CI 与质量保障

**GitHub Actions 工作流**（`.github/workflows/`）：

| 工作流 | 用途 |
|--------|------|
| `ci.yml` | 主 CI 构建与测试 |
| `ci_ascend.yml` | 昇腾环境 CI |
| `ci_cu13.yml` | CUDA 13 CI |
| `e2e-ci.yml` / `integration-test.yml` | 端到端与集成测试 |
| `release*.yaml` | 多平台 PyPI 发布 |

**本地验证**：`.ci/run_test.sh`、`scripts/run_ci_test.sh`（见 `.claude/skills/mooncake-ci-local/`）。

**代码规范**：pre-commit（clang-format、ruff、cmake-format、codespell）。

---

## 11. 贡献指南

- 贡献流程见 `CONTRIBUTING.md`
- PR 标题前缀：`[TransferEngine]`、`[Store]`、`[Integration]`、`[Doc]` 等
- 大型架构变更（>500 LOC）需先开 RFC Issue
- 治理与维护者：`MAINTAINERS.md`、`docs/source/community/governance.md`

---

## 12. 关键设计文档索引

| 主题 | 文档路径 |
|------|----------|
| 整体架构 | `docs/source/design/architecture.md` |
| Transfer Engine | `docs/source/design/transfer-engine/index.md` |
| Mooncake Store | `docs/source/design/mooncake-store.md` |
| TENT 下一代运行时 | `docs/source/design/tent/overview.md` |
| HiCache 集成设计 | `docs/source/design/hicache-design.md` |
| P2P Store | `docs/source/design/p2p-store.md` |
| SSD 卸载 | `docs/source/design/ssd-offload.md` |
| 构建指南 | `docs/source/getting_started/build.md` |
| vLLM 集成 | `docs/source/getting_started/examples/vllm-integration/` |
| SGLang 集成 | `docs/source/getting_started/examples/sglang-integration/` |

---

## 13. 一句话总结

**Mooncake = 以 KVCache 为核心的 LLM 推理基础设施**：用 Transfer Engine 打通异构高速互联，用 Mooncake Store 构建弹性分布式 KV 池，再与 vLLM/SGLang 等框架集成，实现 PD 解耦、分层缓存与 MoE 弹性推理，已在 Kimi 生产环境大规模验证。

---

## 14. 相关链接

- 仓库：<https://github.com/kvcache-ai/Mooncake>
- 文档：<https://kvcache-ai.github.io/Mooncake/>
- PyPI：<https://pypi.org/project/mooncake-transfer-engine/>
- 论文：<https://www.usenix.org/conference/fast25/presentation/qin>
- Slack：<https://join.slack.com/t/mooncake-project/shared_invite/zt-3qx4x35ea-zSSTqTHItHJs9SCoXLOSPA>
- 博客：<https://kvcache.ai/>

---

*文档版本：基于仓库 main 分支梳理，生成日期 2026-06-11。*
