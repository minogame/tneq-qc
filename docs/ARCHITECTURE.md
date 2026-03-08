# tneq-qc 项目架构总结

## 项目概述

tneq-qc 是一个基于张量网络（Tensor Network）的量子电路模拟与机器学习框架。核心思路是将量子电路表示为张量网络，并通过优化张量收缩路径来高效计算量子态的期望值，从而实现量子启发式机器学习（Quantum-Inspired Machine Learning）。

---

## 目录结构

```
tneq-qc/
├── tneq_qc/                  # 核心库
│   ├── backends/             # 计算后端（JAX / PyTorch）
│   ├── core/                 # 核心模型（QCTN、Engine、TNTensor 等）
│   ├── contractor/           # 收缩策略（Einsum / Greedy / MPS）
│   ├── optim/                # 优化器
│   ├── distributed/          # 分布式训练
│   ├── genetic/              # 遗传算法（网络结构搜索）
│   ├── config.py             # 全局配置
│   ├── callbacks.py          # 回调系统
│   └── log_utils.py          # 日志工具
├── tests/                    # 测试套件
├── train.py                  # 训练入口脚本
└── docs/                     # 文档
```

---

## 模块详解

### 1. `backends/` — 计算后端

**职责**：屏蔽底层框架差异，提供统一的张量操作接口。

#### 关键类

| 类 | 文件 | 说明 |
|---|---|---|
| `BackendInfo` | `backend_interface.py` | 后端配置信息（类型、设备、dtype） |
| `ComputeBackend` | `backend_interface.py` | 抽象基类，定义所有张量操作接口 |
| `BackendJAX` | `backend_jax.py` | JAX 后端实现 |
| `BackendPyTorch` | `backend_pytorch.py` | PyTorch 后端实现 |
| `BackendFactory` | `backend_factory.py` | 工厂类，创建/管理后端实例 |
| `ContractorOptEinsum` | `copteinsum.py` | 基于 opt_einsum 的收缩执行器 |

#### 设计模式

- **抽象接口 + 工厂模式**：`ComputeBackend` 定义约 25 个抽象方法，覆盖张量创建、数学运算、梯度计算、JIT 编译、优化器更新等。
- **TNTensor 集成**：后端可选配置 `tensor_type="TNTensor"`，此时 `init_random_core` 自动返回 `TNTensor` 包装对象。
- **默认后端**：`BackendFactory.get_default_backend()` 若未设置则默认返回 JAX GPU 后端。

#### 核心接口

```python
class ComputeBackend(ABC):
    def execute_expression(self, expression, *tensors)     # 执行收缩表达式
    def compute_value_and_grad(self, loss_fn, argnums)     # 值+梯度
    def jit_compile(self, func)                            # JIT 编译
    def optimizer_update(self, params, grads, state, method, hyperparams)  # 优化步骤
    def init_random_core(self, shape)                      # 正交初始化
    def einsum(self, equation, *operands)                  # Einstein 求和
    def is_complex(self, tensor) -> bool                   # 是否复数张量
    def abs_square(self, tensor)                           # Born 规则 |ψ|²
```

---

### 2. `core/` — 核心模型

**职责**：定义量子电路张量网络的数据结构和计算引擎。

#### 关键类

| 类 | 文件 | 说明 |
|---|---|---|
| `TNTensor` | `tn_tensor.py` | 带缩放因子的张量包装器 |
| `TNGraph` | `tn_graph.py` | 张量网络图的 ASCII 表示与解析 |
| `QCTN` | `qctn.py` | 量子电路张量网络核心模型 |
| `QCTNHelper` | `qctn.py` | QCTN 辅助工具（图生成、格式转换） |
| `Engine` | `engine.py` | 标准收缩引擎（后端 + 策略） |
| `EngineCommon` | `engine_common.py` | 通用引擎（支持 QubitOp 配置） |
| `EngineSiamese` | `engine_siamese.py` | 孪生引擎（含 Hermite 多项式权重） |

#### `TNTensor` — 张量包装器

解决张量网络收缩中的数值下溢/上溢问题：

```
真实值 = tensor × scale
```

- `auto_scale()`：自动归一化，将 `scale` 吸收到 `tensor` 中
- `scale_to(new_scale)`：调整 scale，保持真实值不变
- 支持 `log_scale` 以处理极端数值

#### `TNGraph` — 图表示

用 ASCII 字符串表示张量网络拓扑：

```
-2-------B--5--C--3--D-------2-   ← qubit 0
-2-A-4---------------D-------2-   ← qubit 1
-2-A--4--B--7--C--2--D--4--E-2-   ← qubit 2
-2-A--3--B--6--------------E-2-   ← qubit 3
-2-------------C--8--------E-2-   ← qubit 4
```

- 字母（A-Z）= 张量核心（core tensor）
- 数字 = 键合维度（bond dimension）
- 内部表示：`graph[i] = [(tensor_name, left_bond, right_bond), ...]`

#### `QCTN` — 量子电路张量网络

框架的核心模型类，包含：
- 张量网络拓扑（via `TNGraph`）
- 核心张量参数（`cores_dict`）
- 支持 split/merge 操作
- 通过 `QCTNHelper` 生成 einsum 表达式
- `cqctn.py` 提供扩展的复数量子电路版本

#### `QubitOp` 枚举（engine_common.py）

```python
class QubitOp(Enum):
    TRACE          # 迹出（与单位矩阵收缩）
    CIRCUIT_LEFT   # 左侧（bra）乘以线路态
    CIRCUIT_RIGHT  # 右侧（ket）乘以线路态
    CIRCUIT_BOTH   # 两侧均乘以线路态
    MEASURE        # 应用测量矩阵 Mx
    IDENTITY       # 恒等（不操作）
```

#### `EngineSiamese`

专为量子启发式机器学习设计的孪生引擎：
- 内置 Hermite 多项式权重初始化（`mx_K` 阶）
- 处理左/右两侧的张量网络（`A` 和 `A†`）
- 计算 `tr(A · Mx · A†)` 形式的期望值

---

### 3. `contractor/` — 收缩策略

**职责**：将 QCTN 图结构编译为高效的张量收缩计算函数。

#### 架构

```
ContractionStrategy (abstract base)
├── EinsumStrategy      # 基于 opt_einsum，Fast 模式
├── MPSChainStrategy    # MPS 链式收缩（暂禁用）
└── GreedyStrategy      # 贪心逐 qubit 收缩，Balanced/Full 模式

StrategyCompiler        # 策略选择器/编译器
```

#### `ContractionStrategy` 抽象接口

```python
class ContractionStrategy(ABC):
    def check_compatibility(self, qctn, shapes_info) -> bool  # 兼容性检查
    def get_compute_function(self, qctn, shapes_info, backend) -> Callable  # 生成计算函数
    def estimate_cost(self, qctn, shapes_info) -> float        # 估算 FLOPs
    def name(self) -> str                                       # 策略名称
```

#### `StrategyCompiler` — 编译器

- 三种模式：`fast`（仅 Einsum）、`balanced`（Greedy）、`full`（全策略）
- 全局策略注册表（类变量），模块导入时自动注册
- `compile()` 流程：兼容性检查 → 估算成本 → 生成计算函数 → 选最优策略

#### `EinsumStrategy`

- 使用 `opt_einsum` 生成并优化 einsum 表达式
- 兼容所有网络结构
- 返回 `compute_fn(cores_dict, circuit_states, measure_matrices)`

#### `GreedyStrategy`

- 逐 qubit 贪心收缩
- 对称展开：左侧（L）、中间测量矩阵（M）、右侧共轭转置（R）
- 计算 `Σᵢ (Aᵢ · Mx · Aᵢ†)`

---

### 4. `optim/` — 优化器

**职责**：提供参数优化算法。

#### `Optimizer`

支持优化方法：
- `adam`：自适应矩估计
- `sgdg`：Stiefel 流形上的随机梯度下降（适用于正交约束）

关键参数：
- `lr_schedule`：学习率调度表 `[(step, lr), ...]`
- `stiefel`：是否使用 Stiefel 流形优化
- 优化状态（momentum、Adam m/v）保存在 `opt_state`

实际的梯度更新委托给 `backend.optimizer_update()`，Optimizer 主要负责：
1. 学习率调度
2. 数据集迭代控制
3. 调用 engine 计算损失和梯度

---

### 5. `distributed/` — 分布式训练

**职责**：支持多节点/多 GPU 分布式张量网络训练。

#### 子模块结构

```
distributed/
├── comm/                     # 通信抽象层
│   ├── comm_interface.py     # CommBase 抽象接口
│   ├── comm_torch.py         # PyTorch distributed 实现
│   ├── comm_mpi.py           # MPI 实现
│   └── comm_factory.py       # 通信后端工厂
├── parallel/
│   └── distributed_contractor.py  # 分布式收缩
├── engine/
│   └── distributed_engine.py     # 分布式引擎
├── trainer/
│   └── distributed_trainer.py    # 分布式训练器（高层 API）
├── optim/
│   ├── distributed_sgdg.py   # 分布式 SGDG 优化器
│   └── allreduce_grad.py     # AllReduce 梯度同步
├── mpi_core.py               # MPI 核心通信
├── mpi_agent.py              # MPI Agent（工作进程）
└── mpi_overlord.py           # MPI Overlord（主控进程）
```

#### 关键设计

- **通信层抽象**：`CommBase` 定义 AllReduce、Broadcast、AllGather 等集合通信原语，支持 MPI 和 PyTorch Distributed 两种后端互换。
- **`DistributedContext`**：记录 `world_size`、`rank`、`node_rank` 等分布式上下文。
- **`DistributedConfig`**：统一的分布式配置数据类（backend、device、strategy_mode、通信类型等）。

---

### 6. `genetic/` — 遗传算法

**职责**：通过进化搜索自动优化张量网络拓扑结构（网络架构搜索 NAS）。

- `EVOLVE_OPS`：变异（mutation）、交叉（crossover）、淘汰（elimination）、移民（immigration）操作
- `FITNESS_FUNCS`：适应度函数定义
- `mpi_generation.py`：基于 MPI 的并行进化代

---

### 7. `tests/` — 测试套件

| 测试文件 | 覆盖模块 |
|---|---|
| `test_tn_graph.py` | TNGraph 解析与操作 |
| `test_tensor_network.py` | 张量网络基础功能 |
| `test_greedy_strategy.py` | GreedyStrategy 收缩 |
| `test_refactored_backend.py` | Backend 接口测试 |
| `test_sample.py` | 采样功能 |
| `test_probabilities.py` | 概率计算 |
| `test_distributed.py` | 分布式基础 |
| `test_distributed_trainer.py` | 分布式训练器 |
| `test_model_parallel.py` | 模型并行 |
| `test_mpi_agent.py` | MPI Agent |
| `test_mpi_overlord.py` | MPI Overlord |

---

### 8. `train.py` — 训练入口

单机训练脚本，演示完整的训练流程：

1. 配置 PyTorch 后端（`BackendFactory`）
2. 构建 QCTN（图字符串定义拓扑）
3. 生成线路态（`circuit_states_list`）
4. 初始化 `EngineSiamese`（含 Hermite 权重）
5. 初始化 `Optimizer`（Adam / SGDG）
6. 训练循环：计算损失 → 反向传播 → 参数更新
7. 使用 TensorBoard 记录训练曲线

---

## 整体数据流

```
用户定义
  ↓
TNGraph (ASCII图 → 邻接表)
  ↓
QCTN (图 + core tensors参数)
  ↓
Contractor (EinsumStrategy / GreedyStrategy)
  ↓
  compile() → compute_fn(cores_dict, circuit_states, measure_matrices)
  ↓
Engine / EngineSiamese
  ↓
ComputeBackend.execute_expression()
  ↓
  result (期望值 / 损失)
  ↓
Optimizer → backend.optimizer_update() → 参数更新
```

---

## 模块依赖关系

```
train.py
  └─ EngineSiamese (core/)
       ├─ BackendFactory → BackendPyTorch / BackendJAX (backends/)
       ├─ EinsumStrategy / GreedyStrategy (contractor/)
       └─ QCTN (core/)
            └─ TNGraph, TNTensor (core/)

Optimizer (optim/)
  └─ Engine (core/)
       └─ backend.optimizer_update() (backends/)

DistributedTrainer (distributed/)
  ├─ CommBase → CommTorch / CommMPI (distributed/comm/)
  └─ DistributedEngineSiamese (distributed/engine/)
       └─ EngineSiamese (core/)
```

---

## 设计亮点

1. **多后端支持**：统一抽象接口，JAX 和 PyTorch 可无缝切换，未来可扩展 CuPy 等后端。
2. **策略模式**：收缩策略可插拔，通过 `StrategyCompiler` 自动选优。
3. **TNTensor 精度控制**：通过分离 `tensor × scale` 避免大规模张量网络中的数值下溢。
4. **分布式透明化**：通信层抽象使 MPI 和 NCCL 可互换。
5. **量子启发式设计**：Born 规则（`abs_square`）、Hermite 多项式权重、孪生网络结构都是面向量子机器学习的特有设计。
