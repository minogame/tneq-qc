# tneq-qc 项目综述 (Phase 2.6)

> 量子电路张量网络机器学习框架
> Quantum Circuit Tensor Network (QCTN) Machine Learning Framework

---

## 1. 研究背景与动机

量子启发式机器学习（Quantum-Inspired Machine Learning）借鉴量子力学的数学结构——张量网络（Tensor Network）——来构建经典计算机上可高效运行的机器学习模型。其核心思想是：

- **量子态** 可表示为多个局部张量的乘积（张量网络）
- **期望值**（observable）可通过张量收缩高效计算
- **张量核心**（core tensors）作为可训练参数，通过梯度下降优化

tneq-qc 框架将上述思想工程化，提供从**网络拓扑定义**到**梯度训练**的完整流水线，支持 MPS、树形、砖墙等多种拓扑结构，并兼容 PyTorch 和 JAX 后端。

---

## 2. 核心计算模型

框架提供两种计算目标。

### 2.1 目标一：TNEQ（张量网络内积）

TNEQ 计算两个独立张量网络 $A$、$B$ 之间的内积：

$$\mathcal{L} = \langle A | B \rangle = \mathrm{tr}(A^\dagger \cdot B)$$

当 $A = B$ 时退化为范数 $\|A\|^2$。

- **对应模块**：`TNEQ`（`MPS_L` + `MPS_R` 独立参数）、`MPS_with_Ref`（$A = B$，参数共享）
- **学习目标**：用两个小 TN 的内积近似目标矩阵 $M$ 的 Frobenius 内积结构
- **特点**：无外部数据输入，纯参数优化；适用于无监督密度矩阵学习

### 2.2 目标二：二次型

二次型在 TNEQ 基础上引入数据相关的测量算符 $M_x$：

$$\mathcal{L}(x) = \langle \psi(x) | A^\dagger \cdot M_x \cdot A | \psi(x) \rangle = \mathrm{tr}\!\left(A \cdot M_x \cdot A^\dagger\right)$$

其中：
- $|\psi(x)\rangle$：输入态（CircuitState），将样本特征 $x$ 编码为量子态
- $A$：小张量网络参数（MPS 或其他结构），可训练；$A^\dagger$ 为其零拷贝共轭转置
- $M_x$：由输入 $x$ 生成的测量矩阵（如 Hermite 多项式基展开），代表"大矩阵"的数据相关部分

- **对应模块**：`Quadratic`（CS + MPS + Mx + MPS† + CS†）
- **学习目标**：用小 TN $A$ 学习高维算符 $M$ 的低秩分解，使得 $\mathcal{L}(x)$ 拟合监督信号
- **特点**：有数据驱动的测量矩阵；适用于有监督/半监督量子启发式分类、回归

### 2.3 张量网络拓扑

张量网络的拓扑结构通过 **ASCII 图字符串**定义，每一行对应一个 qubit，字母代表张量核心，数字代表 bond 维度：

```
-2-A-5-----C-3-----E-2-      ← qubit 0
-2-----B----4------E-2-      ← qubit 1
-2-A-4-B-7-C-2-D-4-E-2-      ← qubit 2
-2-----B-6-----D-----2-      ← qubit 3
-2-A-3-----C-8-D-----2-      ← qubit 4
```

上例为 5-qubit TensorNetwork，5 个 core tensor A、B、C、D、E。每个 core 出现在部分 qubit 行上，形成非均匀连接图。

---

## 3. 系统架构

### 3.1 分层设计

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Application Layer                                │
│                                                                             │
│            ┌────────────────────────┐        ┌────────────────┐             │
│            │  Trainer / Distribute  │        │   Inference    │             │
│            └──────┬────────┬────────┘        └───────┬────────┘             │
└───────────────────┼────────┼─────────────────────────┼──────────────────────┘
                    │        │                         │
          ┌─────────┘        └──────────┐              │
          ▼                            ▼               ▼
┌──────────────────────┐    ┌──────────────────────────────────────────────────┐
│      Training        │    │                     Engine                       │
│                      │    │                                                  │
│  ┌────────────────┐  │    │   ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │   Optimizer    │  │───▶│   │ TNEQ_QC  │    │ Quadratic│    │  Trace   │   │
│  ├────────────────┤  │    │   └─────┬────┘    └─────┬────┘    └─────┬────┘   │
│  │     Loss       │  │    │         └───────────────┼───────────────┘        │
│  └───────┬────────┘  │    │                (defined via QCTN)                │
└──────────┼───────────┘    └─────────────────────────┼────────────────────────┘
           │                                          │
           │                                          ▼
           │              ┌────────────────────────────────────────────────────┐
           │              │                     Model Layer                    │
           │              │                                                    │
           │              │   TNGraph ──(init)──▶ ┌──────────────────────┐     │
           │              │                       │        QCTN          │     │
           │              │                       │  ┌────────────────┐  │     │
           │              │                       │  │  CircuitState  │  │     │
           │              │                       │  │  MeasureMx     │  │     │
           │              │                       │  └────────────────┘  │     │
           │              │                       └────┬────────────┬────┘     │
           │              └────────────────────────────┼────────────┼──────────┘
           │                                           │            │
           │                              core tensor  │   forward  │
           │                                           │            │
           │              ┌────────────────────────────┼            │         
           │              │                            │            │         
           ▼              ▼                            ▼            ▼         
┌──────────────────────────────┐    ┌───────────────────────────────────────┐ 
│                              │    │           Strategy Layer              │ 
│          TNTensor            │◀───┤                                       │ 
│       (core tensor)          │    │   ┌──────────────────────────┐        │ 
│                              │    │   │    StrategyCompiler      │        │ 
│                              │    │   └─────┬────────┬───────┬───┘        │ 
│                              │    │         ▼        ▼       ▼            │ 
│                              │    │   ┌────────┐┌────────┐┌───────────┐   │ 
│                              │    │   │Einsum  ││Greedy  ││RowPriority│   │ 
│                              │    │   │Strategy││Strategy││Strategy   │   │ 
│                              │    │   └────────┘└────────┘└───────────┘   │ 
└────────────┬─────────────────┘    └──────────────────┬────────────────────┘ 
             │                                         │                      
             └──────────────────┬──────────────────────┘                      
                                ▼                                             
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Backend Layer                                   │
│                                                                              │
│     ┌──────────────┐    ┌──────────────┐    ┌────────────────────────┐       │
│     │Torch Backend │    │ JAX Backend  │    │ Distributed Primitives │       │
│     └──────────────┘    └──────────────┘    └────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 核心组件职责

| 组件 | 职责 |
|---|---|
| **QCTN** | 持有拓扑（adjacency_table）和参数（cores_weights）；支持 concat/chunk 组合 |
| **TNGraph** | ASCII 图字符串 ↔ 结构化邻接表的双向转换 |
| **TNTensor** | 带独立 scale 因子的张量包装，防止收缩过程中数值下溢/上溢 |
| **EngineCommon** | 统一收缩入口；管理 per-qubit 操作配置（测量、输入态、迹）|
| **StrategyCompiler** | 评估各策略代价，自动选择最优收缩策略 |
| **ComputeBackend** | 屏蔽 PyTorch/JAX 差异，统一张量操作、JIT、梯度、优化器接口 |

---

## 4. 模块系统

### 4.1 叶节点模块（Leaf Modules）

每个叶节点模块是一个有明确物理意义的 QCTN 子类，持有一段张量网络图和对应的核心张量参数。

```
MPS(nqubits=3, bond_dim=4, phys_dim=2)
    图: -2-A-4-B-4-C-2-   (3 行相同)
    物理意义: 矩阵乘积态，bond 维度控制纠缠能力

CircuitState(nqubits=3, phys_dim=2)
    图: -A-2- / -B-2- / -C-2-
    物理意义: 输入量子态 |ψ⟩，每 qubit 独立

MeasureMatrix(nqubits=3, phys_dim=2)
    图: -2-A-2- / -2-B-2- / -2-C-2-
    物理意义: 测量算符，每 qubit 一个矩阵
```

### 4.2 应用模块（Application Modules）

应用模块通过 **composite 模式**（graph=None）将叶节点组合为完整的计算图：

| 模块 | 结构 | 计算 |
|---|---|---|
| `PlainMPS` | 单个 MPS | $\langle A \rangle$（核心范数）|
| `TransposeMPS` | MPS 的零拷贝共轭转置视图 | $A^\dagger$（参数共享）|
| `MPS_with_Ref` | left MPS + right = left† | $\|A\|^2$（对称归一化）|
| `Encoding` | CircuitState + MPS | $A|\psi\rangle$（特征编码）|
| `TNEQ` | MPS_L + MPS_R（独立参数）| $\langle\phi\|\psi\rangle$（内积）|
| `Quadratic` | CS + MPS + Mx + MPS† + CS† | $\langle\psi\|A^\dagger M_x A\|\psi\rangle$（二次型）|

### 4.3 水平组合：concat 与 chunk

QCTN 支持**水平拼接**（concat）和**分割**（chunk），实现灵活的模块拼装：

```
concat(CircuitState[3], MPS[3, bond=4], MeasureMatrix[3]):

CS  (3 qubits, 1 core/qubit):   MPS (3 qubits, 3 cores):   MX (3 qubits, 1 core/qubit):
  -a-2-                           -2-a-4-b-4-c-2-              -2-a-2-
  -b-2-                           -2-a-4-b-4-c-2-              -2-b-2-
  -c-2-                           -2-a-4-b-4-c-2-              -2-c-2-

合并结果 (3 qubits, 5 cores/qubit, cores 重命名为 a..i):
  -a-2-d-4-e-4-f-2-g-2-
  -b-2-d-4-e-4-f-2-h-2-
  -c-2-d-4-e-4-f-2-i-2-
```

concat 后，核心名称自动重编号（opt_einsum 符号），权重按映射复制。`chunk()` 是 concat 的逆操作，按 core index 分割为两个子 QCTN。

---

## 5. 收缩策略

### 5.1 三种策略

张量网络的收缩（contraction）是将多个张量通过 Einstein 求和缩并为标量或低阶张量的过程。
可持续扩展计算策略。
框架提供三种策略，由 `StrategyCompiler` 自动选择：

| 策略 | 图逻辑位置 | 适用场景 |
|---|---|---|
| **EinsumStrategy** | 用einsum计算整个模型 | 小规模网络，快速编译 |
| **GreedyStrategy** | 贪心的计算策略 | 中等规模，精细控制 |
| **RowPriorityStrategy** | 逐行进行的计算策略，控制contract的内存占用 | qubits多但相对稀疏的模型|

### 5.2 对称展开（Symmetric Expansion）

计算二次型 $\langle \psi | A^\dagger M_x A | \psi \rangle$ 时，张量网络展开为左-中-右共5列：

```
CIRCUIT  LEFT (A)    MIDDLE (Mx)    RIGHT (A_T)   CIRCUIT
C0        a              Mx₀           a_T          C0
C1        b              Mx₁           b_T          C1
C2        c              Mx₂           c_T          C2
```

### 5.3 收缩流程

```
QCTN + shapes_info
      │
      ▼
StrategyCompiler.compile()
  ├── check_compatibility()   各策略检查是否适用
  ├── estimate_cost()         估算 FLOPs
  └── 选最优策略
      │
      ▼
strategy.get_compute_function()  → compute_fn
      │
      ▼
engine.execute_expression(compute_fn, *tensors)
      │
      ▼
  标量 / 张量结果
```

---

## 6. 数值稳定性：TNTensor

深层张量网络收缩容易产生数值下溢（underflow）或上溢（overflow）。`TNTensor` 通过分离 scale 因子解决这个问题：

$$\text{真实值} = \text{tensor} \times \text{scale}$$

- `auto_scale()`：将 tensor 归一化到 $\max|t|=1$，将比例吸收到 scale
- `log_scale`：用对数 scale 处理极端数值
- `conj_transpose()`: 对复数的转置自动计算共轭
- **引用语义**：`is_ref=True` 的零拷贝视图，不复制底层数据

---

## 7. 后端抽象

框架通过统一的 `ComputeBackend` 接口屏蔽 PyTorch 和 JAX 的差异：

```python
class ComputeBackend:
    execute_expression(expr, *tensors)         # 执行收缩
    compute_value_and_grad(loss_fn, argnums)   # 值 + 梯度
    jit_compile(func)                          # JIT 编译
    optimizer_update(params, grads, state, …)  # 优化步骤
    init_random_core(shape)                    # 正交随机初始化
```

`BackendFactory` 采用工厂 + 单例模式管理后端实例。训练代码对后端透明。

---

## 8. 训练流程

典型训练循环：

```python
# 1. 构建模型
backend = BackendFactory.create_backend("pytorch", device="cuda", dtype="complex64")
model = Quadratic(nqubits=D, bond_dim=χ, phys_dim=K, backend=backend)
model.auto_init()

# 2. 构建引擎
engine = EngineCommon(backend=backend, strategy_mode="balanced")

# 3. 训练
for x_batch in dataloader:
    Mx_list = engine.generate_data(x_batch, K=K)   # 数据 → 测量矩阵
    loss, grads = engine.contract_with_compiled_strategy_for_gradient(
        model, measure_input_list=Mx_list
    )
    optimizer.step(model, grads)
```

其中 `generate_data` 将输入向量 $x$ 通过 Hermite 多项式基展开为测量矩阵 $M_x$，`contract_with_compiled_strategy_for_gradient` 一次性完成收缩与反向传播。

---

## 9. 分布式并行训练

当张量网络规模（qubit 数 × bond 维度）超出单节点内存/算力时，框架采用**模型并行 + 张量并行**的两阶段分布式策略。

### 9.1 整体架构

```
  输入数据 x_batch（由 Rank 0 生成，MPI broadcast 到所有节点）
                              │
              ┌───────────────▼───────────────┐
              │         Data Broadcast         │
              │   Rank 0 生成 Mx_list，广播    │
              └──┬──────────────┬──────────────┘
                 │              │              │
    ┌────────────▼──┐  ┌────────▼──┐  ┌───────▼───────┐
    │   Worker 0    │  │ Worker 1  │  │   Worker N    │   ← 模型并行
    │               │  │           │  │               │     (MPI Ranks)
    │  QCTN chunk 0 │  │ QCTN chunk│  │ QCTN chunk N  │
    │  cores: a,b,c │  │ cores:d,e │  │ cores: ...    │
    │               │  │           │  │               │
    │  ┌──────────┐ │  │┌─────────┐│  │ ┌──────────┐  │
    │  │ GPU 0    │ │  ││ GPU 0   ││  │ │ GPU 0    │  │   ← 张量并行
    │  │ GPU 1    │ │  ││ GPU 1   ││  │ │ GPU 1    │  │     (节点内多卡)
    │  └──────────┘ │  │└─────────┘│  │ └──────────┘  │
    └───────┬───────┘  └─────┬─────┘  └──────┬────────┘
            │                │                │
            │   local forward + backward      │
            │   (各 worker 仅计算本地 cores)  │
            └────────────────┼────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │     Weight Sync (MPI)        │
              │  sync_weights_after_update() │
              │  各 worker 广播本地更新后权重 │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │     Optimizer Step           │
              │  各 worker 独立更新本地 cores │
              └─────────────────────────────┘
```

### 9.2 第一阶段：模型并行（Model Parallel）

将 QCTN 的 core tensors 按 index 均匀划分到各 MPI worker：

```
QCTN (16 cores: a,b,c,...,p)  →  chunk() 划分
                                        │
         ┌──────────────┬───────────────┼───────────────┐
         ▼              ▼               ▼               ▼
    Rank 0          Rank 1          Rank 2          Rank 3
  cores: a,b,c,d  cores: e,f,g,h  cores: i,j,k,l  cores: m,n,o,p
```

- **前向计算**：每个 rank 完整执行一次收缩（缺失 cores 从其他 rank fetch），得到 loss
- **反向计算**：每个 rank 只计算并持有本地 cores 的梯度
- **权重更新**：本地独立更新 → `sync_weights_after_update()` 广播同步
- **实现类**：`ModelParallelManager`、`ModelParallelTrainer`、`DistributedEngineSiamese`

### 9.3 第二阶段：张量并行（Tensor Parallel）

在模型并行的基础上，对每个 rank 内部的大规模张量收缩进一步分片到多 GPU：

```
单个 Worker 内部（张量并行）:

QCTN chunk（本地 cores）
       │
  ┌────▼────────────────────────────────┐
  │  大 core tensor（如 bond_dim=1024） │
  │  沿 bond 维度分片:                  │
  │                                     │
  │  GPU 0: shard[0:256, :]             │
  │  GPU 1: shard[256:512, :]           │
  │  GPU 2: shard[512:768, :]           │
  │  GPU 3: shard[768:1024, :]          │
  │                                     │
  │  各 GPU 独立执行局部 einsum         │
  │  → all-reduce 拼接为完整结果        │
  └─────────────────────────────────────┘
```

- **收缩分片**：沿 bond 维度将大矩阵乘法拆分到多 GPU，各自独立计算后 all-reduce 聚合
- **适用场景**：bond_dim 很大（≥512）而 qubit 数中等的情况
- **后端支持**：`BackendPyTorch` 通过 `Distributed Primitives` 提供跨 GPU 通信原语

### 9.4 两种并行的协同

```
大规模 QCTN 训练
        │
        ├─ 模型维度大（ncores 多）  →  模型并行：cores 分到多节点
        │
        └─ 张量维度大（bond_dim 大）→  张量并行：单 core 分到节点内多卡

组合使用：
  节点间  →  模型并行
  节点内  →  张量并行
```

---

## 10. 与相关工作的对比

TODO:

---

## 11. 关键设计原则

1. **ASCII 图即接口**：拓扑定义与参数完全解耦，修改网络结构无需修改代码逻辑
2. **组合优于继承**：concat/chunk 允许任意模块水平拼装，composite 模式允许层次嵌套
3. **策略与结构分离**：收缩策略不解析图字符串，只消费 adjacency_table（已解析的结构）
4. **参数共享零拷贝**：TNTensor 引用语义使孪生网络（$A$ 和 $A^\dagger$）无内存冗余
5. **后端透明**：训练逻辑、策略逻辑均不依赖具体后端，支持跨平台迁移
6. **分布式训练**：支持分布式训练，联合使用模型并行和张量并行