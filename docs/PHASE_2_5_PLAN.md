# Phase 2.5 重构计划：Engine 与 Contractor

## 一、前序阶段回顾

### Phase 1：TNTensor 增强

为 TNTensor 添加了 scale/log_scale 数值稳定性支持，统一了张量包装接口，为后续重构奠定基础。

### Phase 2：QCTN 重构

将 QCTN 从"大杂烩"重构为纯粹的量子电路张量网络数据结构。核心变更包括：

- **图解析分离**：新增 `get_einsum_info()` 和 `build_core_list()`，将 EinsumStrategy 中的图解析逻辑迁回 QCTN，策略层只负责调用
- **新增结构化接口**：`define()`、`forward()`、`split()`、`merge()`、`named_cores()`、`all_cores`、`register_module()`、`conjugate_transpose_cores()`
- **清理完成项**：移除了 QCTN 上的 contract 方法、清理了冗余属性、废弃了 `envolve_from_another_qctn`、将辅助函数分离至 `QCTNHelper`（graph_generators.py）
- **清理 TODO 项**（暂不处理）：CircuitState / MeasurementMatrix 引入、`tn_graph` 去重、`adjacency_matrix` 移除

### 当前代码结构概览

| 文件 | 行数 | 角色 |
|------|------|------|
| `core/qctn.py` | ~1217 | 量子电路张量网络数据结构 |
| `core/engine_common.py` | ~1418 | 主引擎（27 个方法） |
| `core/engine.py` | — | 旧引擎，含 9 个 legacy 方法 |
| `core/engine_siamese.py` | — | EngineCommon 的近似副本 |
| `contractor/greedy_strategy.py` | ~1040 | 贪心逐 qubit 收缩策略 |
| `contractor/einsum_strategy.py` | — | opt_einsum 策略（已部分重构） |
| `contractor/compiler.py` | — | StrategyCompiler 策略选择器 |
| `contractor/base.py` | — | ContractionStrategy ABC |
| `optim/optimizer.py` | — | 优化器（Phase 3 目标） |

---

## 二、Phase 2.5 目标

在 Phase 2（QCTN）和 Phase 3（Optimizer/Trainer）之间，完成 **Engine 与 Contractor 的职责梳理和重构**。核心原则：

1. EngineCommon 作为唯一引擎，`engine.py` 和 `engine_siamese.py` 将来废弃（暂不处理，因无继承关系）
2. 图构建和布线逻辑属于 QCTN 的职责，应从 GreedyStrategy 迁出
3. 明确区分"已废弃"和"当前主入口"的方法

---

## 三、EngineCommon 方法分类与处置

对 EngineCommon 的 27 个方法按功能分为七类，并明确各类处置方式：

### A. 构造与配置（3 个）— 保留

`__init__`、`nqubits` property/setter、`_resolve_nqubits`

### B. QubitOp 系统（8 个）— 保留

`set_partial_trace`、`set_circuit_left`、`set_circuit_right`、`set_circuit_both`、`set_measure`、`set_identity`、`reset_qubit_ops`、`get_qubit_op`

用于配置每个 qubit 的操作类型（TRACE / CIRCUIT_LEFT / RIGHT / BOTH / MEASURE / IDENTITY）。

### C. Pipeline 系统（2 个）— 标记废弃

`add_pipeline_entry`、`clear_pipeline`

Pipeline 是有序的 TN/MX/CIRCUIT 条目列表，与 `run_pipeline` 绑定。随着 `run_pipeline` 废弃，Pipeline 系统一并标记 deprecated。

### D. 收缩前准备（2 个）— 保留

`build_contraction_inputs`、`_resolve_pipeline_inputs`

为收缩过程准备输入数据。

### E. 收缩入口（4 个）

| 方法 | 处置 |
|------|------|
| `run_pipeline` | **标记废弃** |
| `run_pipeline_for_gradient` | **标记废弃** |
| `contract_with_compiled_strategy` (L757) | **主入口，保留** |
| `contract_with_compiled_strategy_for_gradient` (L847) | **主入口，保留** |

注意：`contract_with_compiled_strategy_for_gradient` 内部硬编码了 Cross Entropy loss（L937-1030），后续需要解耦（Phase 3 处理）。

### F. 数据生成（4 个）— TODO：迁至 data_utils

`_init_mx_weights`、`_eval_hermitenorm_batch`、`_eval_hermitenorm_batch_np`、`generate_data`

Hermite 多项式数据生成功能，属于数据工具而非引擎职责。计划迁移至 `data_utils` 模块，但该模块尚未规划，暂保留并标记 TODO。

### G. 应用函数（4 个）— 暂时保留

`calculate_full_probability`、`calculate_marginal_probability`、`calculate_conditional_probability`、`sample`

---

## 四、GreedyStrategy 重构：图逻辑迁入 QCTN

### 现状

`GreedyStrategy.get_compute_function` 返回一个 `compute_fn` 闭包，其内部流程分为四个阶段：

**Stage 1：图构建（L75-297）**
读取 `qctn.adjacency_table`，构建 `core_tensor_list`，包含：LEFT 核心张量 → LEFT 电路张量 → MIDDLE Mx → RIGHT 核心张量（边反转）→ RIGHT 电路张量。实现了对称展开模式 A · Mx · A†。

**Stage 2：布线（L300-451）**
更新 LEFT/RIGHT 邻居连接关系，为 einsum 分配符号（字符标签）。

**Stage 3：按 qubit 收缩（L458-588）**
遍历每个 qubit，通过 Union-Find 算法找到连通分组，调用 `_contract_symmetric_group` 执行局部 einsum，更新 entry 列表。辅助方法：`_find_connected_groups_symmetric` (L617)、`_get_tensor` (L669)、`_contract_symmetric_group` (L692)。

**Stage 4：返回结果（L589-601）**
单个 entry 直接返回；多个 entry 调用 `_contract_remaining` (L996) 完成最终收缩。

### 重构方案

Stage 1（图构建）和 Stage 2（布线）迁入 QCTN，作为新方法，**与 `get_einsum_info` 并行存在**——`get_einsum_info` 为 EinsumStrategy 提供图信息，新方法为 GreedyStrategy 提供图信息。两者地位对等，是 QCTN 面向不同收缩策略的两套图信息输出接口。

迁移后 GreedyStrategy 只保留 Stage 3-4 的纯收缩逻辑，通过调用 QCTN 接口获取图信息。

### 已知问题

- Stage 3 中硬编码了 `torch.einsum`（L943、L962），应改为 `backend.einsum`
- `compute_fn` 内部有 `import torch`，应消除

---

## 五、重构行动项总览

### 5.0 移除 adjacency_matrix，统一使用 adjacency_table

**最先执行**。移除 `QCTN` 上的 `adjacency_matrix` 属性和 `circuit` 元组，`adjacency_table` 成为唯一的图数据结构。

涉及文件及处理方式：

| 文件 | 当前用法 | 处置 |
|------|----------|------|
| `core/qctn.py` | `_circuit_to_adjacency` 末尾生成 `adjacency_matrix` + `circuit`；`__repr__` 使用 | 移除生成代码；`__repr__` 改用 `adjacency_table` |
| `backends/copteinsum.py` | 所有方法通过 `qctn.circuit` 获取 `adjacency_matrix` | Legacy JAX 代码，被 `mpi_agent.py` 引用。在 `copteinsum.py` 内部自行从 `adjacency_table` 构建所需数据，不再依赖 `qctn.adjacency_matrix` |
| `core/engine.py` L655 | 一行 `input_ranks, adjacency_matrix, output_ranks = qctn.circuit` | Legacy 方法，改用 `adjacency_table` 或标记废弃 |

### 5.1 创建 RowPriorityStrategy（替代 GreedyStrategy 图逻辑）

将 GreedyStrategy 的 Stage 1（图构建）和 Stage 2（布线）迁入 `qctn.py` 作为新方法（与 `get_einsum_info` 并行）。基于此创建新的 **RowPriorityStrategy**，调用 QCTN 接口获取图信息，自身只保留 Stage 3-4 的纯收缩逻辑。

原 GreedyStrategy 保留不变，RowPriorityStrategy 作为新策略与之并存。

### 5.2 标记 EngineCommon 废弃方法

- `run_pipeline`、`run_pipeline_for_gradient` → 标记 `@deprecated`
- `add_pipeline_entry`、`clear_pipeline`（Pipeline 系统）→ 标记 `@deprecated`

### 5.3 清理 EinsumStrategy legacy 方法

EinsumStrategy 中仍存在直接做图解析的 legacy 静态方法：`build_core_only_expression`、`build_with_inputs_expression`、`build_with_vector_inputs_expression`、`build_with_qctn_expression`、`_build_with_self_expression_legacy`。计划移除或标记废弃。

### 5.4 修复 GreedyStrategy 硬编码

- `torch.einsum` → `backend.einsum`
- 消除 `compute_fn` 内的 `import torch`

### 5.5 TODO（暂不执行）

- 数据生成方法（F 类）迁移至 `data_utils` 模块（模块尚未规划）
- `engine.py`、`engine_siamese.py` 正式废弃/移除
- `contract_with_compiled_strategy_for_gradient` 中的 Cross Entropy loss 解耦（Phase 3 处理）

---

## 六、后续阶段展望

### Phase 3：Optimizer / Trainer

- Optimizer 重构为单步 update 接口
- 引入独立的 Loss 模块，解耦当前硬编码在引擎中的 Cross Entropy
- 新建 Trainer 类管理训练循环
- 清理 optimizer.py 中已失效的方法（`optimize_debug`、`optimize_with_target`、`optimize_self_with_inputs` 等调用了已移除的 QCTN 方法）