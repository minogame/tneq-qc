# QCTN 现状 vs 架构设计 对比分析

> 基于 `qctn.py` 实际代码 与 `ARCHITECTURE.md` / `REFACTOR_PLAN.md` 设计目标的对比。

---

## 一、QCTN 现有方法完整清单

### 1.1 QCTNHelper（静态工具类，L11–L485）

| 方法 | 行号 | 类型 | 说明 |
|---|---|---|---|
| `iter_symbols(extend)` | L18 | static | 生成核心符号序列（A-Z 或 CJK） |
| `generate_example_graph(n, target, graph_type, dim_char)` | L34 | static | 生成示例图（mps/tree/wall/circuit/mx 等类型） |
| `generate_example_graph2()` | L460 | static | 空方法（未实现） |
| `generate_random_example_graph(nqubits, ncores)` | L465 | static | 随机生成图 |
| `triu_ndindex(n)` | L481 | static | 上三角矩阵索引生成器 |

内嵌函数（在 `generate_example_graph` 内部）：
- `generate_mps_graph(n, dim_char)` — MPS 链式图
- `generate_tree_graph(n, dim_char)` — 树状图
- `generate_wall_graph_col(n, L, dim_char)` — Brick-wall 图（按列排列）
- `generate_wall_graph(n, L, dim_char)` — Brick-wall 图（按行排列）
- `generate_circuit_graph(n, dim_char)` — 纯线路态图
- `generate_mx_graph(n, dim_char)` — 测量矩阵图

### 1.2 QCTN 类（L487–L1913）

#### 构造 / 初始化

| 方法 | 行号 | 类型 | 说明 |
|---|---|---|---|
| `__init__(graph, backend)` | L513 | instance | 核心构造器：解析图 → 邻接表 → 初始化 cores |
| `from_graph(graph_str, backend)` | L1575 | classmethod | Phase 2 新增，`__init__` 的便捷封装 |
| `from_pretrained(graph, file_path, backend, strict)` | L1013 | classmethod | 从 safetensors 加载预训练权重 |
| `envolve_from_another_qctn(qctn, strategies)` | L573 | classmethod | 从另一个 QCTN 进化生成新实例（遗传算法） |

#### 内部结构解析

| 方法 | 行号 | 类型 | 说明 |
|---|---|---|---|
| `_circuit_to_adjacency()` | L625 | private | 解析图字符串 → `adjacency_table` + `adjacency_matrix` + `circuit` |
| `_init_cores()` | L770 | private | 根据邻接表正交初始化所有 core 张量 |

#### 参数管理

| 方法 | 行号 | 类型 | 说明 |
|---|---|---|---|
| `set_cores(cores, strict)` | L808 | public | 设置核心张量（list 或 dict 输入） |
| `_set_single_core(core_name, tensor)` | L857 | private | 验证形状 + 设置单个 core |
| `_set_cores_from_list(cores, strict)` | L884 | private | 按位置顺序设置 |
| `_set_cores_from_dict(cores, strict)` | L907 | private | 按名称设置 |
| `save_cores(file_path, metadata)` | L948 | public | 保存 cores 到 safetensors |
| `load_cores(file_path, strict)` | L974 | public | 从 safetensors 加载 cores |
| `named_cores(prefix)` | L1613 | public | Phase 2：递归迭代 `(name, tensor)` 对 |
| `all_cores` (property) | L1634 | property | Phase 2：扁平 dict（含子模块前缀） |
| `conjugate_transpose_cores()` | L1889 | public | Phase 2：返回所有 cores 的共轭转置引用 |

#### 收缩操作（Contract）

| 方法 | 行号 | 类型 | 说明 |
|---|---|---|---|
| `contract(attach, engine)` | L1126 | public | 统一入口：根据 attach 类型分派 |
| `_contract_core_only(engine)` | L1032 | private | 仅 core 收缩（无输入） |
| `_contract_with_inputs(inputs, engine)` | L1039 | private | 与单个张量输入收缩 |
| `_contract_with_vector_inputs(inputs, engine)` | L1061 | private | 与 per-qubit 向量列表收缩 |
| `_contract_with_QCTN(qctn, engine)` | L1090 | private | 与另一个 QCTN 收缩 |
| `_contract_with_QCTN_for_gradient(qctn, engine)` | L1108 | private | 与另一个 QCTN 收缩（梯度计算） |
| `contract_with_self(attach, engine)` | L1181 | public | 自收缩 A·A† |
| `_contract_with_self(engine, circuit_array_input, circuit_list_input)` | L1155 | private | 自收缩内部实现 |
| `contract_with_self_for_gradient(attach, engine)` | L1202 | public | 自收缩（梯度版） |
| `_contract_with_self_for_gradient(engine, ...)` | L1168 | private | 自收缩梯度内部实现 |
| `contract_with_QCTN_for_gradient(attach, engine)` | L1223 | public | 与 QCTN 收缩梯度入口 |
| `optimize_contract_with_QCTN(attach, optimizer, engine)` | L1242 | public | 收缩 + 优化（Optimizer 整合） |

#### 图解析分离（Phase 2 / R5）

| 方法 | 行号 | 类型 | 说明 |
|---|---|---|---|
| `get_einsum_info(circuit_states_shapes, measure_shapes, measure_is_matrix)` | L1649 | public | 构建 `A·Mx·A†` 的 einsum 方程和形状列表 |
| `build_core_list(cores_dict, circuit_states, measure_matrices)` | L1826 | public | 按收缩顺序组装张量列表 |

#### 图操作（Split / Merge）

| 方法 | 行号 | 类型 | 说明 |
|---|---|---|---|
| `split(split_idx)` | L1342 | public | 按 core 索引拆分为两个 QCTN |
| `merge(qctn1, qctn2)` | L1450 | static | 左右合并两个 QCTN |
| `merge_with(other)` | L1554 | public | `merge(self, other)` 的便捷方法 |
| `_parse_qubit_line(line)` | L1264 | static | 解析单行图字符串为 token 列表 |
| `_rebuild_qubit_line(tokens)` | L1299 | static | 从 token 列表重建图字符串 |
| `_remap_graph(graph_lines, core_map)` | L1314 | static | 重映射 core 符号 |

#### 子模块管理（Phase 2 / R2）

| 方法 | 行号 | 类型 | 说明 |
|---|---|---|---|
| `register_module(name, module)` | L1589 | public | 注册命名子 QCTN |

#### 其他

| 方法 | 行号 | 类型 | 说明 |
|---|---|---|---|
| `__repr__()` | L609 | special | 字符串表示 |

### 1.3 QCTN 主要属性

| 属性 | 类型 | 说明 |
|---|---|---|
| `qubits` | list[str] | 原始图的每行字符串 |
| `nqubits` | int | qubit 数量 |
| `qubit_indices` | list[int] | `[0, 1, ..., nqubits-1]` |
| `graph` | str | 原始图字符串 |
| `tn_graph` | TNGraph | TNGraph 实例 |
| `cores` | list[str] | core 名称列表（**关键属性，不可重命名**） |
| `ncores` | int | core 数量 |
| `adjacency_table` | list[dict] | 邻接表（主要数据结构） |
| `adjacency_matrix` | np.ndarray | 邻接矩阵（向后兼容） |
| `circuit` | tuple | `(input_ranks, adjacency_matrix, output_ranks)` 向后兼容 |
| `dict_core2idx` | dict | core 名称 → 索引映射 |
| `backend` | ComputeBackend | 计算后端 |
| `cores_weights` | dict | core 名称 → 张量 |
| `einsum_expr` | None | 预留占位（未使用） |
| `_submodules` | dict | Phase 2：子模块注册表 |
| `_loaded_metadata` | dict\|None | 加载的 safetensors 元数据 |

---

## 二、架构设计中 QCTN 应有的功能

根据 `REFACTOR_PLAN.md` R2 + R5 + R3/R4 相关描述：

### 2.1 已实现（Phase 2 完成）

| 设计目标 | 对应方法 | 状态 |
|---|---|---|
| `_cores: Dict[str, TNTensor]` 参数存储 | `cores_weights` | 已有（名称不同但功能等价） |
| `_submodules: Dict[str, QCTN]` 子模块注册 | `_submodules` + `register_module()` | 已实现 |
| `from_graph(graph_str)` classmethod | `from_graph()` | 已实现 |
| `cores()` → Dict 所有 core 张量 | `all_cores` property | 已实现 |
| `named_cores()` 带前缀迭代 | `named_cores(prefix)` | 已实现 |
| `build_core_list(circuit_states, measure_matrices)` | `build_core_list()` | 已实现 |
| `get_einsum_info()` → (equation, shapes) | `get_einsum_info()` | 已实现 |
| 共轭转置引用语义 | `conjugate_transpose_cores()` | 已实现 |
| `split(at_bond)` / `merge(a, b)` | `split()` / `merge()` / `merge_with()` | 已实现 |

### 2.2 未实现

| 设计目标 | 说明 | 优先级 |
|---|---|---|
| `define()` 方法 | 用户重写，定义拓扑结构 | 中（嵌套 QCTN 场景需要） |
| `forward(*args)` 方法 | 用户重写，定义计算图（默认调 `contract`） | 中（nn.Module 风格的核心） |
| `BasicQCTN` 子类 | 从 graph 初始化的具体实现，保留现有逻辑 | 低（可延后到嵌套场景） |
| `CircuitState(QCTN)` 子类 | 线路态作为 QCTN 子类表示 | 中（Phase 3 前置） |
| `MeasurementMatrix(QCTN)` 子类 | 测量矩阵作为 QCTN 子类表示 | 中（Phase 3 前置） |
| 嵌套 `concat(left, right)` | 拼接两个子 QCTN 的结构定义函数 | 低 |
| `from_cores()` classmethod | 非 TNGraph 初始化路径 | 低（Plan A 遗留） |

---

## 三、差异分析

### 3.1 已完成但设计未提及的功能（现有额外功能）

这些方法在架构设计文档中**未被提及**，属于早期开发遗留或后续新增：

| 方法 | 分析 |
|---|---|
| `contract(attach, engine)` 及所有 `_contract_*` 系列（共 10 个） | 直接使用 `ContractorOptEinsum` 做收缩。在新架构中，收缩应通过 Engine + Strategy 完成，而非 QCTN 自身调用。**职责不清** — QCTN 应只管结构和参数，不应直接执行收缩 |
| `contract_with_self` / `contract_with_self_for_gradient` | 自收缩（A·A†），语义上与 `EinsumStrategy.get_compute_function` 重叠 |
| `optimize_contract_with_QCTN(attach, optimizer, engine)` | QCTN 内嵌了优化循环调用，严重违反单一职责。应由 Trainer/Optimizer 管理 |
| `envolve_from_another_qctn(qctn, strategies)` | 遗传算法进化。应属于 `genetic/` 模块，不应在 QCTN 基类中 |
| `set_cores(cores, strict)` 及三个内部 helper | 重新设置 core 权重。设计文档未提，但功能合理，可保留 |
| `save_cores` / `load_cores` / `from_pretrained` | 持久化功能。设计文档未提，但功能合理且实用，可保留 |
| `QCTNHelper` 整个类 | 图生成工具。设计文档中未提到，逻辑上应独立为工具模块 |

### 3.2 设计已要求但仍缺失的功能

| 缺失功能 | 影响 | 建议 |
|---|---|---|
| `define()` / `forward()` 方法 | 无法实现 nn.Module 风格的用户自定义 QCTN | Phase 3 前实现 |
| `CircuitState` / `MeasurementMatrix` 容器类 | circuit states 和 Mx 仍以 raw tensor list 传入，缺乏类型安全和接口统一 | Phase 2 收尾或 Phase 3 前置 |
| 嵌套 QCTN 的 `concat` 结构函数 | `_submodules` 已注册但无法自动拼接图结构 | 低优先级 |
| `from_cores()` classmethod | 无法绕过 TNGraph 直接构建 QCTN | 低优先级 |

### 3.3 职责边界问题（核心差异）

架构设计明确要求 QCTN **仅负责结构定义和参数管理**，收缩/计算/优化应由 Engine/Contractor/Optimizer 负责。但现有实现中：

```
现状                                    设计目标
┌──────────────────────┐            ┌──────────────────────┐
│        QCTN          │            │        QCTN          │
│                      │            │                      │
│  结构定义 ✓          │            │  结构定义 ✓          │
│  参数管理 ✓          │            │  参数管理 ✓          │
│  图解析   ✓          │            │  图解析   ✓          │
│  收缩执行 ✗ ←──┐     │            │  define() ✗          │
│  梯度计算 ✗ ←──┤     │            │  forward() ✗         │
│  优化调用 ✗ ←──┘     │            │                      │
│                      │            └──────────────────────┘
└──────────────────────┘
                                    ┌──────────────────────┐
  10 个 contract_* 方法              │  Engine / Contractor │
  应迁移到 Engine 或                 │  负责收缩执行        │
  标记 deprecated                   └──────────────────────┘
                                    ┌──────────────────────┐
  optimize_contract_with_QCTN        │  Optimizer / Trainer │
  应迁移到 Trainer                   │  负责训练循环        │
                                    └──────────────────────┘
```

### 3.4 冗余与重复

| 问题 | 说明 |
|---|---|
| `adjacency_table` vs `adjacency_matrix` vs `circuit` | 三种结构表示同一信息。`adjacency_table` 是 Phase 2 的主力数据结构；`adjacency_matrix` 和 `circuit` 是向后兼容保留 |
| `tn_graph` vs `_circuit_to_adjacency()` | TNGraph 和 QCTN 内部都解析了图字符串。`tn_graph` 在 Phase 2 后的新方法中未被使用 |
| `contract()` vs Engine 流程 | `contract()` 直接调 `ContractorOptEinsum`，而新架构使用 `EinsumStrategy`/`GreedyStrategy` + Engine |
| `einsum_expr` 属性 | 声明为 `None`，从未被赋值或使用 |

---

## 四、建议行动项

1. **移除**：将以下方法移除：
   - `contract()` 及所有 `_contract_*` 私有方法
   - `contract_with_self()` / `contract_with_self_for_gradient()`
   - `optimize_contract_with_QCTN()`

2. **清理冗余属性**：
   - 移除 `einsum_expr`（从未使用）
   - 考虑移除 `__init__` 中的 `print(f"num cores: {len(self.cores)}")`

3. **作为TODO项**（Plan A 遗留的方案 B）：
   - `CircuitState` — 封装 per-qubit state vectors，提供 `shapes()` / `as_list()`
   - `MeasurementMatrix` — 封装 per-qubit Mx matrices

4. **添加 `define()` / `forward()` 方法**：使 QCTN 支持 nn.Module 风格的用户重写

5. **标记为deprecated `envolve_from_another_qctn`**

6. **作为TODO项**：消除 `tn_graph` 和 `_circuit_to_adjacency()` 的重复，选择一种作为唯一实现

7. **移除向后兼容结构**：当所有下游代码迁移完成后，移除 `adjacency_matrix` 和 `circuit` 元组

8. **拆分 `QCTNHelper`**：移入独立的 `tneq_qc/utils/graph_generators.py`
