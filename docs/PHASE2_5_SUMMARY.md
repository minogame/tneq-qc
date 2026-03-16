# Phase 2.5 重构总结

## 目标回顾

Phase 2.5 目标是在 Phase 2（QCTN 模块化）和 Phase 3（Optimizer/Trainer）之间，完成 **Engine 与 Contractor 的职责梳理和重构**。实现以下子目标：

- **5.0**：移除 `adjacency_matrix`，统一使用 `adjacency_table` 作为唯一图数据结构
- **5.1**：创建 `RowPriorityStrategy`，将 GreedyStrategy 的图构建/布线逻辑迁入 QCTN
- **5.2**：标记 EngineCommon 废弃方法（Pipeline 系统 + `run_pipeline*`）
- **5.3**：标记 EinsumStrategy legacy 静态方法为废弃

---

## 变更文件

### 1. `tneq_qc/core/qctn.py`（主要增量）

#### 移除的属性

| 属性 | 说明 |
|------|------|
| `self.adjacency_matrix` | 原 `_circuit_to_adjacency` 末尾生成的 `np.ndarray`，已删除 |
| `self.circuit` | 原 `(input_ranks, adjacency_matrix, output_ranks)` 元组，已删除 |

#### 新增枚举（模块级）

```python
class TensorSide(Enum):
    LEFT = "L"
    MIDDLE = "M"
    RIGHT = "R"
```

用于标记对称展开图中每个 entry 的侧别（左/中/右）。

#### 新增方法

| 方法 | 说明 |
|------|------|
| `build_symmetric_expansion_graph(circuit_states_shapes, measure_shapes, right_qctn)` | 构建 L-M-R 对称展开图，返回 `(core_tensor_list, maps)`。与 `get_einsum_info` 并行——后者为 EinsumStrategy 提供 einsum 方程，本方法为 RowPriorityStrategy 提供逐 qubit 收缩所需的图信息 |

**参数说明**：

- `circuit_states_shapes`：`list[tuple]` 或 `None`，每个 qubit 的 circuit state 形状
- `measure_shapes`：`list[tuple|None]` 或 `None`，每个 qubit 的 Mx 形状（`None` 表示该 qubit 无测量）
- `right_qctn`：`"symmetric"`（默认，镜像左侧）| `QCTN` 实例 | `None`（无右侧）

**返回值**：

- `core_tensor_list`：entry dict 列表，每个 entry 含 `core_idx`、`core_name`、`tensor_source`、`tensor_key`、`in_edge_list`、`out_edge_list`（已完成布线和符号分配）、`side`（TensorSide）、`batch_symbol`
- `maps`：`dict`，含 `left_core_map`、`right_core_map`、`mx_map`、`left_circuit_map`、`right_circuit_map`

**内部流程**（从 GreedyStrategy Stage 1-2 提取）：

1. 构建 LEFT 核心 entries（从 `self.adjacency_table`）
2. 构建 LEFT 电路 entries（从 `circuit_states_shapes`）
3. 构建 MIDDLE Mx entries（从 `measure_shapes`，ndim 决定 batch_symbol）
4. 构建 RIGHT 核心 entries（symmetric 模式：边反转；QCTN 模式：使用 right_qctn 的 adjacency_table）
5. 构建 RIGHT 电路 entries
6. 布线：更新 LEFT/RIGHT 核心的邻居连接
7. 符号分配：通过 `opt_einsum.get_symbol` 为所有边分配 einsum 符号

#### 其他变更

- `__repr__`：从 `adjacency_matrix` 改为使用 `adjacency_table` 的 `input_shape` / `output_shape`
- docstring：移除 `adjacency_matrix` 和 `circuit` 属性说明

---

### 2. `tneq_qc/contractor/row_priority_strategy.py`（新建）

新策略 `RowPriorityStrategy(ContractionStrategy)`，图逻辑委托给 QCTN，只保留收缩逻辑。

| 属性/方法 | 说明 |
|-----------|------|
| `name` | `"row_priority"` |
| `check_compatibility` | 始终返回 `True` |
| `estimate_cost` | 返回 `5e5`（与 GreedyStrategy 相同） |
| `get_compute_function` | 返回 `compute_fn`，内部调用 `qctn.build_symmetric_expansion_graph()` 获取图信息，然后执行逐 qubit 收缩 |

**与 GreedyStrategy 的区别**：

| 方面 | GreedyStrategy | RowPriorityStrategy |
|------|---------------|---------------------|
| 图构建 | 内嵌在 `compute_fn` 中（~400 行） | 委托给 `QCTN.build_symmetric_expansion_graph()` |
| einsum 调用 | 硬编码 `torch.einsum` | 使用 `backend.einsum` |
| torch 依赖 | `compute_fn` 内 `import torch` | 无 torch 导入 |
| 辅助函数 | 类方法（`_find_connected_groups_symmetric` 等） | 模块级函数（`_find_connected_groups` 等） |

**模块级辅助函数**：

- `_find_connected_groups`：Union-Find 连通分组
- `_get_tensor`：从 entry 的 source info 获取实际张量
- `_contract_group`：收缩一个分组内的所有 entries，处理 TNTensor scale 传播
- `_remap_symbols`：将任意符号重映射为标准 opt_einsum 符号
- `_contract_remaining`：收缩剩余 entries 为最终结果

---

### 3. `tneq_qc/contractor/__init__.py`（注册）

- 新增 `from .row_priority_strategy import RowPriorityStrategy`
- 注册到 `balanced` 和 `full` 模式（与 GreedyStrategy 并存）
- 添加到 `__all__`

---

### 4. `tneq_qc/backends/copteinsum.py`（向后兼容）

新增模块级辅助函数：

```python
def _build_circuit_from_adjacency_table(qctn):
    """从 adjacency_table 构建 (input_ranks, adjacency_matrix, output_ranks)。"""
```

所有原来使用 `qctn.circuit` 的方法改为调用此函数。`target_qctn.circuit` 同理。

---

### 5. `tneq_qc/core/engine.py`（清理）

移除 L655 未使用的 `input_ranks, adjacency_matrix, output_ranks = qctn.circuit` 解构赋值。

---

### 6. `tneq_qc/core/engine_common.py`（废弃标记）

4 个方法添加 `DeprecationWarning`：

| 方法 | 说明 |
|------|------|
| `add_pipeline_entry` | Pipeline 系统废弃 |
| `clear_pipeline` | Pipeline 系统废弃 |
| `run_pipeline` | 使用 `contract_with_compiled_strategy` 替代 |
| `run_pipeline_for_gradient` | 使用 `contract_with_compiled_strategy_for_gradient` 替代 |

---

### 7. `tneq_qc/contractor/einsum_strategy.py`（废弃标记）

5 个 legacy 静态方法添加 `.. deprecated::` 文档标记：

- `build_core_only_expression`
- `build_with_inputs_expression`
- `build_with_vector_inputs_expression`
- `build_with_qctn_expression`
- `_build_with_self_expression_legacy`

统一标注使用 `qctn.get_einsum_info()` 替代。

---

### 8. `examples/example_qctn_merge_split.py`（适配）

`_adjacency_to_array` 函数改为接收 `qctn` 对象并从 `adjacency_table` 构建数值邻接矩阵，不再依赖 `qctn.adjacency_matrix`。

---

### 9. `tneq_qc/core/__init__.py`（导出）

新增 `TensorSide` 导出。

---

### 10. `tests/test_row_priority_strategy.py`（新建）

16 个单元测试，分 2 个测试类：

| 测试类 | 测试数 | 覆盖内容 |
|--------|--------|----------|
| `TestBuildSymmetricExpansionGraph` | 9 | 无 circuit/measure 时 entry 数、含 circuit+measure 时 entry 数、L/M/R side 正确性、所有边有 symbol、symmetric 模式边反转、right_qctn=None、3D/4D Mx batch_symbol、多 qubit entry 数 |
| `TestRowPriorityStrategy` | 7 | name、check_compatibility、estimate_cost、**单 qubit 结果 vs GreedyStrategy**、**多 qubit 结果 vs GreedyStrategy**、**条件 measure 结果 vs GreedyStrategy**、策略注册 |

**全部 16 个测试通过**。

---

## 运行测试

```bash
conda run -n py311 python -m pytest tests/test_row_priority_strategy.py -v
```

完整测试套件（Phase 1 + Phase 2 + Phase 2.5）：

```bash
conda run -n py311 python -m pytest tests/test_tn_tensor.py tests/test_qctn_basic.py tests/test_greedy_strategy.py tests/test_row_priority_strategy.py -v
```

**结果：101 passed, 0 failed**

---

## 设计决策记录

### 1. `adjacency_table` 作为唯一图数据结构

`adjacency_matrix`（ncores × ncores 的 NumPy 数组）和 `circuit` 元组被完全移除。所有下游代码通过 `adjacency_table`（list of dicts）获取图信息。对于仍需要 adjacency_matrix 格式的 legacy 代码（`copteinsum.py`），在其内部通过 `_build_circuit_from_adjacency_table()` 按需构建。

### 2. RowPriorityStrategy 与 GreedyStrategy 并存

不修改 GreedyStrategy，创建新的 RowPriorityStrategy。两者在 balanced/full 模式下同时注册，StrategyCompiler 通过 `estimate_cost` 选择最优策略。好处：

- 零风险——原有 GreedyStrategy 不受影响
- 渐进式迁移——验证 RowPriorityStrategy 正确后，将来可替换 GreedyStrategy

### 3. `build_symmetric_expansion_graph` 接收 shapes 而非 tensors

与 `get_einsum_info` 一致的设计：QCTN 方法只接收形状信息（`circuit_states_shapes`、`measure_shapes`），不接触实际张量数据。实际张量的获取由 strategy 的 `_get_tensor` 函数在收缩时完成。

### 4. `backend.einsum` 替代 `torch.einsum`

RowPriorityStrategy 中所有 einsum 调用通过 `backend.einsum` 执行，消除对 PyTorch 的硬编码依赖。这使得 RowPriorityStrategy 在未来可支持 JAX 后端。

### 5. EngineCommon 方法仅标记废弃，不删除

`run_pipeline*` 和 Pipeline 系统添加 `DeprecationWarning` 但保留实现。这是因为 `train.py` 等训练脚本仍在使用这些方法，正式移除需要在 Phase 3 Trainer 完成后进行。

---

## 向后兼容性

- `QCTN.__init__` 签名不变
- `GreedyStrategy` 完全保留，行为不变
- `EinsumStrategy.get_compute_function` 行为不变
- `copteinsum.py` 所有方法通过内部 shim 保持功能一致
- `engine.py` legacy 方法保留（移除了一行未使用的变量）
- 所有现有测试（Phase 1 + Phase 2 + GreedyStrategy）在重构后继续通过

---

## 遗留 TODO

| 编号 | 内容 | 目标阶段 |
|------|------|----------|
| 5.4 | GreedyStrategy 中 `torch.einsum` → `backend.einsum` | 可选（RowPriorityStrategy 已解决此问题） |
| 5.5 | 数据生成方法迁移至 `data_utils` 模块 | 待规划 |
| 5.5 | `engine.py`、`engine_siamese.py` 正式废弃/移除 | Phase 3 后 |
| 5.5 | `contract_with_compiled_strategy_for_gradient` 中 Cross Entropy loss 解耦 | Phase 3 |
