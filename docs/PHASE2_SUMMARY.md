# Phase 2 重构总结

## 目标回顾

Phase 2 目标是**重新设计 QCTN**（仿 nn.Module），实现以下子目标：

- **R2-基类**：为 QCTN 添加 `_submodules`、`from_graph()`、`named_cores()`、`all_cores`
- **R5（图解析分离）**：将 `build_core_list`、`get_einsum_info` 从 Contractor 移入 QCTN
- **R2-引用语义**：利用 Phase 1 TNTensor 的 `conj_transpose()` 实现孪生网络参数共享

---

## 变更文件

### 1. `tneq_qc/core/qctn.py`（主要增量）

#### 新增初始化属性

```python
# QCTN.__init__ 末尾
self._submodules: dict = {}   # Phase 2: submodule registry
```

#### 新增方法（Phase 2 模块管理，R2）

| 方法 | 说明 |
|---|---|
| `from_graph(cls, graph_str, backend)` | classmethod：从 ASCII 图字符串创建 QCTN，封装 `__init__` |
| `register_module(name, module)` | 显式注册命名子模块（校验类型、名称不含点） |
| `named_cores(prefix='')` | 递归迭代 `(full_name, tensor)` 对，子模块前缀 `"<mod>."` |
| `all_cores` (property) | 返回所有 core 的扁平 dict（含子模块，键带前缀） |

#### 新增方法（图解析分离，R5）

| 方法 | 说明 |
|---|---|
| `get_einsum_info(circuit_states_shapes, measure_shapes, measure_is_matrix)` | 从 `EinsumStrategy.build_with_self_expression` 迁移；根据 `adjacency_table` 构建 `A·Mx·A†` 的 einsum 方程和形状列表 |
| `build_core_list(cores_dict, circuit_states, measure_matrices)` | 按收缩顺序组装张量列表：`[states, left_cores, Mx, right_cores(reversed), states]` |

#### 新增方法（引用语义，R2-引用语义）

| 方法 | 说明 |
|---|---|
| `conjugate_transpose_cores()` | 返回 `{name: tn_tensor.conj_transpose()}` dict；所有值为零拷贝引用（`is_ref=True, is_transposed=True`） |

---

### 2. `tneq_qc/contractor/einsum_strategy.py`（R5 解耦）

#### `get_compute_function` 重构

| 修改 | 说明 |
|---|---|
| `build_with_self_expression(...)` → `qctn.get_einsum_info(...)` | 图解析委托给 QCTN，contractor 只负责编译 |
| 张量组装内联代码 → `qctn.build_core_list(...)` | 张量排序委托给 QCTN |

#### 向后兼容

- `build_with_self_expression` 改为 wrapper，委托给 `qctn.get_einsum_info()`，签名不变
- 原实现保留为 `_build_with_self_expression_legacy`，供对比验证

---

### 3. `tests/test_qctn_basic.py`（新建）

35 个单元测试，覆盖全部新功能，分 6 个测试类：

| 测试类 | 测试数 | 覆盖内容 |
|---|---|---|
| `TestFromGraph` | 4 | classmethod 创建、核心名称检测、backend 存储 |
| `TestSubmodules` | 5 | 初始化、register_module 正常/类型错误/名称错误 |
| `TestNamedCores` | 4 | 自有 cores 迭代、子模块前缀、嵌套前缀传递 |
| `TestAllCores` | 3 | 返回 dict、自有/子模块 cores 均可访问 |
| `TestGetEinsumInfo` | 6 | 无/有 states、有 measure、与 legacy 结果对比、两核 |
| `TestBuildCoreList` | 5 | 无/有 states+measure、默认 cores_weights、左右共享引用、与 einsum_info count 一致 |
| `TestConjugateTransposeCores` | 6 | keys 一致、TNTensor 类型、is_ref、is_transposed、scale 保持、raw tensor 输入 |
| `TestEinsumStrategyDelegation` | 2 | 委托调用一致性、compute_fn 可执行 |

**全部 35 个测试通过**（`pytest tests/test_qctn_basic.py`）。

---

## 运行测试

```bash
conda run -n py311 python -m pytest tests/test_qctn_basic.py -v
```

完整测试套件（Phase 1 + Phase 2）：

```bash
conda run -n py311 python -m pytest tests/test_tn_tensor.py tests/test_tn_graph.py tests/test_greedy_strategy.py tests/test_qctn_basic.py -v
```

---

## 设计决策记录

### 1. 保留 `self.cores`（list）不重命名

原 `QCTN.cores` 是一个核心名称列表，被 `engine.py`、`contractor/` 等大量代码引用（`for name in qctn.cores`）。重命名会导致大范围破坏。因此：
- 保留 `self.cores: list[str]` 不变
- 新增 `all_cores` property（返回 Dict）用于模块化 API
- `named_cores()` iterator 补充带前缀的枚举能力

### 2. `get_einsum_info` 默认 `measure_is_matrix=True`

原 `build_with_self_expression` 默认 `measure_is_matrix=False`，但实际所有调用都传 `True`。迁移时统一为 `True`，避免混淆。`_build_with_self_expression_legacy` 保留原默认值供参考。

### 3. 每个 qubit 对应一个 Mx 矩阵

`get_einsum_info` 的 middle block 每个 output symbol 生成一个收缩项。因此：
- `nqubits` 个 qubit 的 QCTN → 需要 `nqubits` 个 measure 矩阵
- `measure_shapes` 应为长度等于 qubit output 数量的 tuple of tuples

### 4. `conjugate_transpose_cores()` 不自动应用于 `build_core_list`

`build_core_list` 的 right cores 使用与 left cores **相同的张量对象**（共享参数）。einsum 方程本身通过不同的指标赋值处理共轭转置，不需要物理转置张量。`conjugate_transpose_cores()` 是为**显式构造 right QCTN** 场景提供的工具方法。

---

## 向后兼容性

- `QCTN.__init__` 签名不变；仅在末尾新增 `self._submodules = {}`
- `EinsumStrategy.build_with_self_expression` 签名不变，行为不变（wrapper 委托）
- `EinsumStrategy.get_compute_function` 签名不变，结果等价
- 所有现有测试（Phase 1 + greedy/tn_graph）在重构后继续通过
