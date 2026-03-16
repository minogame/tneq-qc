# Phase 2.6 重构计划：QCTN 模块化与小模块/应用模块体系

> 接续 Phase 2.5（Engine/Contractor 清理）；聚焦 QCTN 的模块化设计，使其既能定义
> 独立的小模块，又能通过组合构建应用级复杂模块。

---

## 背景与动机

Phase 2.5 完成后，QCTN 已有：

- 单一图字符串初始化 + `adjacency_table` 作为唯一图数据结构
- `_submodules` 注册机制（`register_module` / `named_cores`）
- `split` / `merge` / `merge_with` 用于结构操作
- `build_core_list` / `get_einsum_info` / `build_symmetric_expansion_graph` 供 Contractor 使用

**现有问题**：

1. Circuit 和 Mx 以外部参数形式传入 Contractor，没有统一为 QCTN 子类
2. `split` / `merge` 名称与业界惯例（chunk / concat）不一致
3. Core tensor 的声明和初始化混合在 `__init__` 中，无法分离"定义结构"与"初始化权重"
4. 缺少面向应用的标准组合：编码、TNEQ、二次型等
5. 测试尚未覆盖小模块 + 应用模块的端到端场景

---

## 目标一览

| 编号 | 子目标 | 优先级 |
|------|--------|--------|
| 2.6.1 | 定义三种小模块类 (`MPS`, `CircuitState`, `MeasureMatrix`) | 高 |
| 2.6.2 | 定义六种应用模块类 | 高 |
| 2.6.3 | `split` → `chunk`，`merge` → `concat` | 中 |
| 2.6.4 | 删除函数签名中的 `circuit` / `mx` 显式参数 | 中 |
| 2.6.5 | 分离 core tensor 声明与初始化 | 高 |
| 2.6.6 | 实现新测试文件 | 高 |

---

## 2.6.1 小模块类定义

所有小模块继承自 `QCTN`，覆盖 `define()` 并在 `__init__` 中完成拓扑声明；权重初始化延迟到 `auto_init()` 或外部 `set_cores()`。

### MPS（矩阵乘积态）

```
图形示意（3 qubit，bond_dim=4，phys_dim=2）：
  -2-A-4-B-4-C-2-
  -2-A-4-B-4-C-2-
  -2-A-4-B-4-C-2-
```

```python
class MPS(QCTN):
    """矩阵乘积态。可通过 QCTNHelper 生成图字符串初始化。

    Args:
        nqubits: qubit 数量
        bond_dim: 内部 bond 维度
        phys_dim: 物理（输出）维度，默认 2
        backend: 计算后端
    """
    def __init__(self, nqubits, bond_dim, phys_dim=2, backend=None):
        graph = QCTNHelper.mps(nqubits, bond_dim, phys_dim)
        super().__init__(graph, backend=backend)
```

### CircuitState（线路态，向量）

每个 qubit 对应一个 core tensor，只有输出维度（物理维），没有左侧输入维度。

```
图形示意（3 qubit，phys_dim=2）：
  -A-2-
  -B-2-
  -C-2-
相当于每行只有一个 core，左端无维度（或维度为 1）
```

```python
class CircuitState(QCTN):
    """线路态（量子电路输出向量）。

    Args:
        nqubits: qubit 数量
        phys_dim: 输出物理维度，默认 2
        backend: 计算后端
    """
    def __init__(self, nqubits, phys_dim=2, backend=None):
        graph = QCTNHelper.circuit_state(nqubits, phys_dim)
        super().__init__(graph, backend=backend)
```

### MeasureMatrix（测量矩阵 Mx）

每个 qubit 对应一个 core tensor，左侧有输入维度，右侧有输出维度。

```
图形示意（3 qubit，phys_dim=2）：
  -2-A-2-
  -2-B-2-
  -2-C-2-
```

```python
class MeasureMatrix(QCTN):
    """测量矩阵（每个 qubit 一个矩阵算符）。

    Args:
        nqubits: qubit 数量
        phys_dim: 输入/输出物理维度，默认 2
        backend: 计算后端
    """
    def __init__(self, nqubits, phys_dim=2, backend=None):
        graph = QCTNHelper.measure_matrix(nqubits, phys_dim)
        super().__init__(graph, backend=backend)
```

> **新增 QCTNHelper 方法**：`mps()`（已有）、`circuit_state()`（新增）、`measure_matrix()`（新增）

---

## 2.6.2 应用模块类定义

应用模块通过 `register_module` 将小模块组合成复合 QCTN，并重写 `forward()` 以实现具体计算逻辑。

### 1. PlainMPS — 普通 MPS 作为完整应用

```python
class PlainMPS(QCTN):
    """MPS 作为独立应用模块（等同于 MPS 本身，但包装为应用层语义）。"""
    def __init__(self, nqubits, bond_dim, phys_dim=2, backend=None):
        super().__init__.__new_as_composite__(backend)  # 不传 graph
        self.register_module("mps", MPS(nqubits, bond_dim, phys_dim, backend))

    def forward(self, strategy, engine):
        return engine.contract(self.mps, strategy=strategy)
```

### 2. TransposeMPS — MPS 的转置

```python
class TransposeMPS(QCTN):
    """MPS 转置：core tensors 共享引用（is_transposed=True）。"""
    def __init__(self, source_mps: MPS):
        super().__init__.__new_as_composite__(source_mps.backend)
        self.register_module("mps", source_mps)
        # forward 中使用 source_mps.cores_weights 的转置引用（TNTensor.transpose()）
```

### 3. MPS_with_Ref — MPS + MPS 引用

```python
class MPS_with_Ref(QCTN):
    """两个 MPS，其中 right 侧是 left 侧的参数引用（共享底层张量）。

    用于孪生/对称网络场景。
    """
    def __init__(self, nqubits, bond_dim, phys_dim=2, backend=None):
        super().__init__.__new_as_composite__(backend)
        self.register_module("left", MPS(nqubits, bond_dim, phys_dim, backend))
        # right 的 cores 是 left 的 TNTensor 引用（is_ref=True）
        self.register_module("right", ...)
```

### 4. Encoding — 编码：CircuitState + MPS

```python
class Encoding(QCTN):
    """编码网络：circuit state（输入向量）卷积到 MPS。

    图拓扑：CircuitState 输出接 MPS 的物理维输入。
    """
    def __init__(self, nqubits, bond_dim, phys_dim=2, backend=None):
        super().__init__.__new_as_composite__(backend)
        self.register_module("circuit", CircuitState(nqubits, phys_dim, backend))
        self.register_module("mps", MPS(nqubits, bond_dim, phys_dim, backend))
```

### 5. TNEQ — 两个独立 MPS

```python
class TNEQ(QCTN):
    """TNEQ 模型：mps_1 与 mps_2 两个独立 MPS 的内积。

    图结构：mps_1 和 mps_2 通过物理维相连（不共享参数）。
    """
    def __init__(self, nqubits, bond_dim, phys_dim=2, backend=None):
        super().__init__.__new_as_composite__(backend)
        self.register_module("mps1", MPS(nqubits, bond_dim, phys_dim, backend))
        self.register_module("mps2", MPS(nqubits, bond_dim, phys_dim, backend))
```

### 6. Quadratic — 二次型：circuit + mps + mx + mps† + circuit†

```python
class Quadratic(QCTN):
    """量子二次型：<circuit | mps† · mx · mps | circuit>

    组件顺序（从外到内）：
        circuit_state (ket) → mps (左) → measure_matrix (中) → mps† (右) → circuit_state† (bra)
    """
    def __init__(self, nqubits, bond_dim, phys_dim=2, backend=None):
        super().__init__.__new_as_composite__(backend)
        self.register_module("circuit", CircuitState(nqubits, phys_dim, backend))
        self.register_module("mps", MPS(nqubits, bond_dim, phys_dim, backend))
        self.register_module("mx", MeasureMatrix(nqubits, phys_dim, backend))
        # mps_T 和 circuit_T 为引用（共享参数，is_ref=True, is_transposed=True）
```

---

## 2.6.3 重命名：split/merge → chunk/concat

### 变更清单

| 旧名称 | 新名称 | 说明 |
|--------|--------|------|
| `QCTN.split(split_idx)` | `QCTN.chunk(split_idx)` | 按 core 索引拆分为两个 QCTN |
| `QCTN.merge(qctn1, qctn2)` (static) | `QCTN.concat(qctn1, qctn2)` (static) | 左右合并两个 QCTN |
| `QCTN.merge_with(other)` | `QCTN.concat_with(other)` | 实例方法版 concat |

**向后兼容**：旧方法保留，加 `DeprecationWarning`，重定向到新方法：

```python
def split(self, split_idx=None):
    warnings.warn("split() is deprecated, use chunk() instead.", DeprecationWarning, stacklevel=2)
    return self.chunk(split_idx)
```

---

## 2.6.4 删除函数签名中的 circuit/mx 显式参数

### 扫描范围

- `QCTN.build_core_list(cores_dict, circuit_states, measure_matrices)`
- `QCTN.get_einsum_info(circuit_states_shapes, measure_shapes, right_qctn, ...)`
- `QCTN.build_symmetric_expansion_graph(circuit_states_shapes, measure_shapes, right_qctn)`

### 处理策略

上述方法中 `circuit_states` 和 `measure_matrices`（或其 shapes 版本）是合法的"外部数据"参数，不应删除——它们是收缩时注入的张量，不是结构定义时的依赖。

**真正需要删除的**：

1. `__init__` 中的 `self.circuit_states = None` 和 `self.measure_matrices = None`（实例属性），这些是遗留的存储位置，Phase 2.6 中 circuit/mx 已经是独立 QCTN 子类，不应再挂在父 QCTN 实例上。

2. 检查所有 `QCTN` 方法，若有通过 `self.circuit_states` / `self.measure_matrices` 访问的逻辑（而非参数传入），将其删除或替换为参数化形式。

3. `copteinsum.py` 中的 `_build_circuit_from_adjacency_table`：这是 legacy shim，Phase 2.6 中如不再需要可移除，否则保留不变。

---

## 2.6.5 分离 core tensor 声明与初始化

### 当前问题

`__init__` 中调用 `_init_cores()` 会立即执行正交随机初始化，用户无法先声明结构再选择初始化方式。

### 新设计

```python
class QCTN:
    def __init__(self, graph, backend=None):
        # ... 解析图，构建 adjacency_table，cores 列表 ...
        self.cores_weights = {}        # 只声明，不初始化
        # 不再调用 _init_cores()

    def auto_init(self, dtype=None, device=None):
        """自动正交随机初始化所有 core tensors（原 _init_cores 逻辑）。

        Args:
            dtype: 可选，覆盖默认 dtype（如 torch.complex64）
            device: 可选，覆盖默认 device（如 'cpu'）

        Returns:
            self（支持链式调用）
        """
        for core_info in self.adjacency_table:
            # 原 _init_cores 逻辑
            ...
        return self

    def set_cores(self, cores, strict=True):
        """从外部传入已有 tensor 初始化 cores（现有方法，保持不变）。"""
        ...
```

**用法示例**：

```python
# 自动初始化
mps = MPS(3, 4).auto_init()

# 从外部传入
mps = MPS(3, 4)
mps.set_cores({"A": tensor_a, "B": tensor_b, "C": tensor_c})

# 复合模块逐层初始化
model = Quadratic(3, 4)
model.mps.auto_init()
model.mx.set_cores(my_mx_tensors)
model.circuit.auto_init()
```

**向后兼容**：

- 现有直接使用 `QCTN(graph)` 的代码：初始化后 `cores_weights` 为空 dict，访问会得到 KeyError
- 解决方案：在测试和示例中将 `QCTN(graph)` 改为 `QCTN(graph).auto_init()`，已有测试文件统一更新

> 注：`_init_cores` 私有方法保留但不再在 `__init__` 中调用；`auto_init` 内部调用 `_init_cores`。

---

## 2.6.6 新测试文件

### 文件：`tests/test_qctn_modules.py`

**默认配置**：
- backend: `BackendPyTorch`（cpu）
- dtype: `torch.complex64`（复数）
- 可通过 `pytest.fixture` 切换

**测试分组**：

#### A. 小模块初始化测试

| 测试名 | 内容 |
|--------|------|
| `test_mps_init` | MPS 初始化后 cores 为空；`auto_init()` 后 core 形状正确 |
| `test_circuit_state_init` | CircuitState 每 qubit 一个 core，输出维度为 phys_dim |
| `test_measure_matrix_init` | MeasureMatrix 每 qubit 一个 core，输入=输出=phys_dim |
| `test_auto_init_dtype` | `auto_init(dtype=complex64)` 生成复数 core tensor |
| `test_set_cores_external` | `set_cores(dict)` 正确覆盖 core weights |

#### B. 应用模块初始化测试

| 测试名 | 内容 |
|--------|------|
| `test_plain_mps_init` | PlainMPS 可正常构造，`all_cores` 包含正确数量的 core |
| `test_encoding_init` | Encoding 含 circuit + mps 子模块，`named_cores` 正确 |
| `test_tneq_init` | TNEQ 含 mps1 + mps2，两者独立（修改 mps1 不影响 mps2） |
| `test_quadratic_init` | Quadratic 含 circuit、mps、mx 三个子模块 |

#### C. 小模块嵌套组合为应用模块测试

| 测试名 | 内容 |
|--------|------|
| `test_encoding_from_small_modules` | 手动构造 CircuitState + MPS，通过 register_module 组合为 Encoding 等价体 |
| `test_quadratic_from_small_modules` | 手动组合所有小模块，验证 all_cores 数量 |

#### D. 转置与引用语义测试

| 测试名 | 内容 |
|--------|------|
| `test_transpose_mps_shares_params` | TransposeMPS 的 core 是原 MPS 的引用；修改原 core → TransposeMPS 的 core 同步变化 |
| `test_ref_mps_independent_of_other` | MPS_with_Ref 的 left/right 若 right 是 left 的引用，修改 left.A 后 right.A 一致 |
| `test_tneq_two_independent_mps` | TNEQ 的 mps1 和 mps2 不共享参数；修改 mps1.A 后 mps2.A 不变 |

#### E. Chunk / Concat 测试

| 测试名 | 内容 |
|--------|------|
| `test_chunk_basic` | `chunk()` 返回两个 QCTN，core 数量之和等于原 QCTN |
| `test_concat_basic` | `concat(q1, q2)` 合并后 core 数量为 q1.ncores + q2.ncores |
| `test_split_deprecated_warning` | 调用 `split()` 触发 DeprecationWarning |
| `test_merge_deprecated_warning` | 调用 `merge()` 触发 DeprecationWarning |

#### F. Contract 基本功能测试

| 测试名 | 内容 |
|--------|------|
| `test_mps_contract_shape` | MPS auto_init 后，通过 EinsumStrategy 收缩，输出形状正确 |
| `test_encoding_contract` | Encoding 收缩：circuit_state 卷积到 mps，输出标量或向量 |
| `test_quadratic_contract_real` | Quadratic 二次型收缩，结果为实数（因 Hermitian Mx） |

---

## 执行顺序

```
Step 1: QCTNHelper 新增 circuit_state() 和 measure_matrix() 图生成函数
Step 2: 分离 QCTN.__init__ 中的 core 声明与初始化（auto_init 方法）
Step 3: 实现 MPS / CircuitState / MeasureMatrix 小模块类
Step 4: 重命名 split→chunk，merge→concat（旧接口加 DeprecationWarning）
Step 5: 删除 QCTN 实例属性 self.circuit_states / self.measure_matrices
Step 6: 实现应用模块类（PlainMPS、TransposeMPS、MPS_with_Ref、Encoding、TNEQ、Quadratic）
Step 7: 实现 tests/test_qctn_modules.py，全部测试通过
Step 8: 更新现有测试以适配 auto_init()（QCTN(graph) → QCTN(graph).auto_init()）
```

---

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `tneq_qc/core/qctn.py` | 修改 | `auto_init()`、`chunk()`、`concat()`、删除 `self.circuit_states/measure_matrices` |
| `tneq_qc/utils/graph_generators.py` | 修改 | 新增 `circuit_state()` 和 `measure_matrix()` |
| `tneq_qc/modules/__init__.py` | 新建 | 导出所有小模块和应用模块 |
| `tneq_qc/modules/small.py` | 新建 | `MPS`、`CircuitState`、`MeasureMatrix` |
| `tneq_qc/modules/app.py` | 新建 | `PlainMPS`、`TransposeMPS`、`MPS_with_Ref`、`Encoding`、`TNEQ`、`Quadratic` |
| `tests/test_qctn_modules.py` | 新建 | 完整模块测试 |
| `tests/test_qctn_basic.py` | 修改 | 旧测试适配 `auto_init()` |

---

## 关键设计决策

### 1. auto_init 不在 `__init__` 中调用

优点：用户可以先声明 MPS(3, 4)，再决定用随机初始化还是从检查点加载。与 PyTorch 的 `nn.Module` 模型加载 (`load_state_dict`) 风格一致。

### 2. 应用模块使用 register_module 而非直接继承图字符串

应用模块（Encoding、TNEQ、Quadratic）的图拓扑是多个独立子图的组合，无法用单一 ASCII 图字符串表达。使用 `register_module` 保持模块化。

### 3. 引用语义依赖 TNTensor（R1 已规划）

`TransposeMPS` 和 `MPS_with_Ref` 中的 `is_ref=True` 语义依赖 TNTensor 的引用实现。Phase 2.6 中若 TNTensor 引用尚未完成，可用 Python 层的对象引用（共享同一 tensor 对象）临时实现，后续升级为 TNTensor 引用。

### 4. chunk/concat 命名

与 PyTorch (`torch.chunk`, `torch.cat`) 惯例一致，比 split/merge 更清晰地描述操作语义（chunk = 切块，concat = 拼接）。

---

## 向后兼容性

| 变更 | 影响 | 处理 |
|------|------|------|
| `__init__` 不再调用 `_init_cores` | 现有 `QCTN(graph)` 的 `cores_weights` 为空 | 测试/示例改为 `QCTN(graph).auto_init()` |
| `split` → `chunk` | 调用 `split` 的代码触发警告但不报错 | DeprecationWarning，Phase 3 后移除 |
| `merge` → `concat` | 同上 | 同上 |
| 删除 `self.circuit_states` / `self.measure_matrices` | 直接访问这两个属性的代码报 AttributeError | 扫描全库，确认无外部使用 |
