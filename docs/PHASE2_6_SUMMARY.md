# Phase 2.6 重构总结

## 目标回顾

Phase 2.6 目标是在 Phase 2（QCTN 图解析/收缩分离）完成后，完成 **QCTN 模块化与小/应用模块体系建设**。实现以下子目标：

- **6.1**：为 `QCTNHelper` 添加 `mps`、`circuit_state`、`measure_matrix` 图生成方法
- **6.2**：分离 core tensor 声明与初始化——新增 `auto_init()` 方法，支持延迟初始化
- **6.3**：支持 composite 模式（`graph=None`）——QCTN 作为纯容器聚合子模块
- **6.4**：实现小模块 `MPS`、`CircuitState`、`MeasureMatrix`（`tneq_qc/modules/small.py`）
- **6.5**：实现应用模块 `PlainMPS`、`TransposeMPS`、`MPS_with_Ref`、`Encoding`、`TNEQ`、`Quadratic`（`tneq_qc/modules/app.py`）
- **6.6**：将 `split` → `chunk`、`merge` → `concat`，旧接口加 `DeprecationWarning`
- **6.7**：移除 `self.circuit_states` / `self.measure_matrices` 实例属性，向后兼容通过 `getattr` 保持
- **6.8**：新增 `tests/test_qctn_modules.py`（40 个测试，默认 complex64 + torch + cpu）

---

## 变更文件

### 1. `tneq_qc/utils/graph_generators.py`（新增静态方法）

为 `QCTNHelper` 新增三个图字符串生成方法：

| 方法 | 说明 | 示例输出（nqubits=3） |
|------|------|----------------------|
| `mps(nqubits, bond_dim, phys_dim=2)` | 均匀 MPS：所有 qubit 经过全部 core，bond_dim 连接相邻 core | `-2-A-4-B-4-C-2-`（每行相同） |
| `circuit_state(nqubits, phys_dim=2)` | 线路态：每 qubit 一个 core，无左侧输入维 | `-A-2-\n-B-2-\n-C-2-` |
| `measure_matrix(nqubits, phys_dim=2)` | 测量矩阵：每 qubit 一个 core，左右各一个 phys_dim | `-2-A-2-\n-2-B-2-\n-2-C-2-` |

这三个方法原来作为嵌套函数散落在 `generate_example_graph()` 内部，现提升为公开静态方法，供模块类直接调用。

---

### 2. `tneq_qc/core/qctn.py`（多处修改）

#### 2.1 `__init__` 签名变更

```python
# 旧签名
def __init__(self, graph, backend=None):

# 新签名
def __init__(self, graph=None, backend=None, *, _defer_init=False):
```

- `graph=None`：触发 composite 模式，返回空容器（不解析图）
- `_defer_init=True`：跳过 `_init_cores()`，等待 `auto_init()` 调用

**Composite 模式（graph=None）初始化内容**：

```python
self.qubits = []; self.nqubits = 0; self.qubit_indices = []
self.graph = None; self.tn_graph = None
self.cores = []; self.ncores = 0; self.adjacency_table = []
self.backend = backend; self._loaded_metadata = None
self.cores_weights = {}; self._submodules = {}
```

#### 2.2 移除的实例属性

| 属性 | 原始值 | 移除原因 |
|------|--------|----------|
| `self.circuit_states` | `None` | 语义模糊，改为运行时通过参数传入 |
| `self.measure_matrices` | `None` | 同上 |

**向后兼容**：`build_symmetric_expansion_graph()` 中改用 `getattr(self, 'circuit_states', None)` 和 `getattr(self, 'measure_matrices', None)`，测试中动态设置这两个属性的代码继续工作。

#### 2.3 新增 `auto_init()` 方法

```python
def auto_init(self, dtype=None, device=None) -> "QCTN":
    """随机正交初始化所有 core tensors，返回 self 支持链式调用。"""
    if self.graph is not None:
        self._init_cores()
    for sub in self._submodules.values():
        sub.auto_init(dtype=dtype, device=device)
    return self
```

- 递归初始化所有子模块
- 返回 `self`，支持 `.auto_init()` 链式调用

#### 2.4 `_init_cores()` 修复（非方形 core 支持）

原始实现 `init_random_core([input_dim, output_dim])` 要求 `input_dim == output_dim`（QR 分解生成正交方阵）。MPS 的 core 通常为非方形（如 `input_dim=8, output_dim=64`），会触发 reshape 错误。

**修复方案**：

```python
if input_dim == output_dim:
    core = self.backend.init_random_core([input_dim, output_dim])
    core = self.backend.reshape(core, full_shape)
else:
    max_dim = max(input_dim, output_dim)
    core = self.backend.init_random_core([max_dim, max_dim])
    raw = core.tensor if isinstance(core, TNTensor) else core
    raw_sliced = raw[:input_dim, :output_dim].contiguous()
    core = self.backend.reshape(
        self.backend.wrap_tensor(raw_sliced) if isinstance(core, TNTensor) else raw_sliced,
        full_shape,
    )
```

生成 `max_dim × max_dim` 正交矩阵后切片至所需形状，保留近似正交性。

#### 2.5 `chunk` / `concat`（原 `split` / `merge`）

| 新方法 | 类型 | 说明 |
|--------|------|------|
| `chunk(split_idx=None)` | 实例方法 | 将 QCTN 按 core 切分为两个子 QCTN（原 `split` 的实现） |
| `concat(qctn1, qctn2)` | 静态方法 | 水平拼接两个 QCTN（原 `merge` 的实现，移入 `_concat_impl`） |
| `concat_with(other)` | 实例方法 | `QCTN.concat(self, other)` 的实例方法版本 |

旧接口保留并加 `DeprecationWarning`：

| 废弃方法 | 重定向至 | 警告信息 |
|----------|----------|---------|
| `split()` | `chunk()` | `"...use QCTN.chunk() instead."` |
| `merge()` | `concat()` | `"...use QCTN.concat() instead."` |
| `merge_with()` | `concat_with()` | `"...use QCTN.concat_with() instead."` |

---

### 3. `tneq_qc/modules/small.py`（新建）

小模块（叶节点）实现。所有类继承 `QCTN`，以 `_defer_init=True` 构建（不在 `__init__` 中调用 `_init_cores()`）。

| 类 | 图结构 | 核心参数 |
|----|--------|---------|
| `MPS(nqubits, bond_dim, phys_dim=2, backend=None)` | `QCTNHelper.mps(nqubits, bond_dim, phys_dim)` | `nqubits`、`bond_dim`、`phys_dim` |
| `CircuitState(nqubits, phys_dim=2, backend=None)` | `QCTNHelper.circuit_state(nqubits, phys_dim)` | `nqubits`、`phys_dim` |
| `MeasureMatrix(nqubits, phys_dim=2, backend=None)` | `QCTNHelper.measure_matrix(nqubits, phys_dim)` | `nqubits`、`phys_dim` |

**使用示例**：

```python
mps = MPS(nqubits=3, bond_dim=4, phys_dim=2, backend=backend).auto_init()
cs  = CircuitState(nqubits=3, phys_dim=2, backend=backend).auto_init()
mx  = MeasureMatrix(nqubits=3, phys_dim=2, backend=backend).auto_init()
```

---

### 4. `tneq_qc/modules/app.py`（新建）

应用模块（复合节点）实现。所有类以 `graph=None, _defer_init=True` 初始化，通过 `register_module()` 聚合小模块。

| 类 | 子模块 | `all_cores` 数（nqubits=3） | 说明 |
|----|--------|----------------------------|------|
| `PlainMPS(nqubits, bond_dim, phys_dim=2, backend=None)` | `"mps"`: MPS | 3 | 最简 MPS 应用 |
| `TransposeMPS(source_mps)` | 无（引用 source） | 同 source | 共轭转置引用视图，`named_cores()` 动态生成 |
| `MPS_with_Ref(nqubits, bond_dim, phys_dim=2, backend=None)` | `"left"`: MPS, `"right"`: TransposeMPS-like | 6 | right 的 core 是 left 的 `conj_transpose()` 引用 |
| `Encoding(nqubits, bond_dim, phys_dim=2, backend=None)` | `"circuit"`: CircuitState, `"mps"`: MPS | 6 | 线路 + MPS 编码 |
| `TNEQ(nqubits, bond_dim, phys_dim=2, backend=None)` | `"mps1"`: MPS, `"mps2"`: MPS | 6 | 两个独立 MPS 的内积 |
| `Quadratic(nqubits, bond_dim, phys_dim=2, backend=None)` | `"circuit"`: CircuitState, `"mps"`: MPS, `"mx"`: MeasureMatrix | 9 | 二次型：circuit + mps + mx |

**`TransposeMPS` 引用语义**：

`TransposeMPS` 不注册子模块，而是保存 `_source_mps` 引用，重写 `named_cores()` 以动态调用 `conj_transpose()`：

```python
def named_cores(self):
    for name, tensor in self._source_mps.named_cores():
        if isinstance(tensor, TNTensor):
            yield name, tensor.conj_transpose()
        else:
            yield name, TNTensor(tensor).conj_transpose()
```

每次调用都返回最新的视图，源 MPS 参数更新后自动可见。

**`MPS_with_Ref` 参数共享**：

`auto_init()` 中先初始化 `left`，再将 `right` 的每个 core 设置为 `left` 对应 core 的 `conj_transpose()` 引用（`is_ref=True`, `is_transposed=True`）：

```python
for name in left.cores:
    tensor = left.cores_weights[name]
    right.cores_weights[name] = tensor.conj_transpose()  # 共享底层存储
```

---

### 5. `tneq_qc/modules/__init__.py`（新建）

包初始化，导出所有小模块和应用模块：

```python
from .small import MPS, CircuitState, MeasureMatrix
from .app import PlainMPS, TransposeMPS, MPS_with_Ref, Encoding, TNEQ, Quadratic
```

---

### 6. `tests/test_qctn_modules.py`（新建）

40 个单元测试，分 6 个测试类：

| 测试类 | 测试数 | 覆盖内容 |
|--------|--------|---------|
| `TestSmallModuleInit` | 10 | auto_init 前后 cores_weights 状态、返回 self、core 形状非空、dtype=complex、set_cores 外部注入 |
| `TestAppModuleInit` | 9 | 子模块注册、named_cores 前缀、all_cores 数量、auto_init 递归传播 |
| `TestComposition` | 2 | 手动组合 vs Encoding/Quadratic 核心数一致性 |
| `TestTransposeAndRef` | 5 | is_ref/is_transposed 标志、source 共享、TNEQ 独立存储、in-place 修改可见性 |
| `TestChunkConcat` | 8 | chunk/concat 返回类型和核心数、split/merge/merge_with 产生 DeprecationWarning |
| `TestContractBasic` | 2 | MPS via EinsumStrategy 收缩不报错、结果非空 |

**默认 fixture**：

```python
@pytest.fixture(scope="module")
def backend():
    return BackendFactory.create_backend("pytorch", device="cpu", dtype="complex64")
```

---

### 7. `tests/test_row_priority_strategy.py`（兼容性修复）

`test_qctn_has_circuit_and_measure_attrs` 测试改用 `getattr` 访问（因 `self.circuit_states` / `self.measure_matrices` 已不在 `__init__` 中初始化）：

```python
# 旧写法（失败）
assert qctn_2bit.circuit_states is None or isinstance(qctn_2bit.circuit_states, list)

# 新写法
cs = getattr(qctn_2bit, 'circuit_states', None)
assert cs is None or isinstance(cs, list)
```

---

## 运行测试

新增测试：

```bash
conda run -n py311 python -m pytest tests/test_qctn_modules.py -v
```

完整测试套件（Phase 1 + Phase 2 + Phase 2.5 + Phase 2.6）：

```bash
conda run -n py311 python -m pytest tests/test_tn_tensor.py tests/test_qctn_basic.py tests/test_greedy_strategy.py tests/test_row_priority_strategy.py tests/test_qctn_modules.py -q
```

**结果：144 passed, 0 failed**

---

## 设计决策记录

### 1. `_defer_init=True` 而非子类重写 `_init_cores`

小模块不重写 `_init_cores()` 的逻辑，只在构造时传入 `_defer_init=True` 跳过自动调用。这样保持 `_init_cores()` 单一实现，避免多处维护。用户通过 `auto_init()` 显式触发初始化。

### 2. Composite 模式（`graph=None`）优先于继承

应用模块不通过复杂继承链组合子结构，而是以 `graph=None` 创建空容器，通过 `register_module()` 动态注册子模块。这与 PyTorch `nn.Module` 的 `__init__` 风格一致，易于扩展。

### 3. `TransposeMPS` 动态 `named_cores()` 而非快照

`TransposeMPS` 每次调用 `named_cores()` 时实时调用 `conj_transpose()`，而非在 `auto_init()` 时存储快照。这确保源 MPS 参数更新后引用侧自动同步，无需额外同步机制。

### 4. 非方形 core 初始化的切片方案

`init_random_core([m,n])` 的 QR 实现要求方阵。对于非方形 core（MPS 两端 core 典型情况），生成 `max(m,n) × max(m,n)` 的正交矩阵后切片至 `[m, n]`。切片结果保留近似正交性，适合作为随机初始值。

### 5. `getattr` 向后兼容而非保留属性初始化

`self.circuit_states = None` 和 `self.measure_matrices = None` 已从 `__init__` 移除，但 `build_symmetric_expansion_graph()` 中改用 `getattr(..., None)` 读取。测试中动态设置这两个属性的代码无需修改，行为不变。

---

## 向后兼容性

| 变更 | 影响 | 处理方式 |
|------|------|---------|
| `QCTN.__init__` 新增 `graph=None` 和 `_defer_init` | 原有调用 `QCTN(graph_str, backend)` 不受影响 | 默认值兼容 |
| 移除 `self.circuit_states` / `self.measure_matrices` | 直接属性访问失败（`AttributeError`） | 改用 `getattr` |
| `split()` / `merge()` / `merge_with()` 保留 | 发出 `DeprecationWarning` | 重定向至新接口 |
| `_init_cores()` 内部修改 | 非方形 core 原来会抛 `RuntimeError`，现正常初始化 | 仅增量修改，方形 core 路径不变 |
| 所有现有测试（Phase 1 + Phase 2 + Phase 2.5）继续通过 | — | — |

---

## 遗留 TODO

| 编号 | 内容 | 目标阶段 |
|------|------|---------|
| 6.9 | `modules` 包加入顶层 `tneq_qc/__init__.py` 导出 | Phase 3 前 |
| 6.10 | `TransposeMPS` 和 `MPS_with_Ref` 接入 Contractor（`build_core_list` / `get_einsum_info`）| Phase 3 |
| 6.11 | `Quadratic` 的完整收缩测试（circuit + mps + mx 联合） | Phase 3 |
| 6.12 | `auto_init` 支持 `dtype` / `device` 透传至 backend | Phase 3 |
