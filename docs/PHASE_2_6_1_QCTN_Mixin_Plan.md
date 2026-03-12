# Phase 2.6.1 计划：QCTN Mixin 拆分

## 目标

将 `tneq_qc/core/qctn.py`（当前 ~1682 行）按职责拆分为 5 个文件，每个文件不超过 400 行。外部调用接口不变。

---

## 拆分方案

```
tneq_qc/core/
├── qctn.py                  # ~250L  主类骨架（保留 __init__ 和对外接口）
├── _qctn_graph.py           # ~350L  图解析 Mixin
├── _qctn_io.py              # ~200L  参数 IO Mixin
├── _qctn_split_merge.py     # ~230L  结构操作 Mixin
└── _qctn_contractor.py      # ~650L  收缩接口 Mixin
```

---

## 各文件职责与迁移内容

### `qctn.py`（主类骨架）

保留内容：
- `TensorSide` 枚举（模块级）
- `QCTN.__init__`
- `QCTN.__repr__`
- `QCTN.from_graph`（classmethod）
- `QCTN.register_module`
- `QCTN.named_cores`
- `QCTN.all_cores`（property）
- `QCTN.define`
- `QCTN.forward`
- `QCTN.conjugate_transpose_cores`
- 继承声明：`class QCTN(QCTNGraphMixin, QCTNIOMixin, QCTNSplitMergeMixin, QCTNContractorMixin)`

### `_qctn_graph.py` → `QCTNGraphMixin`

迁移内容（当前行号）：
- `_circuit_to_adjacency`（L159–L273）
- `_parse_qubit_line`（L552–L584）
- `_rebuild_qubit_line`（L587–L599）
- `_remap_graph`（L602–L624）
- `envolve_from_another_qctn`（L109–L146）

依赖的 `self` 属性：`adjacency_table`、`cores`、`nqubits`、`tn_graph`、`backend`

### `_qctn_io.py` → `QCTNIOMixin`

迁移内容：
- `_init_cores`（L276–L321）
- `set_cores`（L323–L366）
- `_set_single_core`（L372–L397）
- `_set_cores_from_list`（L399–L420）
- `_set_cores_from_dict`（L422–L461）
- `auto_init`（L630–L651）
- `save_cores`（L463–L487）
- `load_cores`（L489–L525）
- `from_pretrained`（L528–L544，staticmethod）

依赖的 `self` 属性：`cores`、`adjacency_table`、`backend`、`cores_weights`、`_submodules`

### `_qctn_split_merge.py` → `QCTNSplitMergeMixin`

迁移内容：
- `chunk`（L657–L760）
- `split`（L762–L769，deprecated wrapper）
- `concat`（L772–L797，staticmethod）
- `_concat_impl`（L800–L878，staticmethod）
- `merge`（L881–L888，deprecated staticmethod）
- `concat_with`（L890–L903）
- `merge_with`（L905–L912，deprecated）

依赖的 `self` 属性：`cores`、`adjacency_table`、`nqubits`、`backend`、`cores_weights`

### `_qctn_contractor.py` → `QCTNContractorMixin`

迁移内容：
- `get_einsum_info`（L1023–L1198）
- `build_core_list`（L1200–L1257）
- `build_symmetric_expansion_graph`（L1263–L1652）

依赖的 `self` 属性：`adjacency_table`、`cores`、`nqubits`、`backend`

---

## 主类最终结构

```python
# qctn.py
from ._qctn_graph import QCTNGraphMixin
from ._qctn_io import QCTNIOMixin
from ._qctn_split_merge import QCTNSplitMergeMixin
from ._qctn_contractor import QCTNContractorMixin

class TensorSide(Enum): ...

class QCTN(QCTNGraphMixin, QCTNIOMixin, QCTNSplitMergeMixin, QCTNContractorMixin):
    def __init__(self, graph=None, backend=None, *, _defer_init=False): ...
    def __repr__(self): ...
    @classmethod
    def from_graph(cls, ...): ...
    def register_module(self, name, module): ...
    def named_cores(self): ...
    @property
    def all_cores(self): ...
    def define(self): ...
    def forward(self, *args, **kwargs): ...
    def conjugate_transpose_cores(self): ...
```

MRO 顺序：`QCTN → QCTNGraphMixin → QCTNIOMixin → QCTNSplitMergeMixin → QCTNContractorMixin → object`

---

## 实施约定

1. **Mixin 不写 `__init__`**，所有状态由 `QCTN.__init__` 初始化
2. **Mixin 文件名以下划线开头**（`_qctn_*.py`），表示内部实现，不对外导出
3. **staticmethod / classmethod 随方法一起迁移**，Mixin 中可以定义 staticmethod
4. **imports 随方法迁移**：各 Mixin 文件只 import 自己用到的模块（`warnings`、`numpy` 等按需分配）
5. **`TensorSide` 留在 `qctn.py`**，`_qctn_contractor.py` 通过相对导入引用：`from .qctn import TensorSide`（或提升到 `_qctn_contractor.py` 本身，`qctn.py` 再 re-export）
6. **外部接口不变**：`from tneq_qc.core.qctn import QCTN, TensorSide` 继续有效

---

## 实施顺序

1. 新建 `_qctn_contractor.py`，迁移 3 个方法（最独立，收益最大）
2. 新建 `_qctn_graph.py`，迁移图解析方法
3. 新建 `_qctn_io.py`，迁移 IO 方法
4. 新建 `_qctn_split_merge.py`，迁移 chunk/concat
5. 精简 `qctn.py` 主类，添加继承声明
6. 运行测试确认全部通过
