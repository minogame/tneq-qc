# tneq-qc 高级用法：四种训练场景实现分析

本文档分析四种 Quadratic 训练场景的实现方案，从用户视角出发，尽量不修改 tneq-qc 库代码。

---

## 目录

1. [公共组件](#1-公共组件)
2. [案例 1：1024 量子比特 MPS + 标准高斯](#2-案例-1-1024-量子比特-mps--标准高斯)
3. [案例 2：1024 量子比特 MPS + 相关高斯](#3-案例-2-1024-量子比特-mps--相关高斯)
4. [案例 3：32 量子比特 Brickwall + 相关高斯](#4-案例-3-32-量子比特-brickwall--相关高斯)
5. [案例 4：自定义 MPS-of-Brickwall 结构](#5-案例-4-自定义-mps-of-brickwall-结构)
6. [必须修改的代码](#6-必须修改的代码)

---

## 1. 公共组件

### 1.1 后端与数值类型

所有案例使用 `complex64`（float32 实部 + float32 虚部），CPU 训练。

```python
from tneq_qc import BackendFactory, EngineCommon

backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='complex64')
engine  = EngineCommon(backend=backend, strategy_mode='full')
```

> **关于 complex32**：PyTorch 的 `torch.complex32`（chalf）使用 float16 实部 + float16 虚部，CPU 支持极差，
> 不建议使用。`complex64` 是标准的单精度复数类型，每个分量 32 位。

### 1.2 Quadratic 五段式结构

所有案例的模型结构均为 Quadratic 形式：

```
⟨circuit | tn† · Mx · tn | circuit⟩
```

即 5 段拼接：`cs + tn + mx + tn_h + cs_t`

其中 `tn` 可以是 MPS、Brickwall 或自定义结构。

- **案例 1, 2**：`tn` = MPS，可直接使用 `Quadratic` 类
- **案例 3, 4**：`tn` = Brickwall / 自定义结构，需手动拼接

### 1.3 自定义数据采样函数

现有 `make_data_fn` 从均匀分布 `U(-1, 1)` 采样。要使用高斯分布等自定义分布，
需自行编写 `data_fn`。模式如下：

```python
import numpy as np
from tneq_qc import DataGenerator

data_gen = DataGenerator(backend, mx_K=PHYS_DIM)

def make_custom_data_fn(data_gen, qctn, sample_fn, batch_size, num_qubits, K):
    """
    创建自定义数据采样的 data_fn。

    Args:
        sample_fn: callable(batch_size, num_qubits) -> np.ndarray [B, D]
                   返回 float32 数组，每行是一个样本。
    """
    # 自动检测 mx 核心名
    names_map = getattr(qctn, 'core_names', {})
    mx_core_names = [
        names_map[sym] for sym in qctn.cores
        if names_map.get(sym, '').startswith('mx.')
    ]

    def data_fn(step):
        x = sample_fn(batch_size, num_qubits)
        Mx_list, _ = data_gen.generate(x, K=K, ret_type='TNTensor')
        for i, name in enumerate(mx_core_names):
            qctn[name] = Mx_list[i]

    return data_fn
```

### 1.4 标准高斯采样函数

```python
def sample_gaussian(batch_size, num_qubits):
    """标准高斯分布 N(0, I)"""
    return np.random.randn(batch_size, num_qubits).astype(np.float32)
```

### 1.5 相关高斯采样函数

```python
import torch

def make_correlated_gaussian_sampler(ndim):
    """
    带近邻相关的高斯分布。
    协方差矩阵：对角线为 1，次对角线为 0.2。
    """
    cov_matrix = torch.eye(ndim, dtype=torch.float64)
    indices = torch.arange(ndim - 1)
    cov_matrix[indices + 1, indices] = 0.2
    cov_matrix[indices, indices + 1] = 0.2
    cov_matrix.diagonal().fill_(1.0)

    dist = torch.distributions.MultivariateNormal(
        loc=torch.zeros(ndim, dtype=torch.float64),
        covariance_matrix=cov_matrix,
    )

    def sample_fn(batch_size, num_qubits):
        # num_qubits 应等于 ndim
        samples = dist.sample((batch_size,))
        return samples.float().numpy()  # -> float32

    return sample_fn
```

---

## 2. 案例 1: 1024 量子比特 MPS + 标准高斯

### 2.1 方案

直接使用 `Quadratic` 类，nqubits=1024, bond_dim=2, phys_dim=2。

**可行性分析**：
- Benchmark 数据显示 200 量子比特 Quadratic（998 核心）初始化约 84 秒。1024 量子比特
  将产生约 5120 核心，预计初始化时间 **5-10 分钟**。
- 收缩时间随量子比特数线性增长，1024 比特单步预计 **1-3 秒**。
- opt_einsum 的 `get_symbol()` 支持 Unicode 扩展，5000+ 符号没有问题。

### 2.2 实现

```python
# examples/train_mps_1024_gaussian.py

from tneq_qc import (
    QCTN, BackendFactory, EngineCommon, Quadratic,
    DataGenerator, create_optimizer,
)

N_QUBITS   = 1024
BOND_DIM   = 2
PHYS_DIM   = 2
BATCH_SIZE = 128
N_STEPS    = 200
LR         = 0.01

backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='complex64')
engine  = EngineCommon(backend=backend, strategy_mode='full')
data_gen = DataGenerator(backend, mx_K=PHYS_DIM)

# 构建模型
model = Quadratic(nqubits=N_QUBITS, bond_dim=BOND_DIM, phys_dim=PHYS_DIM,
                  backend=backend).auto_init()
model._submodules['mps'].requires_grad_(True)
combined = model.build()

# 自定义高斯采样 data_fn
def data_fn(step):
    x = np.random.randn(BATCH_SIZE, N_QUBITS).astype(np.float32)
    Mx_list, _ = data_gen.generate(x, K=PHYS_DIM, ret_type='TNTensor')
    mx_names = model.mx_core_names
    for i, name in enumerate(mx_names):
        combined[name] = Mx_list[i]

# 训练
optimizer = create_optimizer("sgdg", combined.parameters(), backend=backend, lr=LR)
for step in range(1, N_STEPS + 1):
    data_fn(step)
    loss_val, grads = engine.contract_for_gradient(combined, target=1, loss='nll')
    optimizer.step(list(grads))
```

### 2.3 注意事项

1. **初始化时间长**：1024 量子比特的图解析和邻接表构建需要数分钟，这是一次性开销。
2. **bond_dim 限制**：已知 Quadratic 在 bond_dim > 2 时存在 einsum 维度匹配 bug。
   当前案例使用 bond_dim=2，不受影响。
3. **内存**：bond_dim=2 时，1024 量子比特 MPS 参数量约 16,000，内存开销 < 1 MB。

---

## 3. 案例 2: 1024 量子比特 MPS + 相关高斯

### 3.1 方案

与案例 1 完全相同的模型结构，仅替换采样函数为相关高斯分布。

### 3.2 实现

```python
# examples/train_mps_1024_correlated.py

# ... 同案例 1 的模型构建 ...

# 创建相关高斯采样器
sample_fn = make_correlated_gaussian_sampler(N_QUBITS)

def data_fn(step):
    x = sample_fn(BATCH_SIZE, N_QUBITS)
    Mx_list, _ = data_gen.generate(x, K=PHYS_DIM, ret_type='TNTensor')
    mx_names = model.mx_core_names
    for i, name in enumerate(mx_names):
        combined[name] = Mx_list[i]

# 训练循环同案例 1
```

### 3.3 注意事项

1. **协方差矩阵构造**：1024×1024 的协方差矩阵需要用 `float64` 以确保正定性。
   `MultivariateNormal` 的 Cholesky 分解在 float32 下可能失败。
2. **采样开销**：1024 维多元高斯的 Cholesky 分解在首次采样时会计算一次，之后
   被缓存。首次采样约需 1-2 秒。
3. **数据范围**：相关高斯的样本范围理论上是 `(-∞, +∞)`，但 Hermite 基函数的
   有效支撑区间约为 `(-4, 4)`。极端值会导致特征映射值很小，但不影响数值稳定性
   （TNTensor 的 scale 机制会处理）。

---

## 4. 案例 3: 32 量子比特 Brickwall + 相关高斯

### 4.1 方案

**不能使用 `Quadratic` 类**，因为它硬编码了 MPS 作为可训练模块。需要手动构建
5 段结构：`cs + brickwall + mx + brickwall_h + cs_t`。

使用 `QCTNHelper.brickwall()` 生成 brickwall 图，然后手动拼接。

**Brickwall 结构说明**：

```
QCTNHelper.brickwall(nqubits=4, n_layers=4, phys_dim=2)
```

生成如下图：

```
-2-a-2-----b-----2-
-2-a-2-c-2-b-2-d-2-
-2-e-2-c-2-f-2-d-2-
-2-e-2-----f-----2-
```

- `n_layers` 控制列宽（`m = n_layers // 2` 个核心每对相邻行）
- 核心数 = `n_layers * (nqubits // 2)`
- 所有维度均为 `phys_dim`

### 4.2 实现

```python
# examples/train_brickwall_32.py

from tneq_qc import (
    QCTN, BackendFactory, EngineCommon, QCTNHelper,
    DataGenerator, create_optimizer,
)
from tneq_qc.modules.small import CircuitState, MeasureMatrix

N_QUBITS   = 32
N_LAYERS   = 4       # brickwall 层数
PHYS_DIM   = 2
BATCH_SIZE = 128
N_STEPS    = 500
LR         = 0.01

backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='complex64')
engine  = EngineCommon(backend=backend, strategy_mode='full')
data_gen = DataGenerator(backend, mx_K=PHYS_DIM)

# --- 构建各段 ---

# 1) CircuitState (ket)
circuit = CircuitState(N_QUBITS, PHYS_DIM, backend).auto_init()

# 2) Brickwall (可训练)
bw_graph = QCTNHelper.brickwall(N_QUBITS, n_layers=N_LAYERS, phys_dim=PHYS_DIM)
brickwall = QCTN(bw_graph, backend=backend).auto_init()
brickwall.requires_grad_(True)

# 3) MeasureMatrix (数据注入)
mx = MeasureMatrix(N_QUBITS, PHYS_DIM, backend).auto_init()

# 4) Hermitian conjugate of brickwall
bw_hermit = brickwall.hermit()

# 5) CircuitState bra
circuit_bra = circuit.bra()

# --- 拼接 ---
combined = QCTN.concat([
    ('cs',   circuit),
    ('bw',   brickwall),
    ('mx',   mx),
    ('bw_h', bw_hermit),
    ('cs_t', circuit_bra),
])

print(f"Combined: {combined.ncores} cores, {len(combined.parameters())} trainable")

# --- 数据函数（相关高斯）---
sample_fn = make_correlated_gaussian_sampler(N_QUBITS)

names_map = combined.core_names
mx_core_names = [
    names_map[sym] for sym in combined.cores
    if names_map.get(sym, '').startswith('mx.')
]

def data_fn(step):
    x = sample_fn(BATCH_SIZE, N_QUBITS)
    Mx_list, _ = data_gen.generate(x, K=PHYS_DIM, ret_type='TNTensor')
    for i, name in enumerate(mx_core_names):
        combined[name] = Mx_list[i]

# --- 训练 ---
optimizer = create_optimizer("sgdg", combined.parameters(), backend=backend, lr=LR)
for step in range(1, N_STEPS + 1):
    data_fn(step)
    loss_val, grads = engine.contract_for_gradient(combined, target=1, loss='nll')
    optimizer.step(list(grads))
```

### 4.3 关键点

1. **手动拼接替代 `Quadratic` 类**：用 `QCTN.concat` 将 5 段按顺序拼接。
   核心名自动带前缀（`cs.A`, `bw.a`, `mx.A`, `bw_h.a`, `cs_t.A`）。
2. **`brickwall.hermit()`**：对任意 QCTN 均可调用，返回共轭转置版本，
   梯度通过 PyTorch autograd 自动反向传播。
3. **`circuit.bra()`**：返回 bra 向量（输入侧关闭的电路态），用于关闭右边界。
4. **mx 核心检测**：通过检查 `core_names` 中以 `'mx.'` 开头的名称来定位
   数据注入核心。

### 4.4 核心数估算

32 量子比特 brickwall (n_layers=4)：
- brickwall 核心数 = `4 * (32 // 2)` = 64
- circuit state 核心 = 32 × 2（ket + bra）= 64
- measure matrix 核心 = 32
- hermit brickwall 核心 = 64
- **总计 = 224 核心**

对比 MPS Quadratic (32 量子比特)：
- MPS 核心 = 32 × 2（正向 + hermit）= 64
- circuit + mx = 96
- **总计 = 158 核心**

Brickwall 的核心数更多（更多参数），但每个核心只连接 2 行（形状小），
收缩效率依然很高。

---

## 5. 案例 4: 自定义 MPS-of-Brickwall 结构

### 5.1 结构定义

外层是 MPS 链式结构，但每个 MPS "块" 内部是一个 2 层 brickwall。
相邻块之间保持 3 个量子比特的重叠。

**示例（7 量子比特，2 个块）**：

```
Block 1 (ABCD)          Block 2 (EFGH)
qubits 0-4               qubits 2-6
                          ←overlap 3→

-2-A-------------2-
-2-A-2-B---------2-
-2-C-2-B-2-E-----2-      ← 重叠区域开始
-2-C-2-D-2-E-2-F-2-
-2-----D-2-G-2-F-2-      ← 重叠区域结束
-2---------G-2-H-2-
-2-------------H-2-
```

**示例（11 量子比特，2 个块，每块 7 量子比特）**：

```
-2-A-----2-------2-
-2-A-2-D-2-------2-
-2-B-2-D-2-------2-
-2-B-2-E-2-------2-
-2-C-2-E-2-G-----2-      ← 重叠区域开始
-2-C-2-F-2-G-2-J-2-
-2-----F-2-H-2-J-2-
-2---------H-2-K-2-      ← 重叠区域结束
-2---------I-2-K-2-
-2---------I-2-L-2-
-2-----------2-L-2-
```

### 5.2 结构分析

每个 brickwall 块内部：
- **偶数层核心**：配对 (0,1), (2,3), (4,5), ...
- **奇数层核心**：配对 (1,2), (3,4), (5,6), ...
- 对于 `block_qubits` 个量子比特的块，核心数 = `block_qubits - 1`

块间关系：
- 相邻块重叠 `overlap = 3` 个量子比特
- 每增加一个块，增加 `block_qubits - overlap` 个新量子比特
- 总量子比特数 = `block_qubits + (n_blocks - 1) * (block_qubits - overlap)`

### 5.3 图生成器

需要编写一个图生成函数。核心逻辑：

```python
import opt_einsum

def generate_mps_brickwall_graph(total_qubits, block_qubits, overlap=3, phys_dim=2):
    """
    生成 MPS-of-Brickwall 图字符串。

    Args:
        total_qubits: 总量子比特数
        block_qubits: 每个块的量子比特数（必须 >= overlap + 2）
        overlap: 相邻块重叠的量子比特数 (默认 3)
        phys_dim: 物理维度 (默认 2)

    Returns:
        str: QCTN 图字符串
    """
    stride = block_qubits - overlap
    n_blocks = 1 + max(0, (total_qubits - block_qubits + stride - 1) // stride)

    # 实际总量子比特数（可能因整除而调整）
    actual_qubits = block_qubits + (n_blocks - 1) * stride

    dim = str(phys_dim)

    # 为每个块的每个核心分配全局唯一符号
    # 每个块有 block_qubits - 1 个核心
    cores_per_block = block_qubits - 1
    total_cores = n_blocks * cores_per_block
    symbols = [opt_einsum.get_symbol(i) for i in range(total_cores)]

    # 构建每行的核心列表
    # row_cores[q] = [(block_idx, symbol, partner_qubit), ...]
    # 其中 partner_qubit 是同一核心连接的另一个量子比特
    row_cores = [[] for _ in range(actual_qubits)]

    for b in range(n_blocks):
        start = b * stride
        base_sym = b * cores_per_block

        for pair_idx in range(cores_per_block):
            # 在块内，核心 pair_idx 连接量子比特 pair_idx 和 pair_idx+1
            q1 = start + pair_idx
            q2 = start + pair_idx + 1
            sym = symbols[base_sym + pair_idx]

            # 偶数层/奇数层决定核心在行内的位置（左侧或右侧）
            # 但对于 MPS-of-brickwall，核心的列位置由块和层共同决定
            row_cores[q1].append((b, sym))
            row_cores[q2].append((b, sym))

    # 生成图字符串
    lines = []
    for q in range(actual_qubits):
        parts = [f'-{dim}-']
        cores_on_row = row_cores[q]

        if not cores_on_row:
            # 没有核心的行：仅有边界维度
            parts.append(f'-{dim}-')
        else:
            # 按块索引排序，同一块内按符号排序
            cores_on_row.sort(key=lambda x: (x[0], x[1]))

            for i, (block_idx, sym) in enumerate(cores_on_row):
                parts.append(sym)
                if i < len(cores_on_row) - 1:
                    parts.append(f'-{dim}-')

            parts.append(f'-{dim}-')

        lines.append(''.join(parts))

    return '\n'.join(lines)
```

**但是**，上述简化版不处理核心的空位对齐（padding）。在 ASCII 图格式中，
不同行上的核心必须在列方向上对齐：同一核心在两行上的列位置必须相同，
中间用 `-` 填充。

### 5.4 精确图生成（列对齐版）

```python
def generate_mps_brickwall_graph(total_qubits, block_qubits, overlap=3, phys_dim=2):
    """
    生成 MPS-of-Brickwall 图字符串（精确列对齐）。

    每个块是一个 2 层 brickwall：
    - 偶数层核心连接 (start+0, start+1), (start+2, start+3), ...
    - 奇数层核心连接 (start+1, start+2), (start+3, start+4), ...

    这保证了 brickwall 的交错结构，而非简单的逐对连接。
    """
    import opt_einsum

    dim = str(phys_dim)
    stride = block_qubits - overlap
    n_blocks = 1 + max(0, (total_qubits - block_qubits + stride - 1) // stride)
    actual_qubits = block_qubits + (n_blocks - 1) * stride

    # 为每个块生成 brickwall 核心对
    # 每个块的核心按 brickwall 层次排列
    all_core_pairs = []  # list of (q1, q2, block_idx, layer_in_block)
    sym_idx = 0

    for b in range(n_blocks):
        start = b * stride
        bq = block_qubits

        # 偶数层：(start+0, start+1), (start+2, start+3), ...
        for p in range(bq // 2):
            q1 = start + 2 * p
            q2 = start + 2 * p + 1
            all_core_pairs.append((q1, q2, b, 0, sym_idx))
            sym_idx += 1

        # 奇数层：(start+1, start+2), (start+3, start+4), ...
        for p in range((bq - 1) // 2):
            q1 = start + 2 * p + 1
            q2 = start + 2 * p + 2
            all_core_pairs.append((q1, q2, b, 1, sym_idx))
            sym_idx += 1

    total_cores = sym_idx
    symbols = [opt_einsum.get_symbol(i) for i in range(total_cores)]

    # 为核心对分配列位置
    # 按 (block_idx, layer) 排序，同层按 q1 排序
    all_core_pairs.sort(key=lambda x: (x[2], x[3], x[0]))

    # 确定每个核心的列位置
    # 列位置 = 块偏移 + 层偏移
    # 块偏移 = block_idx * cols_per_block
    # 层偏移 = layer_in_block * 1

    # 计算每个块需要的列数
    cols_per_block = 2  # 偶数层 + 奇数层

    total_cols = n_blocks * cols_per_block

    # 构建网格：grid[qubit][col] = symbol or None
    grid = [[None] * total_cols for _ in range(actual_qubits)]

    for (q1, q2, b, layer, sidx) in all_core_pairs:
        col = b * cols_per_block + layer
        grid[q1][col] = symbols[sidx]
        grid[q2][col] = symbols[sidx]

    # 生成图字符串
    lines = []
    for q in range(actual_qubits):
        parts = [f'-{dim}-']
        for col in range(total_cols):
            if grid[q][col] is not None:
                parts.append(grid[q][col])
                parts.append(f'-{dim}-')
            else:
                # 空位：用 padding 填充（保持列对齐）
                parts.append(f'---{dim}---')
        lines.append(''.join(parts))

    return '\n'.join(lines)
```

> **注意**：上述代码是概念性的。实际的图生成需要仔细处理 ASCII 列对齐，
> 确保同一核心在相邻两行的相同列位置出现。推荐的做法是参考
> `QCTNHelper.brickwall()` 的实现（基于二维字符数组 `line_list`），
> 为每个核心计算精确的字符偏移量。

### 5.5 实现方案

```python
# examples/train_mps_brickwall.py

from tneq_qc import QCTN, BackendFactory, EngineCommon, DataGenerator, SGDG
from tneq_qc.modules.small import CircuitState, MeasureMatrix

TOTAL_QUBITS = 32     # 总量子比特数
BLOCK_QUBITS = 7      # 每个 brickwall 块的量子比特数
OVERLAP      = 3      # 相邻块重叠
PHYS_DIM     = 2
BATCH_SIZE   = 128
N_STEPS      = 500
LR           = 0.01

backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='complex64')
engine  = EngineCommon(backend=backend, strategy_mode='full')
data_gen = DataGenerator(backend, mx_K=PHYS_DIM)

# 1) 生成自定义 TN 图
tn_graph = generate_mps_brickwall_graph(TOTAL_QUBITS, BLOCK_QUBITS, OVERLAP, PHYS_DIM)
custom_tn = QCTN(tn_graph, backend=backend).auto_init()
custom_tn.requires_grad_(True)

# 2) 辅助结构
circuit    = CircuitState(TOTAL_QUBITS, PHYS_DIM, backend).auto_init()
mx         = MeasureMatrix(TOTAL_QUBITS, PHYS_DIM, backend).auto_init()
tn_hermit  = custom_tn.hermit()
circuit_bra = circuit.bra()

# 3) 拼接
combined = QCTN.concat([
    ('cs',   circuit),
    ('tn',   custom_tn),
    ('mx',   mx),
    ('tn_h', tn_hermit),
    ('cs_t', circuit_bra),
])

# 4) 数据函数（标准高斯）
names_map = combined.core_names
mx_core_names = [
    names_map[sym] for sym in combined.cores
    if names_map.get(sym, '').startswith('mx.')
]

def data_fn(step):
    x = np.random.randn(BATCH_SIZE, TOTAL_QUBITS).astype(np.float32)
    Mx_list, _ = data_gen.generate(x, K=PHYS_DIM, ret_type='TNTensor')
    for i, name in enumerate(mx_core_names):
        combined[name] = Mx_list[i]

# 5) 训练
optimizer = create_optimizer("sgdg", combined.parameters(), backend=backend, lr=LR)
for step in range(1, N_STEPS + 1):
    data_fn(step)
    loss_val, grads = engine.contract_for_gradient(combined, target=1, loss='nll')
    optimizer.step(list(grads))
```

### 5.6 参数估算

以 `TOTAL_QUBITS=32, BLOCK_QUBITS=7, OVERLAP=3` 为例：
- `stride = 7 - 3 = 4`
- `n_blocks = 1 + ceil((32 - 7) / 4) = 1 + 7 = 8` 块
- 实际量子比特 = `7 + 7 * 4 = 35`（可能比 32 略多）
- 每块核心数 = `3 + 2 = 5`（偶数层 3 对 + 奇数层 2 对，对于 7 量子比特块）

  等等，重新计算：
  - 7 量子比特的 brickwall：
    - 偶数层对数 = `7 // 2 = 3` → 核心 (0,1), (2,3), (4,5)
    - 奇数层对数 = `(7-1) // 2 = 3` → 核心 (1,2), (3,4), (5,6)
    - 每块 6 个核心
- 总 TN 核心 = `8 * 6 = 48`
- circuit + mx + bra = `35 * 2 + 35 = 105`
- hermit 核心 = 48
- **总计约 201 核心**

---

## 6. 必须修改的代码

### 6.1 无需修改的部分

以下操作完全通过现有 API 实现，**无需修改**任何库代码：

| 功能 | 实现方式 |
|------|---------|
| 自定义数据分布 | 自写 `data_fn` 闭包，替代 `make_data_fn` |
| Brickwall Quadratic | 用 `QCTN.concat` 手动拼接 5 段 |
| 自定义 TN 结构 | 直接传 ASCII 图字符串给 `QCTN()` 构造函数 |
| hermit / bra | `QCTN.hermit()` 和 `QCTN.bra()` 对任意 QCTN 通用 |
| 梯度训练 | `requires_grad_()` 和 `parameters()` 对任意 QCTN 通用 |

### 6.2 可能需要修改/新增的代码

#### 6.2.1 MPS-of-Brickwall 图生成器（新增工具函数）

**文件**：`tneq_qc/utils/graph_generators.py` 或用户脚本内

当前的 `QCTNHelper` 提供了 `mps()` 和 `brickwall()` 两种独立拓扑，
但没有组合结构的生成器。

**建议方案**：在用户脚本中实现图生成函数（不改库代码），或在
`QCTNHelper` 中添加一个新的静态方法：

```python
@staticmethod
def mps_brickwall(total_qubits, block_qubits, overlap=3,
                  phys_dim=2) -> str:
    """MPS-of-Brickwall: 外层 MPS 链，每个块内部为 brickwall。"""
    ...
```

**推荐**：先放在用户脚本中验证正确性，待稳定后再考虑合入库。

#### 6.2.2 `Quadratic` 类支持自定义 TN 模块（可选增强）

当前 `Quadratic.__init__` 硬编码创建 `MPS` 作为可训练模块。
如果需要频繁使用不同 TN 结构做 Quadratic 训练，可以增强 `Quadratic` 类：

```python
class Quadratic(QCTN):
    def __init__(self, nqubits, bond_dim=None, phys_dim=2,
                 backend=None, tn_module=None):
        ...
        if tn_module is not None:
            self.register_module("mps", tn_module)
        else:
            self.register_module("mps", MPS(nqubits, bond_dim, phys_dim, backend))
```

**当前不需要修改**：手动拼接 `QCTN.concat` 已能满足所有案例。

#### 6.2.3 `make_data_fn` 支持自定义采样（可选增强）

可以给 `make_data_fn` 添加 `sample_fn` 参数：

```python
def make_data_fn(data_generator, qctn, ..., sample_fn=None):
    def data_fn(step):
        if sample_fn is not None:
            x = sample_fn(batch_size, num_qubits)
        else:
            x = np.random.uniform(-1.0, 1.0, ...)
        ...
```

**当前不需要修改**：用户自己写 `data_fn` 闭包即可。

---

## 附录：四个案例的对照表

| | 案例 1 | 案例 2 | 案例 3 | 案例 4 |
|---|--------|--------|--------|--------|
| **量子比特** | 1024 | 1024 | 32 | 32（可调） |
| **TN 结构** | MPS | MPS | Brickwall | MPS-of-Brickwall |
| **数据分布** | 标准高斯 | 相关高斯 | 相关高斯 | 标准高斯 |
| **数值类型** | complex64 | complex64 | complex64 | complex64 |
| **设备** | CPU | CPU | CPU | CPU |
| **使用 `Quadratic` 类** | 是 | 是 | 否（手动拼接） | 否（手动拼接） |
| **需要库修改** | 否 | 否 | 否 | 否（图生成器放用户脚本） |
| **脚本名** | `train_mps_1024_gaussian.py` | `train_mps_1024_correlated.py` | `train_brickwall_32.py` | `train_mps_brickwall.py` |
