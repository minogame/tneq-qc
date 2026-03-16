# Phase 1 重构总结

## 目标回顾

Phase 1 的目标是**丰富 TNTensor**，使其成为框架内的通用张量类型，并让 backend 方法透明支持 TNTensor 输入输出。

---

## 变更文件

### 1. `tneq_qc/core/tn_tensor.py`（完全重写）

#### 新增属性

| 属性 | 类型 | 说明 |
|---|---|---|
| `device` | property | 委托给底层 tensor（PyTorch 返回 `tensor.device`，JAX 返回 `tensor.devices()`） |
| `is_ref` | `bool` | 是否是另一个 TNTensor 的引用（零拷贝视图） |
| `is_transposed` | `bool` | 是否是转置引用 |
| `source` | `Optional[TNTensor]` | 被引用的原始 TNTensor |

`is_ref` / `is_transposed` / `source` 为孪生网络中 right 侧共享 left 侧参数提供基础（Phase 2 QCTN 重构的依赖）。

#### 新增布局操作（均返回新 TNTensor，scale 不变）

| 方法 | 说明 |
|---|---|
| `reshape(shape)` | 返回 `is_ref=True` 的视图 |
| `transpose(*dims)` | 无参数反转所有轴；有参数按 dims 排列；返回 `is_ref=True, is_transposed=True` |
| `conj()` | 复数共轭；返回 `is_ref=True` |
| `conj_transpose(*dims)` | 共轭转置（`†`） |
| `clone()` | 独立深拷贝，不共享内存 |
| `to(device, dtype)` | 迁移设备/类型；PyTorch 调用 `.to()`，JAX 调用 `jax.device_put` |

#### 新增算术运算（scale 传播规则）

| 运算 | scale 传播规则 |
|---|---|
| `a @ b` (`__matmul__`) | `scale_a * scale_b` |
| `a * scalar` (`__mul__` / `__rmul__`) | `scale_a * scalar`（tensor 不变） |
| `a * b`（两个 TNTensor） | `scale_a * scale_b` |
| `a / scalar` (`__truediv__`) | `scale_a / scalar` |
| `a + b` (`__add__`) | 归一化到 `a.scale`，`b.tensor` 乘以 `b.scale / a.scale` |
| `-a` (`__neg__`) | scale 取负，tensor 不变 |
| `a.sum(dim)` | scale 不变 |
| `a.mean(dim)` | scale 不变 |
| `a.einsum(eq, *others)` | 所有输入 scale 的乘积 |

**核心不变量**：所有运算均满足 `result.tensor * result.scale == 数学真实值`。

#### 内部重构

- `_update_scale(new_scale)` 统一更新 `scale` 和 `log_scale`，消除重复代码
- `scale_to` / `scale_with` / `auto_scale` 改用 `_update_scale`

---

### 2. `tneq_qc/backends/backend_interface.py`（追加）

在 `ComputeBackend` 末尾新增三个非抽象方法（mixin），为所有后端提供 TNTensor 双入口：

| 方法 | 说明 |
|---|---|
| `tn_einsum(equation, *operands)` | 若任意 operand 是 TNTensor，自动 unwrap 计算，返回 TNTensor（scale = 所有输入 scale 之积）；否则行为与 `einsum` 相同 |
| `tn_reshape(tensor, shape)` | TNTensor 输入返回 TNTensor；raw tensor 输入走原 `reshape` |
| `tn_matmul(a, b)` | 透明处理 TNTensor 矩阵乘法 |

---

### 3. `tneq_qc/backends/backend_pytorch.py` 和 `backend_jax.py`（小改动）

`reshape` 和 `einsum` 两个方法增加 `unwrap_tensor` 调用，使其在传入 TNTensor 时**自动解包**，返回原始 tensor：

```python
# backend_pytorch.py / backend_jax.py
def reshape(self, tensor, shape):
    tensor = self.unwrap_tensor(tensor)   # 新增
    return tensor.reshape(shape)

def einsum(self, equation, *operands):
    raw_ops = [self.unwrap_tensor(op) for op in operands]   # 新增
    return self.torch.einsum(equation, *raw_ops)
```

---

### 4. `tests/test_tn_tensor.py`（新建）

41 个单元测试，覆盖全部新功能，分 5 个测试类：

| 测试类 | 测试数 | 覆盖内容 |
|---|---|---|
| `TestMetadata` | 7 | shape / ndim / dtype / device / scale / is_ref / repr |
| `TestScaleHelpers` | 5 | scale_to / scale_with / auto_scale 及边界错误 |
| `TestLayoutOps` | 8 | reshape / transpose / conj / conj_transpose / clone / to |
| `TestArithmetic` | 13 | 所有算术运算的 scale 传播正确性 |
| `TestBackendWrappers` | 6 | tn_einsum / tn_reshape / backend.einsum auto-unwrap |

**全部 41 个测试通过**（`pytest tests/test_tn_tensor.py`）。

---

## 运行测试

```bash
conda run -n py311 python -m pytest tests/test_tn_tensor.py -v
```

---

## 设计决策记录

1. **`__mul__` 中 scalar 进 scale**：`a * k` 返回 `TNTensor(a.tensor, scale=a.scale * k)` 而非 `TNTensor(a.tensor * k, scale=a.scale)`。好处是零内存操作，缺点是 scale 可能积累误差。在框架内 scale 本身就是浮点数，精度足够。

2. **`__add__` 归一化策略**：加法时将 `b` 归一化到 `a.scale`（`b.tensor * (b.scale / a.scale)`），结果 scale 取 `a.scale`。这保证了数值正确性，但对 scale 差异悬殊的情况可能引入浮点误差。后续若有需要可改为取两者的几何均值。

3. **`transpose` 惰性求值**：只设置 `is_transposed=True` 标记，底层 tensor 使用 `.T` 或 `.permute`（PyTorch 的 `.T` 本身是视图，不拷贝内存）。

4. **backend.einsum 不返回 TNTensor**：`backend.einsum` 只 unwrap 输入，返回原始 tensor。若需要 TNTensor 结果，使用 `backend.tn_einsum` 或 `TNTensor.einsum`。这保持了与现有代码（`EinsumStrategy`、`GreedyStrategy` 中大量直接使用 `backend.einsum`）的兼容性。

---

## 向后兼容性

- `TNTensor.__init__` 签名：新增的 `is_ref`、`is_transposed`、`source` 均为关键字参数，默认值保持旧行为，**完全向后兼容**。
- 现有代码中 `TNTensor(tensor, scale)` 的调用无需修改。
- `backend.einsum` / `backend.reshape` 对原始 tensor 输入的行为不变。
