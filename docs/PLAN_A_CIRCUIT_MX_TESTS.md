# Plan A — Circuit / Mx 数值测试总结

## 目标

在**零新生产代码**的前提下，通过端到端数值验证，测试 QCTN 的 circuit states 和
measurement matrix（Mx）结构在完整计算管道中的正确性。

使用的现有 API（Phase 1+2 已完成）：
- `QCTN.from_graph()` / `QCTN.cores_weights`
- `QCTN.get_einsum_info()` / `QCTN.build_core_list()`
- `EinsumStrategy.get_compute_function()` → `compute_fn(cores_dict, states, mx)`

---

## 发现的 Bug：`build_core_list` 右侧 states 顺序错误

### 问题描述

`get_einsum_info` 生成的 einsum 方程中，右侧 circuit states 的 symbols 是
input_symbols_stack 字符串**反转**后映射得到的：

```python
# get_einsum_info 内部
circuit_states_symbols = ','.join(input_symbols_stack)   # "a,b"
output_states_symbols = ''
for char in circuit_states_symbols[::-1]:               # 遍历 "b,a"
    output_states_symbols += char if char==',' else new_symbol_mapping[char]
# output_states_symbols = "f,e"   ← qubit 1 先，qubit 0 后
```

方程 `a,b,abcd,efcd,f,e->` 中：
- position 4 ("f") = qubit 1 的右侧 state
- position 5 ("e") = qubit 0 的右侧 state

但 `build_core_list` 对右侧 states 使用**相同顺序** `[phi0, phi1]`：

```python
# 修复前（错误）
tensors.extend(circuit_states)          # [phi0, phi1] → position 4=phi0, 5=phi1

# 修复后（正确）
tensors.extend(reversed(circuit_states)) # [phi1, phi0] → position 4=phi1, 5=phi0
```

### 影响

| 场景 | 修复前 | 修复后 |
|---|---|---|
| `phi0 == phi1`（训练数据相同） | 结果正确 | 结果正确 |
| `phi0 ⊥ phi1`（正交 states）| 结果 ≈ 0（错误） | 结果 = 1.0（正确）|
| 任意不同 states | 计算 `⟨phi1⊗phi0\|A†A\|phi0⊗phi1⟩`（错）| 计算 `⟨phi0⊗phi1\|A†A\|phi0⊗phi1⟩`（正）|

原有代码中，生产使用（`EngineSiamese`、`train.py`）的 circuit states 通常对所有
qubit 使用相同的向量，因此该 bug 从未在实际训练中触发，但会在 qubit 独立输入时
产生错误结果。

### 修复位置

**[tneq_qc/core/qctn.py](../tneq_qc/core/qctn.py)** — `build_core_list` 末尾：

```python
# 修复后（一行改动）
tensors.extend(reversed(circuit_states))
```

---

## 新建测试文件

**[tests/test_circuit_mx.py](../tests/test_circuit_mx.py)** — 23 个测试，全部通过。

### 数学性质验证

| 标记 | 性质 | 测试方法 |
|---|---|---|
| F1 | Identity core + 任意 states: result = ‖φ₀‖²·‖φ₁‖² | 精确数值对比 |
| F2 | Orthogonal core + 正规化 states: result = 1.0 | 精确数值对比 |
| F3 | Orthogonal core + Mx=I: result[0] = 1.0 | 精确数值对比 |
| F4 | 单一 state 缩放 λ → result × λ² | 相对比例验证 |
| F5 | 所有 states 缩放 λ → result × λ^(2n) | 相对比例验证 |
| F6 | Zero state → result = 0 | 绝对值比较 |
| F7 | 所有 Mx 缩放 λ → result × λ²（双线性）| 相对比例验证 |
| F8 | Zero Mx → result = 0 | 绝对值比较 |
| F9 | Mx shape (K,d,d) → result shape (K,) | shape 断言 |
| F10 | 单 Mx slot 线性：result(Mx_a+Mx_b, fixed) = result(Mx_a) + result(Mx_b) | 加法验证 |
| F11 | 无 states 一致性：result_no_Mx == result_I_Mx[0] | 数值等价验证 |

### 测试类结构

```
TestCircuitStates (8 tests)
  ├── test_zero_state_gives_zero               [F6]
  ├── test_identity_core_normalized_state_no_mx [F1]
  ├── test_identity_core_norm_product           [F1]
  ├── test_ortho_core_normalized_state_no_mx    [F2]
  ├── test_ortho_core_arbitrary_normalized_state[F2]
  ├── test_state_scaling_one_vector_quadratic   [F4]
  ├── test_all_states_scaling                   [F5]
  └── test_result_nonneg_real_inputs            [sign]

TestMeasurementMatrix (7 tests)
  ├── test_identity_mx_unit_result              [F3]
  ├── test_zero_mx_gives_zero                   [F8]
  ├── test_mx_bilinear_scaling                  [F7]
  ├── test_mx_batch_shape                       [F9]
  ├── test_mx_single_slot_linearity             [F10]
  ├── test_identity_mx_matches_no_mx_...        [F11 variant]
  └── test_multibatch_mx_each_element_linear    [F7+F9 combined]

TestNoStatesMode (5 tests)
  ├── test_no_states_no_mx_positive             [sign]
  ├── test_no_states_no_mx_identity_core        [F1 no-states]
  ├── test_no_states_identity_mx_matches_no_mx  [F11]
  ├── test_no_states_zero_mx                    [F8]
  └── test_no_states_mx_bilinear_scaling        [F7]

TestTwoCoreQCTN (3 tests)
  ├── test_normalized_states_no_mx              [F2, 4-qubit]
  ├── test_identity_mx_batch_shape              [F9, 4-qubit]
  └── test_zero_state_no_output                 [F6, 4-qubit]
```

---

## Mx 双线性与单线性的说明

2-qubit QCTN 中，Mx 以**每 qubit 一个**的方式参与收缩，因此计算结果对 Mx 是
**双线性**的（而非线性）：

```
result(λ·Mx₀, λ·Mx₁) = λ² · result(Mx₀, Mx₁)   ← 对称双线性，λ² 缩放
result(Mx₀_a + Mx₀_b, Mx₁_fixed) = result(Mx₀_a, Mx₁_fixed) + result(Mx₀_b, Mx₁_fixed)  ← 固定一侧时线性
```

对于 n-qubit QCTN（每 qubit 一个 Mx slot），缩放全部 Mx 时：
```
result(λ·Mx₀, ..., λ·Mx_{n-1}) = λⁿ · result(Mx₀, ..., Mx_{n-1})
```

---

## 运行方法

```bash
# 仅新测试
conda run -n py311 python -m pytest tests/test_circuit_mx.py -v

# 完整套件（Phase 1+2+Plan A）
conda run -n py311 python -m pytest tests/test_tn_tensor.py tests/test_qctn_basic.py tests/test_greedy_strategy.py tests/test_circuit_mx.py -v
```

---

## 已通过测试统计

| 测试文件 | 数量 |
|---|---|
| `test_tn_tensor.py` | 41 |
| `test_qctn_basic.py` | 35 |
| `test_greedy_strategy.py` | 9 |
| `test_circuit_mx.py` | 23 |
| **合计** | **108** |

---

## 遗留工作

本次（Plan A）仅做集成测试，circuit states 和 Mx 仍以 raw tensor list 形式传入。
下一步可考虑：

- **方案 B（Phase 2 收尾）**：新建 `CircuitState` / `MeasurementMatrix` 轻量容器类，
  使 shapes() / as_list() 接口与 QCTN build_core_list 对接，并补充对应单元测试。
- **方案 C（Phase 3 前置）**：QCTN 增加 `from_cores()` 非 TNGraph 初始化路径，
  CircuitState/MeasurementMatrix 作为 QCTN 子类（需要 Phase 3 Trainer 重构支撑）。
