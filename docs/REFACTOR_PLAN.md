# tneq-qc 重构设计文档

> 基于 `docs/重构.md` 中的重构方向，结合当前架构（详见 `docs/ARCHITECTURE.md`）制定的可操作重构计划。

---

## 一、重构目标概览

| 编号 | 方向 | 优先级 |
|---|---|---|
| R1 | 丰富 TNTensor，成为框架通用张量 | 高 |
| R2 | 重新设计 QCTN（仿 nn.Module） | 高 |
| R3 | 分离 Loss 模块 | 中 |
| R4 | 重构 Optimizer / Trainer | 中 |
| R5 | 分离 Contractor 的图解析与编译 | 中 |
| R6 | 完善测试体系 | 低（持续） |

---

## 二、R1：丰富 TNTensor

### 现状

`TNTensor` 当前只是一个轻量包装器，持有 `tensor`、`scale`、`log_scale`，仅有 `auto_scale`、`scale_to`、`scale_with` 等精度控制方法，不支持任何张量运算。

### 目标

使 `TNTensor` 成为框架内的**通用张量类型**，支持：

1. 常用张量操作（优先覆盖框架内实际使用的操作）
2. 双入口访问：`tensor.op(...)` 和 `backend.op(tensor, ...)`
3. 引用/转置/独立标记（为 R2 的 QCTN core 管理服务）
4. 设备（device）和 dtype 属性

### 新增属性与方法设计

```python
class TNTensor:
    # 新增属性
    device: str          # 'cpu', 'cuda:0', ...
    dtype: str           # 'float32', 'complex64', ...

    # 核心标记（R2 需要）
    is_ref: bool         # 是否是另一个 TNTensor 的引用
    is_transposed: bool  # 是否是转置引用
    source: Optional['TNTensor']  # 原始张量（当 is_ref=True 时）

    # 张量操作（与 backend 对应）
    def reshape(self, shape) -> 'TNTensor'
    def transpose(self, *dims) -> 'TNTensor'   # 返回转置引用（is_transposed=True）
    def conj(self) -> 'TNTensor'               # 共轭
    def clone(self) -> 'TNTensor'              # 独立副本
    def to(self, device=None, dtype=None) -> 'TNTensor'

    # 数学运算（保留 scale 语义）
    def __matmul__(self, other) -> 'TNTensor'
    def __mul__(self, scalar) -> 'TNTensor'
    def abs(self) -> 'TNTensor'
    def sum(self, dim=None) -> 'TNTensor'
    def mean(self, dim=None) -> 'TNTensor'

    # 与 backend 交互
    def einsum(self, eq: str, *others, backend) -> 'TNTensor'
```

### backend 双入口

`ComputeBackend` 的现有方法需更新，当参数为 `TNTensor` 时自动 unwrap/wrap：

```python
# 框架内使用示例
result = backend.einsum('ij,jk->ik', tn_a, tn_b)   # 返回 TNTensor
result = tn_a @ tn_b                                  # 等价
```

### 实现要点

- `transpose()` 不做实际内存拷贝，只设置 `is_transposed=True`，惰性求值
- `is_ref=True` 的 `TNTensor` 和 source 共享底层张量（for zero-copy）
- `scale` 在运算时需要正确传播（乘法：scales 相乘；einsum：scales 相乘）

### 工作量估计

- `tn_tensor.py`：~200 行新增
- `backend_interface.py`：各方法增加 TNTensor 判断，约 ~50 行
- `backend_pytorch.py` / `backend_jax.py`：同步更新

---

## 三、R2：重新设计 QCTN（仿 nn.Module）

### 现状问题

- `QCTN` 是一个大类，混合了结构定义、参数管理、einsum 生成等职责
- 不支持嵌套（QCTN 包含 QCTN）
- Circuit、Mx 测量矩阵与 QCTN 分开管理，不统一
- 邻接矩阵维护繁琐，拆分/合并不完善

### 新设计

#### 3.1 类继承结构

仿照 `torch.nn.Module`：

```python
class QCTN:
    """量子电路张量网络基类"""

    def __init__(self):
        self._cores: Dict[str, TNTensor]    # 本级的 core tensors（叶节点参数）
        self._submodules: Dict[str, 'QCTN'] # 子 QCTN（支持嵌套）

    def define(self):
        """用户重写：定义拓扑结构（图字符串或子模块关系）"""
        raise NotImplementedError

    def contract(self, *args, **kwargs):
        """调用 define() 构建完整张量列表，然后执行收缩"""
        ...

    def forward(self, *args, **kwargs):
        """用户重写：定义计算图（默认调用 contract）"""
        return self.contract(*args, **kwargs)

    # 参数管理
    def cores(self) -> Dict[str, TNTensor]   # 所有核心张量（含子模块）
    def named_cores(self)                    # 带名前缀的迭代

    # 初始化
    @classmethod
    def from_graph(cls, graph_str: str) -> 'QCTN':  # 从 ASCII 图创建
```

#### 3.2 具体实现说明

**基础 QCTN**（从 graph 字符串初始化，保持现有功能）：

```python
class BasicQCTN(QCTN):
    def __init__(self, graph_str: str, bond_dims=None):
        super().__init__()
        self._graph = TNGraph(graph_str)      # 构建邻接表（不再用邻接矩阵）
        self._init_cores()                    # 正交初始化 core tensors

    def define(self):
        return self._graph                    # 返回图结构
```

**嵌套 QCTN 示例**：

```python
class MyModel(QCTN):
    def __init__(self):
        super().__init__()
        self.left = BasicQCTN("-2-A-4-B-2-")
        self.right = BasicQCTN("-2-C-4-D-2-")

    def define(self):
        return concat(self.left, self.right)  # 拼接两个 QCTN

    def forward(self, x):
        return self.contract(x)
```

**Circuit 和 Mx 用 QCTN 表示**：

```python
class CircuitState(QCTN):
    """线路态，每个 qubit 一个向量"""
    ...

class MeasurementMatrix(QCTN):
    """测量矩阵 Mx"""
    ...
```

#### 3.3 core tensor 引用语义（依赖 R1）

```python
# 孪生网络中，right 侧是 left 侧的共轭转置引用
right_core = left_core.conj().transpose()   # is_ref=True, is_transposed=True
# 修改 left_core 的参数，right_core 自动感知（因为共享底层张量）
```

#### 3.4 split / merge 保留

- `split(qctn, at_bond)` → 返回两个 QCTN（保留现有逻辑）
- `merge(qctn_a, qctn_b)` → 合并为一个 QCTN

#### 3.5 Contractor 图解析移入 QCTN

按重构.md 第 6 条：**图解析**（遍历 graph，处理 core tensor 的复制/转置/边关系，生成 `core_tensor_list`）从 contractor 移到 QCTN 的方法中：

```python
class QCTN:
    def build_core_list(self, circuit_states, measure_matrices) -> List[TNTensor]:
        """根据当前结构，生成用于收缩的 tensor 列表（含 circuit_states 和 Mx）"""
        ...

    def get_einsum_info(self) -> Tuple[str, List]:
        """返回 einsum 方程和形状信息（供 Contractor 使用）"""
        ...
```

---

## 四、R3：Loss 模块

### 现状

损失函数散落在 `train.py` 和 `engine_siamese.py` 中，无统一接口。

### 新设计

新增 `tneq_qc/loss/` 模块：

```python
# tneq_qc/loss/base.py
class Loss(ABC):
    @abstractmethod
    def forward(self, prediction, target) -> TNTensor:
        ...

# tneq_qc/loss/losses.py
class NLLLoss(Loss): ...          # 负对数似然
class KLDivLoss(Loss): ...        # KL 散度
class WassersteinLoss(Loss): ...  # Wasserstein 距离（现已在 train.py 实现）
class ExpectationLoss(Loss): ...  # 期望值损失（量子测量）
```

- 仿照 PyTorch：`loss_fn = NLLLoss(); loss = loss_fn(pred, target)`
- Loss 支持配置（reduction='mean'|'sum'）

---

## 五、R4：重构 Optimizer / 新增 Trainer

### 现状问题

`Optimizer.optimize()` 承担了太多：数据集迭代、损失计算、参数更新都在其中，职责混乱。

### 新设计

#### 5.1 Optimizer — 只负责单步参数更新

```python
class Optimizer:
    """仅提供单步梯度更新，不控制训练循环"""

    def __init__(self, method='adam', lr=0.01, **kwargs):
        ...

    def step(self, qctn: QCTN, grads: Dict[str, TNTensor]):
        """执行一步参数更新"""
        ...

    def zero_grad(self):
        """清零梯度状态"""
        ...

    def set_lr(self, lr: float):
        """设置学习率（供 Trainer 的 LR schedule 调用）"""
        ...
```

#### 5.2 Trainer — 控制训练循环

```python
class Trainer:
    """训练循环管理器"""

    def __init__(self,
                 model: QCTN,
                 optimizer: Optimizer,
                 loss_fn: Loss,
                 engine: Engine,
                 lr_schedule=None,
                 callbacks=None):
        ...

    def fit(self, dataset, epochs: int, batch_size: int = None):
        """执行训练"""
        for epoch in range(epochs):
            for batch in dataset:
                loss = self._train_step(batch)
            self._epoch_end(epoch, loss)

    def _train_step(self, batch):
        pred = self.model.forward(batch)
        loss = self.loss_fn(pred, batch.target)
        grads = self.engine.compute_grads(loss, self.model)
        self.optimizer.step(self.model, grads)
        return loss

    def evaluate(self, dataset):
        """评估模型"""
        ...
```

---

## 六、R5：分离 Contractor 图解析与编译

### 现状

`EinsumStrategy.get_compute_function()` 中同时做了：
1. 图解析（遍历 QCTN 生成 einsum 字符串和 tensor 列表）
2. 编译（`opt_einsum.contract_expression`）
3. 计算（`compute_fn` 闭包）

### 新设计

按重构.md 描述，分三层：

```
Layer 1: 图解析（移入 QCTN）
  QCTN.build_core_list()     → core_tensor_list
  QCTN.get_einsum_info()     → (einsum_eq, shapes)

Layer 2: 编译（ContractionStrategy）
  strategy.get_compute_function(qctn, shapes_info, backend)
  → compute_fn
  当前：get_compute_function 返回 compute_fn，compute_fn 内部同时做编译+计算
  近期不变，未来：get_compute_function 获取最优计算顺序，compute_fn 按顺序计算

Layer 3: 计算（Engine）
  engine.execute(compute_fn, cores_dict, circuit_states, measure_matrices)
  → result
```

**当前阶段任务**（最小化改动）：

- 将图解析从 `EinsumStrategy` 和 `GreedyStrategy` 中提取，放入 `QCTN` 方法
- `get_compute_function` 改为调用 `qctn.build_core_list()` 和 `qctn.get_einsum_info()`
- `compile` 和 `compute_fn` 保持现有结构不变

---

## 七、R6：测试体系

### 测试规划

测试分三层逐步实现，优先级从高到低：

#### 单元测试（Unit Tests）

| 文件 | 内容 | 状态 |
|---|---|---|
| `test_tn_tensor.py` | TNTensor 所有方法（scale、引用语义、运算）| 待实现 |
| `test_tn_graph.py` | TNGraph 解析、to_string、节点操作 | 已有，需补充 |
| `test_backends.py` | 各 backend 方法一致性测试 | 已有部分 |
| `test_qctn_basic.py` | QCTN 初始化、from_graph、cores 访问 | 待实现 |
| `test_loss.py` | 各 Loss 的数值正确性 | 待实现 |
| `test_optimizer.py` | Optimizer 单步更新、lr_schedule | 待实现 |

#### 集成测试（Integration Tests）

| 文件 | 内容 | 状态 |
|---|---|---|
| `test_engine_contract.py` | Engine + Strategy + Backend 端到端收缩 | 已有部分 |
| `test_trainer.py` | Trainer 完整训练循环（toy dataset）| 待实现 |
| `test_qctn_nested.py` | 嵌套 QCTN 的 forward | 待实现 |
| `test_greedy_strategy.py` | GreedyStrategy 结果与 Einsum 一致性 | 已有 |

#### 系统测试（System Tests）

| 文件 | 内容 | 状态 |
|---|---|---|
| `test_distributed_trainer.py` | 分布式训练（多进程） | 已有 |
| `test_full_pipeline.py` | 完整训练 + 推理 + 采样 pipeline | 待实现 |

### 测试原则

- 所有 backend（JAX / PyTorch）用相同测试用例验证数值一致性
- 收缩策略（Einsum / Greedy）结果互相对比验证
- 使用 `pytest.fixture` 管理 backend 和 QCTN 实例的复用
- 数值测试设合理容差（`rtol=1e-4` for float32，`rtol=1e-6` for float64）

---

## 八、执行路线图

### Phase 1：基础设施（推荐首先执行）

1. **R1-基础**：为 `TNTensor` 添加 `device`、`dtype`、`is_ref`、`is_transposed` 属性，以及 `reshape`、`transpose`、`clone`、`to` 方法
2. **R1-运算**：实现 `__matmul__`、`__mul__`、`sum`、`mean`，确保 scale 正确传播
3. 更新 `backend_interface.py`：`einsum`、`reshape` 等方法支持 `TNTensor` 输入/输出

### Phase 2：QCTN 重构

4. **R2-基类**：实现 `QCTN` 基类（`__init__`、`cores()`、`named_cores()`、`from_graph()`）
5. **R2-图解析分离（R5）**：将 `build_core_list`、`get_einsum_info` 作为 QCTN 方法，从 contractor 中解耦
6. **R2-嵌套支持**：实现 `_submodules` 注册、`concat` 结构定义函数
7. **R2-引用语义**：利用 R1 的 TNTensor 引用实现孪生网络 right 侧共享参数

### Phase 3：训练流程重构

8. **R4-Optimizer**：精简 Optimizer 为单步更新 API
9. **R3-Loss**：抽取 Loss 模块，将 `train.py` 中的损失函数迁移
10. **R4-Trainer**：实现 `Trainer` 类，包含训练循环、lr_schedule、callbacks

### Phase 4：测试补全

11. 逐步补全 Phase 1~3 对应的单元测试和集成测试
12. 持续增加，不要求一次性全部完成

---

## 九、向后兼容策略

- `QCTN` 基类新增，原有 `QCTN`（`qctn.py`）改名为 `BasicQCTN` 或保留并作为 `QCTN` 子类
- `Optimizer.optimize()` 保留但标记为 deprecated，新逻辑通过 `Trainer` 暴露
- contractor 层的 `get_compute_function` 签名不变，内部改为调用 `qctn.build_core_list`
- 所有现有 test 保证在重构后继续通过

---

## 十、关键风险与注意事项

| 风险 | 应对 |
|---|---|
| TNTensor scale 在运算中传播错误 | 每个运算方法配备单元测试，对比 `tensor × scale` 数值 |
| 嵌套 QCTN 的 cores 命名冲突 | 使用前缀区分（`"left.A"` vs `"right.A"`） |
| 引用语义下 backward 梯度共享 | 确认 PyTorch autograd 对 view/transpose 的处理，必要时用 `.clone()` |
| Contractor 解耦后性能变化 | 在 Phase 2 完成后运行 benchmark，对比重构前后收缩时间 |
| 分布式模块依赖 QCTN 结构 | Phase 2 完成后再更新 `distributed/engine/distributed_engine.py` |
