import sys; sys.path.insert(0,"/home/wangyifeng/work/code/tneq-qc")
import numpy as np, jax
from tneq_qc import QCTN, EngineCommon, BackendFactory, create_optimizer
from tneq_qc.modules.small import MPS

def run(device):
    b = BackendFactory.create_backend("jax", device=device, dtype="float32")
    np.random.seed(0)
    teacher = MPS(nqubits=4, bond_dim=4, phys_dim=2, backend=b).auto_init(orthogonal=True)
    student = MPS(nqubits=4, bond_dim=4, phys_dim=2, backend=b).auto_init(orthogonal=True)
    # perturb student from teacher so there is clear gradient signal
    for n in student.cores:
        tw = np.asarray(b.tensor_to_numpy(teacher.cores_weights[n]))
        sw = tw + 0.3 * np.random.randn(*tw.shape).astype(np.float32)
        from tneq_qc.core.tn_tensor import TNTensor
        t = b.convert_to_tensor(sw); t = t if isinstance(t,TNTensor) else TNTensor(t)
        t.requires_grad_(True); student.cores_weights[n] = t
    combined = QCTN.concat([("u", student), ("t", teacher)]); combined.set_trace("all")
    eng = EngineCommon(backend=b, strategy="cotengra")
    opt = create_optimizer("adam", combined.parameters(), backend=b, lr=0.02)
    losses=[]
    for step in range(120):
        loss, grads = eng.contract_for_gradient(combined, target=1.0, loss="mse")
        opt.step(list(grads)); losses.append(float(np.asarray(b.tensor_to_numpy(loss))))
    pdev = list(combined.parameters()[0].tensor.devices())
    return losses, pdev, combined._cotengra_planner.nslices

lg, dev_g, ns = run("gpu")
print(f"GPU param device={dev_g} nslices={ns}")
print(f"GPU loss: {lg[0]:.6f} -> {lg[-1]:.6f}  (decreased: {lg[-1] < lg[0]*0.5})")
