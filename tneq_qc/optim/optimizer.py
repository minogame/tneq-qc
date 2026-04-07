"""Legacy trainer-style optimizer wrapper.

Deprecated compatibility layer around the new optimizer registry.
"""

from __future__ import annotations

import random
import warnings
from typing import Optional

from .registry import create_optimizer


class Optimizer:
    """Legacy trainer wrapper that internally uses a modern optimizer."""

    def __init__(
        self,
        method='adam',
        learning_rate=0.01,
        max_iter=1000,
        tol=1e-6,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        engine=None,
        lr_schedule: Optional[list] = None,
        momentum=0.0,
        stiefel=True,
    ):
        self.method = method
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iter = 0
        self.momentum = momentum
        self.stiefel = stiefel
        self.lr_schedule = lr_schedule
        self.engine = engine
        self.opt_state = {}
        self._optimizer = None

        warnings.warn(
            "`tneq_qc.optim.optimizer.Optimizer` is legacy. "
            "Prefer `create_optimizer(...)` or built-in optimizer classes directly.",
            DeprecationWarning,
            stacklevel=2,
        )

    def _optimizer_kwargs(self):
        kwargs = {"lr": self.learning_rate}
        method = str(self.method).lower()
        if method == "adam":
            kwargs.update({
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
            })
        elif method == "momentum":
            kwargs.update({"momentum": self.momentum})
        elif method == "rmsprop":
            kwargs.update({"epsilon": self.epsilon})
        elif method == "sgdg":
            kwargs.update({
                "momentum": self.momentum,
                "stiefel": self.stiefel,
                "epsilon": self.epsilon,
            })
        return kwargs

    def _ensure_optimizer(self, params_list):
        if self._optimizer is None:
            backend = self.engine.backend if self.engine is not None else None
            self._optimizer = create_optimizer(
                self.method,
                params_list,
                backend=backend,
                **self._optimizer_kwargs(),
            )
        return self._optimizer

    def _sync_legacy_state(self):
        if self._optimizer is None:
            return
        self.opt_state = self._optimizer.state
        self.learning_rate = self._optimizer.lr

    def _apply_lr_schedule(self):
        if self.lr_schedule is None:
            return

        for step, lr in reversed(self.lr_schedule):
            if self.iter >= step:
                self.learning_rate = lr
                if self._optimizer is not None:
                    self._optimizer.lr = lr
                return

    def optimize(self, qctn, data_list, **kwargs):
        loss_value = 0

        while self.iter < self.max_iter:
            data_index = self.iter % len(data_list)
            loss, grads = self.engine.contract_for_gradient(
                qctn, **data_list[data_index], **kwargs
            )
            params_list = qctn.parameters()

            loss_value = float(loss) if hasattr(loss, 'item') else loss
            self._apply_lr_schedule()

            summary_writer = getattr(self, "summary_writer", None)
            if summary_writer is not None:
                try:
                    summary_writer.add_scalar("train/loss", loss_value, self.iter)
                except Exception:
                    pass

            if self.tol and loss_value < self.tol:
                print(f"Convergence achieved at iteration {self.iter} with loss {loss_value}.")
                break

            self.step(params_list, list(grads))

            eval_every = getattr(self, "eval_every", 0)
            eval_fn = getattr(self, "eval_fn", None)
            if eval_every and eval_fn is not None and ((self.iter + 1) % eval_every == 0):
                try:
                    metrics = eval_fn(self.iter + 1, qctn)
                except Exception as e:
                    print(f"[Optimizer] Eval function raised an exception at iter {self.iter + 1}: {e}")
                    metrics = None

                if metrics and summary_writer is not None:
                    for name, value in metrics.items():
                        try:
                            scalar = float(value)
                        except Exception:
                            continue
                        try:
                            summary_writer.add_scalar(f"eval/{name}", scalar, self.iter + 1)
                        except Exception:
                            pass

            save_every = getattr(self, "save_every", 0)
            checkpoint_fn = getattr(self, "checkpoint_fn", None)
            if save_every and checkpoint_fn is not None and ((self.iter + 1) % save_every == 0):
                try:
                    checkpoint_fn(self.iter + 1, qctn, loss_value)
                except Exception as e:
                    print(f"[Optimizer] Checkpoint function raised an exception at iter {self.iter + 1}: {e}")

            self.iter += 1
        else:
            print(f"Maximum iterations reached: {self.max_iter} with final loss {loss_value}.")

        return loss_value

    def optimize_debug(self, qctn, data_list, **kwargs):
        while self.iter < self.max_iter:
            data_index = self.iter % len(data_list)
            loss, grads = self.engine.contract_for_gradient(
                qctn, **data_list[data_index], **kwargs
            )

            loss_value = float(loss) if hasattr(loss, 'item') else loss
            self._apply_lr_schedule()
            if self.tol and loss_value < self.tol:
                print(f"Convergence achieved at iteration {self.iter} with loss {loss_value}.")
                break

            print(f"Iteration {self.iter}: loss = {loss_value}")
            self.step(qctn.parameters(), list(grads))
            self.iter += 1
        else:
            print(f"Maximum iterations reached: {self.max_iter} with final loss {loss_value}.")

    def optimize_with_target(self, qctn, target_qctn):
        while self.iter < self.max_iter:
            loss, grads = self.engine.contract_for_gradient(qctn, target=target_qctn, loss='mse')
            loss_value = float(loss) if hasattr(loss, 'item') else loss
            self._apply_lr_schedule()
            if loss_value < self.tol:
                print(f"Convergence achieved at iteration {self.iter} with loss {loss_value}.")
                break

            self.step(qctn.parameters(), list(grads))
            self.iter += 1
        else:
            print(f"Maximum iterations reached: {self.max_iter} with final loss {loss_value}.")

    def optimize_self_with_inputs(self, qctn, inputs_list):
        input_index_list = list(range(len(inputs_list)))
        train_index_list = random.sample(input_index_list, len(input_index_list))
        print(f"train_index_list : {train_index_list}")

        while self.iter < self.max_iter:
            inputs = inputs_list[train_index_list[self.iter % len(inputs_list)]]
            loss, grads = qctn.contract_with_self_for_gradient(inputs)
            loss_value = float(loss) if hasattr(loss, 'item') else loss
            self._apply_lr_schedule()
            if loss_value < self.tol:
                print(f"Convergence achieved at iteration {self.iter} with loss {loss_value}.")
                break

            self.step(qctn.parameters(), list(grads))
            self.iter += 1
        else:
            print(f"Maximum iterations reached: {self.max_iter} with final loss {loss_value}.")

    def step(self, params_list, grads):
        optimizer = self._ensure_optimizer(params_list)
        optimizer.lr = self.learning_rate
        optimizer.step(grads)
        self._sync_legacy_state()
