"""Parameter IO Mixin for QCTN.

Contains all methods related to core tensor initialization, loading,
saving, and injection from external sources.
"""
import warnings
import numpy as np
from pathlib import Path
from typing import Union, Optional, Mapping

from .tn_tensor import TNTensor


class QCTNIOMixin:
    """Mixin: core tensor init, set_cores, save/load, auto_init."""

    def _init_cores(self, distribution: str = "gaussian"):
        """
        Initialize the cores of the quantum circuit with random values.

        For each core, use the pre-computed values from adjacency_table:
        - input_shape: ranks from in_edge_list (already ordered by qubit_idx)
        - output_shape: ranks from out_edge_list (already ordered by qubit_idx)
        - input_dim: product of input_shape
        - output_dim: product of output_shape

        The core tensor is initialized with shape [input_dim, output_dim],
        then reshaped to input_shape + output_shape.

        Returns:
            None: The cores are stored in the `cores_weights` attribute.
        """
        for idx, core_info in enumerate(self.adjacency_table):
            core_name = core_info['core_name']
            input_shape = core_info['input_shape']
            output_shape = core_info['output_shape']
            input_dim = core_info['input_dim']
            output_dim = core_info['output_dim']

            full_shape = input_shape + output_shape

            if input_dim == output_dim:
                # Square case: use orthogonal (QR) initialization
                core = self.backend.init_random_core(
                    [input_dim, output_dim], distribution=distribution
                )
                core = self.backend.reshape(core, full_shape)
            else:
                # Non-square case: orthogonal init is not applicable;
                # sample a larger square matrix from the chosen distribution,
                # orthogonalize it, then slice to the target shape.
                max_dim = max(input_dim, output_dim)
                core = self.backend.init_random_core(
                    [max_dim, max_dim], distribution=distribution
                )
                # Slice to [input_dim, output_dim] then reshape
                raw = core.tensor if isinstance(core, TNTensor) else core
                raw_sliced = raw[:input_dim, :output_dim].contiguous()
                core = self.backend.reshape(
                    self.backend.wrap_tensor(raw_sliced) if isinstance(core, TNTensor) else raw_sliced,
                    full_shape,
                )

            self.cores_weights[core_name] = core

    def set_cores(self, cores, strict: bool = True):
        """
        Set core tensors from a list or dict.

        Each supplied tensor is validated to have the **same total number of
        elements** (numel) as the corresponding existing core weight.  If the
        shapes differ but the sizes match, the tensor is reshaped to the
        target core weight's shape.

        Args:
            cores (list | dict):
                * **list** – tensors are matched to ``self.cores`` by
                  positional order.

                  - *strict=True*: ``len(cores)`` must equal ``self.ncores``.
                  - *strict=False*: only the first ``min(len(cores), ncores)``
                    cores are set; a warning is emitted if the lengths differ.

                * **dict** – keys are core names (single-character symbols).

                  - *strict=True*: the key set must exactly equal
                    ``set(self.cores)`` (no missing, no extra keys).
                  - *strict=False*: only the intersection of keys is used;
                    warnings list any missing or extra keys.

            strict (bool): Whether to require an exact one-to-one match.
                Defaults to ``True``.

        Raises:
            TypeError: If *cores* is neither a list nor a dict.
            ValueError: If *strict=True* and the sizes / keys do not match,
                or if any tensor's total element count differs from its
                target core weight.
        """
        if isinstance(cores, list):
            self._set_cores_from_list(cores, strict)
        elif isinstance(cores, dict):
            self._set_cores_from_dict(cores, strict)
        else:
            raise TypeError(
                f"cores must be a list or dict, got {type(cores).__name__}"
            )

    def _set_single_core(self, core_name: str, tensor):
        """
        Validate *tensor* against the existing weight for *core_name*,
        reshape if necessary, and store it.

        Raises:
            ValueError: If the total number of elements does not match.
        """
        target = self.cores_weights[core_name]
        target_shape = tuple(target.shape)
        target_numel = int(np.prod(target_shape))

        src_shape = tuple(tensor.shape)
        src_numel = int(np.prod(src_shape))

        if src_numel != target_numel:
            raise ValueError(
                f"Core '{core_name}': size mismatch — input has "
                f"{src_numel} elements (shape {src_shape}) but target "
                f"has {target_numel} elements (shape {target_shape})."
            )

        if src_shape != target_shape:
            tensor = self.backend.reshape(tensor, list(target_shape))

        self.cores_weights[core_name] = tensor

    def _set_cores_from_list(self, cores: list, strict: bool):
        if strict:
            if len(cores) != self.ncores:
                raise ValueError(
                    f"strict=True: expected {self.ncores} core tensors, "
                    f"got {len(cores)}."
                )
            for idx, tensor in enumerate(cores):
                self._set_single_core(self.cores[idx], tensor)
        else:
            n = min(len(cores), self.ncores)
            if len(cores) != self.ncores:
                warnings.warn(
                    f"strict=False: input list has {len(cores)} tensors but "
                    f"QCTN has {self.ncores} cores. Only the first {n} will "
                    f"be set.",
                    stacklevel=3,
                )
            for idx in range(n):
                self._set_single_core(self.cores[idx], cores[idx])

    def _set_cores_from_dict(self, cores: dict, strict: bool):
        input_keys = set(cores.keys())
        self_keys = set(self.cores)

        if strict:
            if input_keys != self_keys:
                missing = self_keys - input_keys
                extra = input_keys - self_keys
                parts = []
                if missing:
                    parts.append(f"missing keys ({len(missing)}): {missing}")
                if extra:
                    parts.append(f"extra keys ({len(extra)}): {extra}")
                raise ValueError(
                    f"strict=True: key mismatch — {'; '.join(parts)}."
                )
            for core_name in self.cores:
                self._set_single_core(core_name, cores[core_name])
        else:
            common = input_keys & self_keys
            missing = self_keys - input_keys
            extra = input_keys - self_keys
            if missing:
                warnings.warn(
                    f"strict=False: {len(missing)} core(s) missing from "
                    f"input dict and will keep their current weights: "
                    f"{missing}",
                    stacklevel=3,
                )
            if extra:
                warnings.warn(
                    f"strict=False: {len(extra)} extra key(s) in input dict "
                    f"will be ignored: {extra}",
                    stacklevel=3,
                )
            for core_name in self.cores:
                if core_name in common:
                    self._set_single_core(core_name, cores[core_name])

    def auto_init(
        self,
        dtype=None,
        device=None,
        distribution: str = "gaussian",
    ) -> "QCTNIOMixin":
        """Initialize (or re-initialize) all core tensors with random orthogonal values.

        For graph-based modules, calls :meth:`_init_cores` to populate
        ``cores_weights``.  For composite modules (``graph=None``), recursively
        calls ``auto_init`` on every registered submodule.

        Args:
            dtype: Optional dtype hint forwarded to submodule ``auto_init``
                calls.  Not yet used by :meth:`_init_cores` directly; reserved
                for future backend-level dtype control.
            device: Optional device hint forwarded to submodule ``auto_init``
                calls.
            distribution: Random distribution used to generate the matrix
                before orthogonalization. Supported values currently include
                ``"gaussian"`` (default) and ``"uniform"``.

        Returns:
            self — supports chaining, e.g. ``MPS(3, 4).auto_init()``.
        """
        if self.graph is not None:
            self._init_cores(distribution=distribution)
        for sub in self._submodules.values():
            sub.auto_init(dtype=dtype, device=device, distribution=distribution)
        return self

    def save_cores(self, file_path: Union[str, Path], metadata: Optional[Mapping[str, str]] = None):
        """Save all core tensors into a safetensors file."""
        if self.backend is None:
            raise RuntimeError("Backend must be initialized before saving cores.")

        try:
            from safetensors.numpy import save_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is required to save cores; install it with `pip install safetensors`."
            ) from exc

        tensor_dict = {}
        for core_name, tensor in self.cores_weights.items():
            if isinstance(tensor, TNTensor):
                arr = self.backend.tensor_to_numpy(tensor.tensor * tensor.scale)
            else:
                arr = self.backend.tensor_to_numpy(tensor)
            if np.iscomplexobj(arr):
                tensor_dict[f"core_{core_name}_real"] = np.ascontiguousarray(arr.real)
                tensor_dict[f"core_{core_name}_imag"] = np.ascontiguousarray(arr.imag)
            else:
                tensor_dict[f"core_{core_name}"] = np.ascontiguousarray(arr)

        metadata_dict = {} if metadata is None else {str(k): str(v) for k, v in metadata.items()}

        # Persist core_names mapping so load_cores can restore it.
        core_names = getattr(self, 'core_names', {})
        if core_names:
            import json
            metadata_dict['_core_names'] = json.dumps(core_names)

        save_file(tensor_dict, str(file_path), metadata=metadata_dict)

    def load_cores(self, file_path: Union[str, Path], strict: bool = True) -> Mapping[str, str]:
        """Load saved core tensors from a safetensors file."""
        if self.backend is None:
            raise RuntimeError("Backend must be initialized before loading cores.")

        try:
            from safetensors.numpy import load_file
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError(
                "safetensors is required to load cores; install it with `pip install safetensors`."
            ) from exc

        tensor_dict = load_file(str(file_path))

        # Extract metadata via safe_open (load_file only returns tensors).
        metadata = {}
        try:
            with safe_open(str(file_path), framework="numpy") as f:
                meta = f.metadata()
                if meta:
                    metadata = dict(meta)
        except Exception:
            pass

        for core_name in self.cores:
            key = f"core_{core_name}"
            key_real, key_imag = f"core_{core_name}_real", f"core_{core_name}_imag"
            if key_real in tensor_dict:
                array = tensor_dict[key_real] + 1j * tensor_dict[key_imag]
            elif key in tensor_dict:
                array = tensor_dict[key]
            else:
                if strict:
                    raise KeyError(f"Missing tensor for core {core_name} in {file_path}")
                continue
            tensor = self.backend.convert_to_tensor(array)
            if isinstance(tensor, TNTensor):
                tn_tensor = tensor
            else:
                tn_tensor = TNTensor(tensor)
            tn_tensor.auto_scale()
            self.cores_weights[core_name] = tn_tensor

        metadata_dict = {str(k): str(v) for k, v in metadata.items()}
        self._loaded_metadata = metadata_dict

        # Restore core_names if saved.
        if '_core_names' in metadata_dict:
            import json
            saved_names = json.loads(metadata_dict['_core_names'])
            # Only restore names for cores that exist in this instance.
            for sym in self.cores:
                if sym in saved_names:
                    self.core_names[sym] = saved_names[sym]

        return metadata_dict

    @classmethod
    def from_pretrained(
        cls,
        graph: str,
        file_path: Union[str, Path],
        backend=None,
        strict: bool = True,
    ):
        """Create a QCTN instance loading core tensors from safetensors."""
        if backend is None:
            from ..backends.backend_factory import BackendFactory
            backend = BackendFactory.get_default_backend()

        instance = cls(graph, backend=backend)
        instance.load_cores(file_path, strict=strict)
        return instance
