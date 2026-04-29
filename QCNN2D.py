#!/usr/bin/env python3
"""Standalone QCNN module extracted from the notebook.

This module keeps the quantum/scipy convolution operators and the CFD
multi-backend solver so they can be reused outside the notebook.

It also provides two entry points:
- compare_scipy_and_qft_consistency(): validate SciPy vs quantum conv outputs
- run_flow_past_block_demo(): run a small flow-past-block CFD smoke test
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]
import torch.nn as nn  # type: ignore[import-not-found]
import torch.nn.functional as F  # type: ignore[import-not-found]
from scipy import signal  # type: ignore[import-not-found]

from AI4PDEs_bounds import (
	boundary_condition_2D_cw,
	boundary_condition_2D_p,
	boundary_condition_2D_u,
	boundary_condition_2D_v,
)
from AI4PDEs_utils import create_tensors_2D, get_weights_linear_2D

try:
	import pennylane as qml  # type: ignore[import-not-found]

	PENNYLANE_AVAILABLE = True
except ImportError:
	qml = None
	PENNYLANE_AVAILABLE = False

Quantum_device = "default.qubit"


def get_laplacian_kernel() -> np.ndarray:
	"""Return a simple 3x3 Laplacian kernel."""
	return np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float64)


def _to_numpy(tensor: Any) -> np.ndarray:
	if hasattr(tensor, "detach"):
		return tensor.detach().cpu().numpy()
	if hasattr(tensor, "cpu"):
		return tensor.cpu().numpy()
	return np.asarray(tensor)


def _ensure_torch_device(device_obj: Optional[torch.device] = None) -> torch.device:
	if device_obj is not None:
		return device_obj
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_kernels(dx: float = 1.0):
	"""Wrap AI4PDEs_utils.get_weights_linear_2D so the module can build its kernels."""
	return get_weights_linear_2D(dx)


class QuantumConvolutionStride2Engine:
	"""Stride-2 quantum convolution engine using amplitude embedding + BlockEncode."""

	def __init__(self, input_size: int = 4, kernel_3x3: Optional[np.ndarray] = None):
		if not PENNYLANE_AVAILABLE:
			raise ImportError("PennyLane is required for the quantum backend.")
		if input_size < 3:
			raise ValueError(f"input_size must be >= 3, got {input_size}")

		self.K = input_size
		self.output_size = self.K - 2
		self.input_dim = self.K * self.K
		self.output_dim = self.output_size * self.output_size
		self.n_encoding_qubits = int(np.ceil(np.log2(self.input_dim)))
		self.total_qubits = self.n_encoding_qubits + 1
		self.hilbert_dim = 2 ** self.n_encoding_qubits
		self.dev = qml.device(Quantum_device, wires=self.total_qubits)

		if kernel_3x3 is None:
			self.kernel = get_laplacian_kernel()
		else:
			self.kernel = np.asarray(kernel_3x3, dtype=np.float64)
			if self.kernel.shape != (3, 3):
				raise ValueError(f"kernel_3x3 must be 3x3, got {self.kernel.shape}")

		self.matrix_raw = self._build_matrix_from_kernel(self.kernel)
		self.alpha = np.linalg.norm(self.matrix_raw, ord=2)
		if self.alpha < 1e-9:
			self.alpha = 1.0
		self.matrix_norm = self.matrix_raw / self.alpha
		self.output_indices = self._compute_output_indices()
		self.cal_factor = self._calibrate()

	def _build_matrix_from_kernel(self, kernel: np.ndarray) -> np.ndarray:
		mat = np.zeros((self.hilbert_dim, self.hilbert_dim), dtype=np.float64)
		for out_r in range(self.output_size):
			for out_c in range(self.output_size):
				center_r = out_r + 1
				center_c = out_c + 1
				row_idx = center_r * self.K + center_c
				for ki in range(3):
					for kj in range(3):
						img_r = center_r + (ki - 1)
						img_c = center_c + (kj - 1)
						col_idx = img_r * self.K + img_c
						mat[row_idx, col_idx] = kernel[ki, kj]
		return mat

	def _compute_output_indices(self) -> List[int]:
		indices: List[int] = []
		for out_r in range(self.output_size):
			for out_c in range(self.output_size):
				center_r = out_r + 1
				center_c = out_c + 1
				indices.append(center_r * self.K + center_c)
		return indices

	def _calibrate(self) -> float:
		rng = np.random.default_rng(seed=42)
		cal_input = rng.random(self.hilbert_dim)
		cal_input = cal_input / np.linalg.norm(cal_input)

		expected = self.matrix_norm @ cal_input
		expected_norm = np.linalg.norm(expected)
		if expected_norm < 1e-9:
			return 1.0

		encoding_wires = list(range(1, self.total_qubits))
		all_wires = list(range(self.total_qubits))

		@qml.qnode(self.dev)
		def circuit():
			qml.AmplitudeEmbedding(features=cal_input, wires=encoding_wires, pad_with=0.0)
			qml.BlockEncode(self.matrix_norm, wires=all_wires)
			return qml.state()

		state = circuit()
		measured_norm = np.linalg.norm(state[: self.hilbert_dim])
		if measured_norm < 1e-9:
			return 1.0
		return expected_norm / measured_norm

	def run_block(self, block_KxK: np.ndarray) -> np.ndarray:
		if block_KxK.shape != (self.K, self.K):
			raise ValueError(f"Expected {self.K}x{self.K} block, got {block_KxK.shape}")

		x_flat = block_KxK.flatten()
		input_norm = np.linalg.norm(x_flat)
		if input_norm < 1e-6:
			return np.zeros((self.output_size, self.output_size))

		x_padded = np.zeros(self.hilbert_dim, dtype=np.float64)
		x_padded[: self.input_dim] = x_flat
		x_padded_norm = np.linalg.norm(x_padded)
		x_unit = x_padded / x_padded_norm

		encoding_wires = list(range(1, self.total_qubits))
		all_wires = list(range(self.total_qubits))

		@qml.qnode(self.dev)
		def execution_circuit():
			qml.AmplitudeEmbedding(features=x_unit, wires=encoding_wires, pad_with=0.0)
			qml.BlockEncode(self.matrix_norm, wires=all_wires)
			return qml.state()

		state_vec = execution_circuit()
		psi_signal = state_vec[: self.hilbert_dim]
		y_recovered = psi_signal.real * self.alpha * x_padded_norm * self.cal_factor
		output_flat = y_recovered[self.output_indices]
		return output_flat.reshape(self.output_size, self.output_size)


class QuantumConvolutionStride2Engine_LCU:
	"""Stride-2 quantum convolution engine using LCU + QFT."""

	def __init__(self, input_size: int = 4, kernel_3x3: Optional[np.ndarray] = None):
		if not PENNYLANE_AVAILABLE:
			raise ImportError("PennyLane is required for the quantum backend.")
		if input_size < 4:
			raise ValueError(f"Input size must be >= 4 for efficient LCU, got {input_size}")

		self.K = input_size
		self.output_size = self.K - 2
		self.n_wires_dim = int(np.ceil(np.log2(self.K)))
		self.N_pow2 = 2 ** self.n_wires_dim
		self.n_data_qubits = 2 * self.n_wires_dim
		self.n_ancilla_qubits = 4
		self.total_qubits = self.n_data_qubits + self.n_ancilla_qubits

		if kernel_3x3 is None:
			self.kernel = get_laplacian_kernel()
		else:
			self.kernel = np.asarray(kernel_3x3, dtype=np.float64)
			if self.kernel.shape != (3, 3):
				raise ValueError(f"kernel_3x3 must be 3x3, got {self.kernel.shape}")

		self.coeffs = self.kernel.flatten()
		self.abs_coeffs = np.abs(self.coeffs)
		self.signs = np.sign(self.coeffs)
		self.lcu_lambda = float(np.sum(self.abs_coeffs))
		if self.lcu_lambda < 1e-12:
			self.prep_state = np.zeros(16, dtype=np.float64)
			self.prep_state[0] = 1.0
		else:
			self.prep_state = np.sqrt(self.abs_coeffs / self.lcu_lambda)
			if len(self.prep_state) < 16:
				self.prep_state = np.pad(self.prep_state, (0, 16 - len(self.prep_state)))

		self.dev = qml.device(Quantum_device, wires=self.total_qubits)
		self.wires_ancilla = list(range(self.n_ancilla_qubits))
		self.wires_row = list(range(self.n_ancilla_qubits, self.n_ancilla_qubits + self.n_wires_dim))
		self.wires_col = list(range(self.n_ancilla_qubits + self.n_wires_dim, self.total_qubits))
		self.output_indices = self._compute_output_indices()

	def _compute_output_indices(self) -> List[int]:
		indices: List[int] = []
		for r in range(1, self.K - 1):
			for c in range(1, self.K - 1):
				indices.append(r * self.N_pow2 + c)
		return indices

	def _apply_shift_phase(self, shift_int, wires):
		for i, wire in enumerate(wires):
			theta = 2 * np.pi * shift_int / (2 ** (i + 1))
			qml.PhaseShift(theta, wires=wire)

	def run_block(self, block_KxK: np.ndarray) -> np.ndarray:
		if block_KxK.shape != (self.K, self.K):
			raise ValueError(f"Expected {self.K}x{self.K} block, got {block_KxK.shape}")

		pad_h = self.N_pow2 - self.K
		pad_w = self.N_pow2 - self.K
		block_padded = np.pad(block_KxK, ((0, pad_h), (0, pad_w)), mode="constant")
		x_flat = block_padded.flatten()
		norm_x = np.linalg.norm(x_flat)
		if norm_x < 1e-9:
			return np.zeros((self.output_size, self.output_size))

		state_x = x_flat / norm_x

		@qml.qnode(self.dev)
		def circuit():
			qml.AmplitudeEmbedding(features=state_x, wires=self.wires_row + self.wires_col, pad_with=0.0)
			qml.AmplitudeEmbedding(features=self.prep_state, wires=self.wires_ancilla, pad_with=0.0)
			qml.QFT(wires=self.wires_row)
			qml.QFT(wires=self.wires_col)

			shifts_r = [1, 1, 1, 0, 0, 0, -1, -1, -1]
			shifts_c = [1, 0, -1, 1, 0, -1, 1, 0, -1]

			for i in range(9):
				ctrl_string = bin(i)[2:].zfill(4)
				ctrl_vals = [int(x) for x in ctrl_string]

				if self.signs[i] < 0:
					ctrl_wires = self.wires_ancilla[:-1]
					target_wire = self.wires_ancilla[-1]
					c_vals = ctrl_vals[:-1]
					t_val = ctrl_vals[-1]
					if t_val == 0:
						qml.PauliX(wires=target_wire)
					qml.ctrl(qml.PauliZ, control=ctrl_wires, control_values=c_vals)(wires=target_wire)
					if t_val == 0:
						qml.PauliX(wires=target_wire)

				if shifts_r[i] != 0:
					qml.ctrl(self._apply_shift_phase, control=self.wires_ancilla, control_values=ctrl_vals)(
						shifts_r[i], self.wires_row
					)
				if shifts_c[i] != 0:
					qml.ctrl(self._apply_shift_phase, control=self.wires_ancilla, control_values=ctrl_vals)(
						shifts_c[i], self.wires_col
					)

			qml.adjoint(qml.QFT)(wires=self.wires_row)
			qml.adjoint(qml.QFT)(wires=self.wires_col)
			qml.adjoint(qml.AmplitudeEmbedding)(features=self.prep_state, wires=self.wires_ancilla, pad_with=0.0)
			return qml.state()

		full_state = circuit()
		hilbert_dim_data = 2 ** self.n_data_qubits
		output_amplitudes = full_state[:hilbert_dim_data]
		output_reconstructed = output_amplitudes.real * norm_x * self.lcu_lambda
		output_valid = output_reconstructed[self.output_indices]
		return output_valid.reshape(self.output_size, self.output_size)


def apply_scipy_conv2d(input_grid: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	return signal.correlate2d(input_grid, kernel, mode="valid")


def apply_torch_conv2d(input_grid: np.ndarray, kernel: np.ndarray) -> np.ndarray:
	input_tensor = torch.from_numpy(input_grid).float().unsqueeze(0).unsqueeze(0)
	kernel_tensor = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)
	output_tensor = F.conv2d(F.pad(input_tensor, (1, 1, 1, 1), mode="constant", value=0.0), kernel_tensor)
	return output_tensor.squeeze().cpu().numpy()


def apply_quantum_conv2d(input_grid: np.ndarray, q_engine) -> np.ndarray:
	h, w = input_grid.shape
	input_padded = np.pad(input_grid, 1, mode="constant", constant_values=0.0)
	K = q_engine.K
	stride = q_engine.output_size
	output_grid = np.zeros((h, w), dtype=np.float64)
	H_pad, W_pad = input_padded.shape

	for r in range(0, h, stride):
		for c in range(0, w, stride):
			h_valid = min(stride, h - r)
			w_valid = min(stride, w - c)
			r_end = r + K
			c_end = c + K

			if r_end > H_pad or c_end > W_pad:
				block_raw = input_padded[r : min(r_end, H_pad), c : min(c_end, W_pad)]
				res_scipy = signal.correlate2d(block_raw, q_engine.kernel, mode="valid")
				sh, sw = res_scipy.shape
				output_grid[r : r + sh, c : c + sw] = res_scipy
			else:
				block = input_padded[r:r_end, c:c_end]
				res = q_engine.run_block(block)
				output_grid[r : r + h_valid, c : c + w_valid] = res[0:h_valid, 0:w_valid]

	return output_grid


class AI4CFD_MultiBackend(nn.Module):
	"""CFD solver with PyTorch / SciPy / Quantum / Quantum_LCU backends."""

	def __init__(
		self,
		backend: str = "scipy",
		quantum_block_size: int = 16,
		w1_t=None,
		w2_t=None,
		w3_t=None,
		wA_t=None,
		w_res_t=None,
		bias_init=None,
		device_obj: Optional[torch.device] = None,
		nu_val: Optional[float] = None,
		ub_val: Optional[float] = None,
		diag_val=None,
		nlevel_val: Optional[int] = None,
	):
		super().__init__()
		self.backend = backend
		self.quantum_block_size = quantum_block_size
		self._device = _ensure_torch_device(device_obj)
		self._nu = 0.1 if nu_val is None else nu_val
		self._ub = -1.0 if ub_val is None else ub_val
		self._diag = diag_val
		self._nlevel = nlevel_val

		_w1 = w1_t if w1_t is not None else build_kernels(1.0)[0]
		_w2 = w2_t if w2_t is not None else build_kernels(1.0)[1]
		_w3 = w3_t if w3_t is not None else build_kernels(1.0)[2]
		_wA = wA_t if wA_t is not None else build_kernels(1.0)[3]
		_w_res = w_res_t if w_res_t is not None else build_kernels(1.0)[4]
		_bias = bias_init if bias_init is not None else torch.tensor([0.0], device=self._device)

		if backend == "pytorch":
			self._init_pytorch(_w1, _w2, _w3, _wA, _w_res, _bias)
		elif backend == "scipy":
			self._init_scipy(_w1, _w2, _w3, _wA, _w_res, _bias)
		elif backend == "quantum":
			self._init_quantum(_w1, _w2, _w3, _wA, _w_res, _bias)
		elif backend == "quantum_lcu":
			self._init_quantum_lcu(_w1, _w2, _w3, _wA, _w_res, _bias)
		else:
			raise ValueError("backend must be one of: pytorch, scipy, quantum, quantum_lcu")

	def _init_pytorch(self, _w1, _w2, _w3, _wA, _w_res, _bias):
		self.xadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
		self.yadv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
		self.diff = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
		self.A = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
		self.res = nn.Conv2d(1, 1, kernel_size=2, stride=2, padding=0)
		self.prol = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"))

		self.xadv.weight.data = _w2
		self.yadv.weight.data = _w3
		self.diff.weight.data = _w1
		self.A.weight.data = _wA
		self.res.weight.data = _w_res
		self.xadv.bias.data = _bias
		self.yadv.bias.data = _bias
		self.diff.bias.data = _bias
		self.A.bias.data = _bias
		self.res.bias.data = _bias

	def _init_scipy(self, _w1, _w2, _w3, _wA, _w_res, _bias):
		self.xadv_kernel = _to_numpy(_w2)[0, 0]
		self.yadv_kernel = _to_numpy(_w3)[0, 0]
		self.diff_kernel = _to_numpy(_w1)[0, 0]
		self.A_kernel = _to_numpy(_wA)[0, 0]
		self.res_kernel = _to_numpy(_w_res)[0, 0]
		self.bias_val = float(_to_numpy(_bias)[0])

	def _init_quantum(self, _w1, _w2, _w3, _wA, _w_res, _bias):
		if not PENNYLANE_AVAILABLE:
			raise ImportError("PennyLane is required for the quantum backend.")
		K = self.quantum_block_size
		xadv_np = _to_numpy(_w2)[0, 0]
		yadv_np = _to_numpy(_w3)[0, 0]
		diff_np = _to_numpy(_w1)[0, 0]
		A_np = _to_numpy(_wA)[0, 0]
		self.q_xadv = QuantumConvolutionStride2Engine(input_size=K, kernel_3x3=xadv_np)
		self.q_yadv = QuantumConvolutionStride2Engine(input_size=K, kernel_3x3=yadv_np)
		self.q_diff = QuantumConvolutionStride2Engine(input_size=K, kernel_3x3=diff_np)
		self.q_A = QuantumConvolutionStride2Engine(input_size=K, kernel_3x3=A_np)
		self.res_kernel = _to_numpy(_w_res)[0, 0]
		self.bias_val = float(_to_numpy(_bias)[0])

	def _init_quantum_lcu(self, _w1, _w2, _w3, _wA, _w_res, _bias):
		if not PENNYLANE_AVAILABLE:
			raise ImportError("PennyLane is required for the quantum backend.")
		K = self.quantum_block_size
		xadv_np = _to_numpy(_w2)[0, 0]
		yadv_np = _to_numpy(_w3)[0, 0]
		diff_np = _to_numpy(_w1)[0, 0]
		A_np = _to_numpy(_wA)[0, 0]
		self.q_lcu_xadv = QuantumConvolutionStride2Engine_LCU(input_size=K, kernel_3x3=xadv_np)
		self.q_lcu_yadv = QuantumConvolutionStride2Engine_LCU(input_size=K, kernel_3x3=yadv_np)
		self.q_lcu_diff = QuantumConvolutionStride2Engine_LCU(input_size=K, kernel_3x3=diff_np)
		self.q_lcu_A = QuantumConvolutionStride2Engine_LCU(input_size=K, kernel_3x3=A_np)
		self.res_kernel = _to_numpy(_w_res)[0, 0]
		self.bias_val = float(_to_numpy(_bias)[0])

	def _conv2d(self, input_tensor, op_name):
		if self.backend == "pytorch":
			return getattr(self, op_name)(input_tensor)
		if self.backend == "scipy":
			kernel = getattr(self, f"{op_name}_kernel")
			return self._scipy_conv2d(input_tensor, kernel)
		if self.backend == "quantum":
			q_engine = getattr(self, f"q_{op_name}")
			return self._quantum_conv2d(input_tensor, q_engine)
		if self.backend == "quantum_lcu":
			q_engine = getattr(self, f"q_lcu_{op_name}")
			return self._quantum_lcu_conv2d(input_tensor, q_engine)
		raise ValueError(f"Unknown op backend: {self.backend}")

	def _scipy_conv2d(self, input_tensor, kernel):
		input_device = input_tensor.device
		input_np = input_tensor[0, 0].detach().cpu().numpy()
		output_np = signal.correlate2d(input_np, kernel, mode="valid") + self.bias_val
		return torch.from_numpy(output_np).float().unsqueeze(0).unsqueeze(0).to(input_device)

	def _quantum_conv2d(self, input_tensor, q_engine):
		input_device = input_tensor.device
		input_np = input_tensor[0, 0].detach().cpu().numpy()
		h, w = input_np.shape
		K = q_engine.K
		stride = q_engine.output_size
		full_h, full_w = h - 2, w - 2
		full_output = np.zeros((full_h, full_w), dtype=np.float64)

		for r in range(0, full_h, stride):
			for c in range(0, full_w, stride):
				block = input_np[r : r + K, c : c + K]
				if block.shape != (K, K):
					remaining_h = min(stride, full_h - r)
					remaining_w = min(stride, full_w - c)
					patch = input_np[r : r + remaining_h + 2, c : c + remaining_w + 2]
					fallback = signal.correlate2d(patch, q_engine.kernel, mode="valid")
					fh, fw = fallback.shape
					full_output[r : r + fh, c : c + fw] = fallback
					continue

				result = q_engine.run_block(block)
				rh, rw = result.shape
				actual_h = min(rh, full_h - r)
				actual_w = min(rw, full_w - c)
				full_output[r : r + actual_h, c : c + actual_w] = result[:actual_h, :actual_w]

		output_np = full_output + self.bias_val
		return torch.from_numpy(output_np).float().unsqueeze(0).unsqueeze(0).to(input_device)

	def _quantum_lcu_conv2d(self, input_tensor, q_engine):
		input_device = input_tensor.device
		input_np = input_tensor[0, 0].detach().cpu().numpy()
		h, w = input_np.shape
		K = q_engine.K
		stride = q_engine.output_size
		full_h, full_w = h - 2, w - 2
		full_output = np.zeros((full_h, full_w), dtype=np.float64)

		for r in range(0, full_h, stride):
			for c in range(0, full_w, stride):
				block = input_np[r : r + K, c : c + K]
				if block.shape != (K, K):
					remaining_h = min(stride, full_h - r)
					remaining_w = min(stride, full_w - c)
					patch = input_np[r : r + remaining_h + 2, c : c + remaining_w + 2]
					fallback = signal.correlate2d(patch, q_engine.kernel, mode="valid")
					fh, fw = fallback.shape
					full_output[r : r + fh, c : c + fw] = fallback
					continue

				result = q_engine.run_block(block)
				rh, rw = result.shape
				actual_h = min(rh, full_h - r)
				actual_w = min(rw, full_w - c)
				full_output[r : r + actual_h, c : c + actual_w] = result[:actual_h, :actual_w]

		output_np = full_output + self.bias_val
		return torch.from_numpy(output_np).float().unsqueeze(0).unsqueeze(0).to(input_device)

	def _restrict(self, input_tensor):
		if self.backend == "pytorch":
			return self.res(input_tensor)
		input_device = input_tensor.device
		input_np = input_tensor[0, 0].detach().cpu().numpy()
		output_np = signal.correlate2d(input_np, self.res_kernel, mode="valid")[::2, ::2]
		return torch.from_numpy(output_np).float().unsqueeze(0).unsqueeze(0).to(input_device)

	def _prolong(self, input_tensor):
		if self.backend == "pytorch":
			return self.prol(input_tensor)
		return F.interpolate(input_tensor, scale_factor=2, mode="nearest")

	def solid_body(self, values_u, values_v, sigma, dt):
		values_u = values_u / (1 + dt * sigma)
		values_v = values_v / (1 + dt * sigma)
		return values_u, values_v

	def F_cycle_MG(self, values_uu, values_vv, values_p, values_pp, iteration, diag_val, dt, nlevel_val):
		b = -(self._conv2d(values_uu, "xadv") + self._conv2d(values_vv, "yadv")) / dt

		for _ in range(iteration):
			w = torch.zeros((1, 1, 1, 1), device=values_p.device)
			padded_p = boundary_condition_2D_p(values_p, values_pp)
			r = self._conv2d(padded_p, "A") - b
			r_s = [r]

			for _ in range(1, nlevel_val):
				r = self._restrict(r)
				r_s.append(r)

			for i in reversed(range(1, nlevel_val)):
				ww = boundary_condition_2D_cw(w)
				A_w = self._conv2d(ww, "A")
				w = w - A_w / diag_val + r_s[i] / diag_val
				w = self._prolong(w)

			values_p = values_p - w
			padded_p = boundary_condition_2D_p(values_p, values_pp)
			values_p = values_p - self._conv2d(padded_p, "A") / diag_val + b / diag_val

		return values_p, w, r

	def forward(self, values_u, values_uu, values_v, values_vv, values_p, values_pp, sigma, b_uu, b_vv, dt, iteration):
		_nu = self._nu
		_ub = self._ub
		_diag = self._diag
		_nlevel = self._nlevel
		if _diag is None or _nlevel is None:
			raise ValueError("diag and nlevel must be provided to AI4CFD_MultiBackend")

		values_uu = boundary_condition_2D_u(values_u, values_uu, _ub)
		values_vv = boundary_condition_2D_v(values_v, values_vv, _ub)
		values_pp = boundary_condition_2D_p(values_p, values_pp)

		Grapx_p = self._conv2d(values_pp, "xadv") * dt
		Grapy_p = self._conv2d(values_pp, "yadv") * dt

		ADx_u = self._conv2d(values_uu, "xadv")
		ADy_u = self._conv2d(values_uu, "yadv")
		ADx_v = self._conv2d(values_vv, "xadv")
		ADy_v = self._conv2d(values_vv, "yadv")
		AD2_u = self._conv2d(values_uu, "diff")
		AD2_v = self._conv2d(values_vv, "diff")

		b_u = values_u + 0.5 * (_nu * AD2_u * dt - values_u * ADx_u * dt - values_v * ADy_u * dt) - Grapx_p
		b_v = values_v + 0.5 * (_nu * AD2_v * dt - values_u * ADx_v * dt - values_v * ADy_v * dt) - Grapy_p
		values_u, values_v = self.solid_body(b_u, b_v, sigma, dt)

		b_uu = boundary_condition_2D_u(values_u, b_uu, _ub)
		b_vv = boundary_condition_2D_v(values_v, b_vv, _ub)
		ADx_u = self._conv2d(b_uu, "xadv")
		ADy_u = self._conv2d(b_uu, "yadv")
		ADx_v = self._conv2d(b_vv, "xadv")
		ADy_v = self._conv2d(b_vv, "yadv")
		AD2_u = self._conv2d(b_uu, "diff")
		AD2_v = self._conv2d(b_vv, "diff")

		values_u = values_u + _nu * AD2_u * dt - b_u * ADx_u * dt - b_v * ADy_u * dt - Grapx_p
		values_v = values_v + _nu * AD2_v * dt - b_u * ADx_v * dt - b_v * ADy_v * dt - Grapy_p
		values_u, values_v = self.solid_body(values_u, values_v, sigma, dt)

		values_uu = boundary_condition_2D_u(values_u, values_uu, _ub)
		values_vv = boundary_condition_2D_v(values_v, values_vv, _ub)
		values_p, w, r = self.F_cycle_MG(values_uu, values_vv, values_p, values_pp, iteration, _diag, dt, _nlevel)

		values_pp = boundary_condition_2D_p(values_p, values_pp)
		values_u = values_u - self._conv2d(values_pp, "xadv") * dt
		values_v = values_v - self._conv2d(values_pp, "yadv") * dt
		values_u, values_v = self.solid_body(values_u, values_v, sigma, dt)
		return values_u, values_v, values_p, w, r


def compare_scipy_and_qft_consistency(input_size: int = 4, seed: int = 42, use_lcu: bool = True) -> Dict[str, Dict[str, float]]:
	"""Compare SciPy convolution vs quantum convolution for one block.

	Returns a metrics dict so the caller can inspect max and relative errors.
	"""
	if not PENNYLANE_AVAILABLE:
		raise ImportError("PennyLane is required for the quantum consistency test.")

	_, _, _, wA, _, _ = build_kernels(1.0)
	kernel = _to_numpy(wA)[0, 0]
	rng = np.random.default_rng(seed)
	block = rng.random((input_size, input_size)) * 5.0
	ref = signal.correlate2d(block, kernel, mode="valid")

	results: Dict[str, Dict[str, float]] = {}

	engines: List[Tuple[str, Any]] = [("quantum_origin", QuantumConvolutionStride2Engine)]
	if use_lcu:
		engines.append(("quantum_lcu", QuantumConvolutionStride2Engine_LCU))

	for name, engine_cls in engines:
		engine = engine_cls(input_size=input_size, kernel_3x3=kernel)
		out = engine.run_block(block)
		diff = out - ref
		max_err = float(np.max(np.abs(diff)))
		rel_err = float(np.linalg.norm(diff) / (np.linalg.norm(ref) + 1e-12))
		results[name] = {
			"max_abs_error": max_err,
			"relative_error": rel_err,
			"output_rows": float(out.shape[0]),
			"output_cols": float(out.shape[1]),
		}
		print(f"[{name}] max_abs_error={max_err:.6e}, relative_error={rel_err:.6e}")

	return results


def run_flow_past_block_demo(
	backend: str = "scipy",
	nx: int = 32,
	ny: int = 16,
	nsteps: int = 2,
	iteration: int = 2,
	quantum_block_size: int = 8,
	solid_body: Tuple[float, float, float, float] = (0.25, 0.5, 0.05, 0.2),
) -> Dict[str, Any]:
	"""Run a small flow-past-block CFD smoke test."""
	if backend in {"quantum", "quantum_lcu"} and not PENNYLANE_AVAILABLE:
		raise ImportError("PennyLane is required for the quantum backend.")

	dt = 0.05
	dx = 1.0
	dy = 1.0
	nu = 0.1
	ub = -1.0
	nlevel = int(math.log(ny, 2)) + 1

	w1, w2, w3, wA, w_res, diag = build_kernels(dx)
	device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	values_u, values_v, values_p, values_uu, values_vv, values_pp, b_uu, b_vv = create_tensors_2D(nx, ny)
	values_u = values_u.to(device_obj)
	values_v = values_v.to(device_obj)
	values_p = values_p.to(device_obj)
	values_uu = values_uu.to(device_obj)
	values_vv = values_vv.to(device_obj)
	values_pp = values_pp.to(device_obj)
	b_uu = b_uu.to(device_obj)
	b_vv = b_vv.to(device_obj)
	sigma = torch.zeros((1, 1, ny, nx), device=device_obj)

	rel_x, rel_y, rel_size_x, rel_size_y = solid_body
	cor_x = int(rel_x * nx)
	cor_y = int(rel_y * ny)
	size_x = int(rel_size_x * nx)
	size_y = int(rel_size_y * ny)
	sigma[0, 0, max(0, cor_y - size_y) : min(ny, cor_y + size_y), max(0, cor_x - size_x) : min(nx, cor_x + size_x)] = 1e8

	model = AI4CFD_MultiBackend(
		backend=backend,
		quantum_block_size=quantum_block_size,
		w1_t=w1,
		w2_t=w2,
		w3_t=w3,
		wA_t=wA,
		w_res_t=w_res,
		device_obj=device_obj,
		nu_val=nu,
		ub_val=ub,
		diag_val=diag,
		nlevel_val=nlevel,
	).to(device_obj)

	residual_history: List[float] = []
	wall_start = time.time()
	with torch.no_grad():
		for step in range(nsteps):
			values_u, values_v, values_p, w, r = model(
				values_u,
				values_uu,
				values_v,
				values_vv,
				values_p,
				values_pp,
				sigma,
				b_uu,
				b_vv,
				dt,
				iteration,
			)
			residual = float(np.max(np.abs(_to_numpy(w))))
			residual_history.append(residual)
			print(f"[flow-demo] step={step + 1}/{nsteps}, residual={residual:.6e}")

	elapsed = time.time() - wall_start
	print(f"[flow-demo] backend={backend}, elapsed={elapsed:.4f}s")
	return {
		"backend": backend,
		"elapsed": elapsed,
		"residual_history": residual_history,
		"final_residual": residual_history[-1] if residual_history else None,
	}


def main():
	print("=== SciPy vs QFT consistency check ===")
	compare_scipy_and_qft_consistency(input_size=4, use_lcu=True)
	print("\n=== Flow past block smoke test ===")
	run_flow_past_block_demo(backend="scipy", nx=32, ny=16, nsteps=1, iteration=1, quantum_block_size=8)


if __name__ == "__main__":
	main()
