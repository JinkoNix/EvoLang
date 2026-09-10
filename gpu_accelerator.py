"""
gpu_accelerator.py — High-Performance PyTorch GPU Acceleration Engine for EvoLang.
Accelerates 7D semantic distance matrices, batch synonym pruning, and k-NN candidate queries
via CUDA tensor kernels. Falls back gracefully to vectorized NumPy if PyTorch/CUDA is unavailable.
"""

from __future__ import annotations

import math
from typing import Sequence

try:
    import torch
    HAS_TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    HAS_TORCH = False
    DEVICE = "cpu"

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class GPUSemanticEngine:
    """CUDA-accelerated 7D tensor mathematics for large-scale lexicons (10k to 100k words)."""

    device = DEVICE
    has_cuda = HAS_TORCH and torch.cuda.is_available()

    @classmethod
    def get_hardware_info(cls) -> str:
        if cls.has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return f"PyTorch CUDA Engine ACTIVE [GPU: {gpu_name} | VRAM: {vram_gb:.2f} GB]"
        elif HAS_TORCH:
            return "PyTorch CPU Vectorized Engine ACTIVE [No CUDA GPU detected]"
        elif HAS_NUMPY:
            return "NumPy SIMD Vectorized Engine ACTIVE"
        return "Pure Python Fallback [Warning: Install PyTorch for 100x GPU acceleration]"

    @classmethod
    def batch_pairwise_distances(
        cls,
        coords_matrix: Sequence[Sequence[float]],
        weights: Sequence[float] | None = None,
    ) -> list[list[float]] | object:
        """Computes all-to-all weighted Euclidean distance matrix in parallel on GPU."""
        if cls.has_cuda:
            coords_t = torch.tensor(coords_matrix, dtype=torch.float32, device=cls.device)
            if weights is not None:
                w_t = torch.sqrt(torch.tensor(weights, dtype=torch.float32, device=cls.device))
                coords_t = coords_t * w_t
            # GPU pairwise distance tensor kernel
            dist_matrix = torch.cdist(coords_t, coords_t, p=2.0)
            return dist_matrix

        elif HAS_NUMPY:
            mat = np.array(coords_matrix, dtype=np.float32)
            if weights is not None:
                w = np.sqrt(np.array(weights, dtype=np.float32))
                mat = mat * w
            diff = mat[:, np.newaxis, :] - mat[np.newaxis, :, :]
            return np.sqrt(np.sum(diff ** 2, axis=-1))

        # Pure Python fallback
        n = len(coords_matrix)
        out = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = math.sqrt(sum((coords_matrix[i][k] - coords_matrix[j][k]) ** 2 for k in range(7)))
                out[i][j] = d
                out[j][i] = d
        return out

    @classmethod
    def filter_candidate_density_gpu(
        cls,
        candidates_coords: Sequence[Sequence[float]],
        existing_coords: Sequence[Sequence[float]],
        weights: Sequence[float],
        d_min: float,
    ) -> list[bool]:
        """
        Evaluates thousands of new candidate words against the entire living lexicon
        simultaneously in GPU VRAM, returning a boolean acceptance mask.
        """
        if not existing_coords:
            return [True] * len(candidates_coords)

        if cls.has_cuda:
            cand_t = torch.tensor(candidates_coords, dtype=torch.float32, device=cls.device)
            exist_t = torch.tensor(existing_coords, dtype=torch.float32, device=cls.device)
            w_t = torch.sqrt(torch.tensor(weights, dtype=torch.float32, device=cls.device))

            cand_t = cand_t * w_t
            exist_t = exist_t * w_t

            # Batch distance kernel [N_candidates, N_existing]
            dists = torch.cdist(cand_t, exist_t, p=2.0)
            min_dists, _ = torch.min(dists, dim=1)
            accepted_mask = (min_dists >= d_min).tolist()
            return accepted_mask

        elif HAS_NUMPY:
            cand_mat = np.array(candidates_coords, dtype=np.float32)
            exist_mat = np.array(existing_coords, dtype=np.float32)
            w_mat = np.sqrt(np.array(weights, dtype=np.float32))

            cand_mat = cand_mat * w_mat
            exist_mat = exist_mat * w_mat

            # Broadcast distance computation
            diff = cand_mat[:, np.newaxis, :] - exist_mat[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
            min_dists = np.min(dists, axis=1)
            return (min_dists >= d_min).tolist()

        # Pure Python fallback
        accepted = []
        w0, w1, w2, w3, w4, w5, w6 = weights
        d_min_sq = d_min ** 2
        for c in candidates_coords:
            c0, c1, c2, c3, c4, c5, c6 = c
            valid = True
            for v0, v1, v2, v3, v4, v5, v6 in existing_coords:
                if abs(c0 - v0) > d_min and abs(c1 - v1) > d_min:
                    continue
                dist_sq = (
                    w0 * (c0 - v0)**2 + w1 * (c1 - v1)**2 + w2 * (c2 - v2)**2 +
                    w3 * (c3 - v3)**2 + w4 * (c4 - v4)**2 + w5 * (c5 - v5)**2 +
                    w6 * (c6 - v6)**2
                )
                if dist_sq < d_min_sq:
                    valid = False
                    break
            accepted.append(valid)
        return accepted