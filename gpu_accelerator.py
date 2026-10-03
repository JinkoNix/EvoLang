"""
gpu_accelerator.py — Unified Vectorized & GPU Acceleration Engine for EvoLang.
Provides zero-copy batched semantic distance filtering, k-NN lexical queries,
and vectorized territorial propagation. Supports PyTorch CUDA with NumPy fallback.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

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
    """Unified hardware acceleration for 7D semantic space and 2D spatial geography."""

    device = DEVICE
    has_cuda = HAS_TORCH and torch.cuda.is_available()

    @classmethod
    def get_hardware_info(cls) -> str:
        if cls.has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return f"PyTorch CUDA Engine ACTIVE [GPU: {gpu_name} | VRAM: {vram_gb:.2f} GB]"
        elif HAS_TORCH:
            return "PyTorch CPU Vectorized Engine ACTIVE"
        elif HAS_NUMPY:
            return "NumPy SIMD Vectorized Engine ACTIVE"
        return "Pure Python Fallback Engine ACTIVE"

    # =========================================================================
    # 1. BATCHED 7D SEMANTIC DISTANCE ENGINE
    # =========================================================================

    @classmethod
    def batch_pairwise_distances(
        cls,
        coords_matrix: Sequence[Sequence[float]],
        weights: Sequence[float] | None = None,
    ) -> list[list[float]] | object:
        """Computes all-to-all weighted Euclidean distance matrix in parallel."""
        if not coords_matrix:
            return []

        if cls.has_cuda:
            coords_t = torch.tensor(coords_matrix, dtype=torch.float32, device=cls.device)
            if weights is not None:
                w_t = torch.sqrt(torch.tensor(weights, dtype=torch.float32, device=cls.device))
                coords_t = coords_t * w_t
            return torch.cdist(coords_t, coords_t, p=2.0)

        elif HAS_NUMPY:
            mat = np.asarray(coords_matrix, dtype=np.float32)
            if weights is not None:
                w = np.sqrt(np.asarray(weights, dtype=np.float32))
                mat = mat * w
            diff = mat[:, np.newaxis, :] - mat[np.newaxis, :, :]
            return np.sqrt(np.sum(diff ** 2, axis=-1))

        n = len(coords_matrix)
        out = [[0.0] * n for _ in range(n)]
        w = weights or [1.0] * 7
        for i in range(n):
            for j in range(i + 1, n):
                d = math.sqrt(sum(w[k] * (coords_matrix[i][k] - coords_matrix[j][k]) ** 2 for k in range(7)))
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
        """Validates proposed candidates against all living words in a single batch tensor call."""
        if not candidates_coords:
            return []
        if not existing_coords:
            return [True] * len(candidates_coords)

        d_min_sq = d_min ** 2

        if cls.has_cuda:
            cand_t = torch.tensor(candidates_coords, dtype=torch.float32, device=cls.device)
            exist_t = torch.tensor(existing_coords, dtype=torch.float32, device=cls.device)
            w_t = torch.tensor(weights, dtype=torch.float32, device=cls.device).view(1, 1, 7)

            diff = cand_t.unsqueeze(1) - exist_t.unsqueeze(0)
            weighted_d_sq = torch.sum(w_t * (diff ** 2), dim=-1)
            min_dists, _ = torch.min(weighted_d_sq, dim=1)
            return (min_dists >= d_min_sq).tolist()

        elif HAS_NUMPY:
            cand_arr = np.asarray(candidates_coords, dtype=np.float32)
            exist_arr = np.asarray(existing_coords, dtype=np.float32)
            w_arr = np.asarray(weights, dtype=np.float32).reshape(1, 1, 7)

            diff = cand_arr[:, np.newaxis, :] - exist_arr[np.newaxis, :, :]
            weighted_d_sq = np.sum(w_arr * (diff ** 2), axis=-1)
            min_dists = np.min(weighted_d_sq, axis=1)
            return (min_dists >= d_min_sq).tolist()

        accepted = []
        w0, w1, w2, w3, w4, w5, w6 = weights
        for c in candidates_coords:
            valid = True
            c0, c1, c2, c3, c4, c5, c6 = c
            for e in existing_coords:
                d_sq = (
                    w0 * (c0 - e[0]) ** 2 + w1 * (c1 - e[1]) ** 2 + w2 * (c2 - e[2]) ** 2 +
                    w3 * (c3 - e[3]) ** 2 + w4 * (c4 - e[4]) ** 2 + w5 * (c5 - e[5]) ** 2 +
                    w6 * (c6 - e[6]) ** 2
                )
                if d_sq < d_min_sq:
                    valid = False
                    break
            accepted.append(valid)
        return accepted

    @classmethod
    def find_nearest_semantic_neighbor(
        cls,
        target_coords: Sequence[float],
        candidate_pool: Sequence[Sequence[float]],
        weights: Sequence[float],
    ) -> Tuple[int, float]:
        """Fast k-NN query for loanwords and calquing across languages."""
        if not candidate_pool:
            return 0, 0.0

        if cls.has_cuda:
            t_t = torch.tensor(target_coords, dtype=torch.float32, device=cls.device).view(1, 7)
            p_t = torch.tensor(candidate_pool, dtype=torch.float32, device=cls.device)
            w_t = torch.tensor(weights, dtype=torch.float32, device=cls.device).view(1, 7)

            diff = (p_t - t_t) ** 2
            dists = torch.sqrt(torch.sum(diff * w_t, dim=1))
            best_idx = int(torch.argmin(dists).item())
            return best_idx, float(dists[best_idx].item())

        elif HAS_NUMPY:
            t_arr = np.asarray(target_coords, dtype=np.float32)
            p_arr = np.asarray(candidate_pool, dtype=np.float32)
            w_arr = np.asarray(weights, dtype=np.float32)

            diff = (p_arr - t_arr) ** 2
            dists = np.sqrt(np.sum(diff * w_arr, axis=1))
            best_idx = int(np.argmin(dists))
            return best_idx, float(dists[best_idx])

        w0, w1, w2, w3, w4, w5, w6 = weights
        t0, t1, t2, t3, t4, t5, t6 = target_coords
        best_idx, min_dist = 0, float("inf")
        for i, c in enumerate(candidate_pool):
            d = math.sqrt(
                w0 * (t0 - c[0]) ** 2 + w1 * (t1 - c[1]) ** 2 + w2 * (t2 - c[2]) ** 2 +
                w3 * (t3 - c[3]) ** 2 + w4 * (t4 - c[4]) ** 2 + w5 * (t5 - c[5]) ** 2 +
                w6 * (t6 - c[6]) ** 2
            )
            if d < min_dist:
                min_dist = d
                best_idx = i
        return best_idx, min_dist

    # =========================================================================
    # 2. VECTORIZED 2D TERRITORIAL PROPAGATION (NON-TOROIDAL PLANAR)
    # =========================================================================

    @classmethod
    def propagate_territory_fast(
        cls,
        res: int,
        settlement_seeds: list[tuple[int, int, int]],  # (row, col, civ_id)
        friction_matrix: Sequence[Sequence[float]],    # [res, res] float32
        civ_max_cells: dict[int, int],                 # civ_id -> max allowed cells
    ) -> tuple[dict[tuple[int, int], int], dict[int, int]]:
        """
        Fast parallel 8-neighbour Euclidean wavefront propagation.
        Strict planar boundary clamping completely prevents top-to-bottom and left-to-right world wrapping.
        """
        if not HAS_NUMPY:
            return {}, {}

        f_arr = np.asarray(friction_matrix, dtype=np.float32)
        grid_owner = np.zeros((res, res), dtype=np.int32)
        grid_cost = np.full((res, res), 1e8, dtype=np.float32)

        for r, c, cid in settlement_seeds:
            if 0 <= r < res and 0 <= c < res:
                grid_owner[r, c] = cid
                grid_cost[r, c] = 0.0

        neighbors_8 = [
            (-1, 0, 1.000), (1, 0, 1.000), (0, -1, 1.000), (0, 1, 1.000),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)
        ]

        num_passes = max(16, int(res * 0.50))
        for _ in range(num_passes):
            for dr, dc, dist_mult in neighbors_8:
                s_cost = np.roll(grid_cost, (dr, dc), axis=(0, 1))
                s_owner = np.roll(grid_owner, (dr, dc), axis=(0, 1))

                # STRICT NON-TOROIDAL PLANAR BOUNDARY CLAMPING:
                # Clears wrapped edges so world never wraps top-to-bottom or left-to-right
                if dr > 0:
                    s_cost[:dr, :] = 1e8; s_owner[:dr, :] = 0
                elif dr < 0:
                    s_cost[dr:, :] = 1e8; s_owner[dr:, :] = 0
                if dc > 0:
                    s_cost[:, :dc] = 1e8; s_owner[:, :dc] = 0
                elif dc < 0:
                    s_cost[:, dc:] = 1e8; s_owner[:, dc:] = 0

                step_cost = s_cost + (f_arr * dist_mult)
                update_mask = (step_cost < grid_cost) & (s_owner > 0) & (f_arr < 999.0)
                grid_cost[update_mask] = step_cost[update_mask]
                grid_owner[update_mask] = s_owner[update_mask]

        claims: dict[tuple[int, int], int] = {}
        cell_counts: dict[int, int] = {cid: 0 for cid in civ_max_cells}

        valid_r, valid_c = np.where((grid_owner > 0) & (f_arr < 999.0))
        valid_costs = grid_cost[valid_r, valid_c]
        sort_order = np.argsort(valid_costs)

        for idx in sort_order:
            r = int(valid_r[idx])
            c = int(valid_c[idx])
            cid = int(grid_owner[r, c])
            if cell_counts.get(cid, 0) < civ_max_cells.get(cid, 9999):
                claims[(r, c)] = cid
                cell_counts[cid] += 1

        return claims, cell_counts