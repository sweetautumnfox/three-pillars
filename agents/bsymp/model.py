from __future__ import annotations

import numpy as np
from scipy.special import digamma, expit as sigmoid, logit as inv_sigmoid

from .config import BSyMPConfig


def _compute_ln_params(params: np.ndarray) -> np.ndarray:
    """Compute digamma log-expectations for Dirichlet blocks.

    Each ``params[..., :, c]`` column is a Dirichlet concentration vector.

    Args:
        params: shape ``(..., 2, 2)``.

    Returns:
        Log-expectations, same shape.
    """
    col_sums = params.sum(axis=-2, keepdims=True)
    return digamma(params) - digamma(col_sums)


def _soft_onehot_vec(v: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to soft one-hot 2-vectors.

    Args:
        v: shape ``(N,)`` with values in [0, 1].

    Returns:
        shape ``(N, 2)`` where row *i* is ``[1 - v[i], v[i]]``.
    """
    return np.stack([1.0 - v, v], axis=-1)


class BSyMPModel:
    """Core BSyMP generative model with variational inference.

    Implements Eqs. 13-17 from Tazawa & Isomura (2026).
    """

    def __init__(self, N_o: int, N_s: int, config: BSyMPConfig):
        self.N_o = N_o
        self.N_s = N_s

        # Dirichlet concentrations for likelihood A — (N_o, N_s, 2, 2)
        self.a = np.full((N_o, N_s, 2, 2), config.a_prior, dtype=np.float64)

        # Dirichlet concentrations for transition B — (N_s, N_s, 2, 2)
        self.b = np.full((N_s, N_s, 2, 2), config.b_prior, dtype=np.float64)

        # Connectivity posteriors
        self.c_A = np.full((N_o, N_s), config.c_A_init, dtype=np.float64)
        self.c_B = np.full((N_s, N_s), config.c_B_init, dtype=np.float64)

        # Hidden state posteriors
        self.s = np.full(N_s, 0.5, dtype=np.float64)
        self.s_prev = np.full(N_s, 0.5, dtype=np.float64)

        # Session accumulators for connectivity updates
        self.acc_A = np.zeros((N_o, N_s, 2, 2), dtype=np.float64)
        self.acc_B = np.zeros((N_s, N_s, 2, 2), dtype=np.float64)

        self.timestep = 0
        self.session_length = config.session_length

    # ------------------------------------------------------------------
    # Eq. 13 — state inference
    # ------------------------------------------------------------------

    def infer_states(self, o_t: np.ndarray) -> np.ndarray:
        """Infer hidden state posteriors from observation (Eq. 13).

        Args:
            o_t: binary observation, shape ``(N_o,)``.

        Returns:
            Updated posterior ``s_t``, shape ``(N_s,)``, values in [0, 1].
        """
        lnA = _compute_ln_params(self.a)  # (N_o, N_s, 2, 2)
        lnB = _compute_ln_params(self.b)  # (N_s, N_s, 2, 2)

        o_oh = _soft_onehot_vec(o_t)  # (N_o, 2)
        s_prev_oh = _soft_onehot_vec(self.s_prev)  # (N_s, 2)

        # Likelihood messages — lnA[i,n] @ o_oh[i] for all (i, n)
        msg_A = np.einsum("ijkl,il->ijk", lnA, o_oh)  # (N_o, N_s, 2)
        contrib_A = (self.c_A * (msg_A[:, :, 1] - msg_A[:, :, 0])).sum(
            axis=0
        )  # (N_s,)

        # Transition messages — lnB[n,j] @ s_prev_oh[j] for all (n, j)
        msg_B = np.einsum("ijkl,jl->ijk", lnB, s_prev_oh)  # (N_s, N_s, 2)
        contrib_B = (self.c_B * (msg_B[:, :, 1] - msg_B[:, :, 0])).sum(
            axis=1
        )  # (N_s,)

        logit = contrib_A + contrib_B

        self.s_prev = self.s.copy()
        self.s = sigmoid(logit)
        return self.s

    # ------------------------------------------------------------------
    # Eq. 14 — likelihood parameter update
    # ------------------------------------------------------------------

    def update_likelihood(self, o_t: np.ndarray) -> None:
        """Incremental update of Dirichlet concentrations for A (Eq. 14)."""
        o_oh = _soft_onehot_vec(o_t)  # (N_o, 2)
        s_oh = _soft_onehot_vec(self.s)  # (N_s, 2)

        outer = np.einsum("ik,jl->ijkl", o_oh, s_oh)  # (N_o, N_s, 2, 2)
        self.a += self.c_A[:, :, np.newaxis, np.newaxis] * outer
        self.acc_A += outer

    # ------------------------------------------------------------------
    # Eq. 15 — transition parameter update
    # ------------------------------------------------------------------

    def update_transition(self) -> None:
        """Incremental update of Dirichlet concentrations for B (Eq. 15)."""
        s_oh = _soft_onehot_vec(self.s)  # (N_s, 2)
        s_prev_oh = _soft_onehot_vec(self.s_prev)  # (N_s, 2)

        outer = np.einsum("ik,jl->ijkl", s_oh, s_prev_oh)  # (N_s, N_s, 2, 2)
        self.b += self.c_B[:, :, np.newaxis, np.newaxis] * outer
        self.acc_B += outer

    # ------------------------------------------------------------------
    # Eqs. 16-17 — connectivity update (called every session)
    # ------------------------------------------------------------------

    def update_connectivity(self) -> None:
        """Update connectivity parameters C_A, C_B (Eqs. 16-17).

        The +ln2 term in the paper is *inside* the per-timestep sum,
        so over a session of length L the total bias is L * ln2.
        """
        LN2 = np.log(2.0)
        ln2_total = LN2 * self.session_length

        # --- C_A (Eq. 16) ---
        lnA = _compute_ln_params(self.a)
        frob_A = (lnA * self.acc_A).sum(axis=(-2, -1))  # (N_o, N_s)
        c_A_clipped = np.clip(self.c_A, 1e-7, 1.0 - 1e-7)
        self.c_A = sigmoid(inv_sigmoid(c_A_clipped) + frob_A + ln2_total)

        # --- C_B (Eq. 17) ---
        lnB = _compute_ln_params(self.b)
        frob_B = (lnB * self.acc_B).sum(axis=(-2, -1))  # (N_s, N_s)
        c_B_clipped = np.clip(self.c_B, 1e-7, 1.0 - 1e-7)
        self.c_B = sigmoid(inv_sigmoid(c_B_clipped) + frob_B + ln2_total)

        # Reset accumulators
        self.acc_A = np.zeros_like(self.acc_A)
        self.acc_B = np.zeros_like(self.acc_B)

    # ------------------------------------------------------------------
    # Full timestep
    # ------------------------------------------------------------------

    def step(self, o_t: np.ndarray) -> np.ndarray:
        """Process one timestep: infer states, update parameters.

        Returns:
            Updated state posterior, shape ``(N_s,)``.
        """
        s_t = self.infer_states(o_t)
        self.update_likelihood(o_t)
        self.update_transition()

        self.timestep += 1

        if self.timestep % self.session_length == 0:
            self.update_connectivity()

        return s_t

    # ------------------------------------------------------------------
    # Neurogenesis support
    # ------------------------------------------------------------------

    def add_states(
        self,
        n_new: int,
        config: BSyMPConfig,
        c_A_new: np.ndarray | None = None,
        c_B_new_col: np.ndarray | None = None,
        c_B_new_row: np.ndarray | None = None,
    ) -> None:
        """Extend the model by ``n_new`` hidden state dimensions.

        New states are appended after existing ones (action nodes at
        indices 0..n_arms-1 are unaffected).

        Args:
            n_new: number of states to add.
            config: for default prior values.
            c_A_new: initial C_A for new states, shape ``(N_o, n_new)``.
            c_B_new_col: new columns for C_B, shape ``(N_s_old, n_new)``.
            c_B_new_row: new rows for C_B, shape ``(n_new, N_s_new)``.
        """
        old_N_s = self.N_s
        self.N_s += n_new

        # --- a: (N_o, N_s, 2, 2) ---
        a_ext = np.full((self.N_o, n_new, 2, 2), config.a_prior)
        self.a = np.concatenate([self.a, a_ext], axis=1)

        # --- b: (N_s, N_s, 2, 2) — new rows and columns ---
        b_col = np.full((old_N_s, n_new, 2, 2), config.b_prior)
        b_wide = np.concatenate([self.b, b_col], axis=1)
        b_row = np.full((n_new, self.N_s, 2, 2), config.b_prior)
        self.b = np.concatenate([b_wide, b_row], axis=0)

        # --- c_A: (N_o, N_s) ---
        if c_A_new is None:
            c_A_new = np.full((self.N_o, n_new), config.c_A_init)
        self.c_A = np.concatenate([self.c_A, c_A_new], axis=1)

        # --- c_B: (N_s, N_s) — new rows and columns ---
        if c_B_new_col is None:
            c_B_new_col = np.full((old_N_s, n_new), config.c_B_init)
        if c_B_new_row is None:
            c_B_new_row = np.full((n_new, self.N_s), config.c_B_init)
        c_B_wide = np.concatenate([self.c_B, c_B_new_col], axis=1)
        self.c_B = np.concatenate([c_B_wide, c_B_new_row], axis=0)

        # --- State posteriors ---
        self.s = np.concatenate([self.s, np.full(n_new, 0.5)])
        self.s_prev = np.concatenate([self.s_prev, np.full(n_new, 0.5)])

        # --- Accumulators ---
        acc_A_ext = np.zeros((self.N_o, n_new, 2, 2))
        self.acc_A = np.concatenate([self.acc_A, acc_A_ext], axis=1)

        acc_B_col = np.zeros((old_N_s, n_new, 2, 2))
        acc_B_wide = np.concatenate([self.acc_B, acc_B_col], axis=1)
        acc_B_row = np.zeros((n_new, self.N_s, 2, 2))
        self.acc_B = np.concatenate([acc_B_wide, acc_B_row], axis=0)
