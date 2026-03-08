import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

# Cell states
EMPTY = np.uint8(0)
ALIVE = np.uint8(1)
NECROTIC = np.uint8(2)

# Activity labels (for oxygen uptake)
ACT_PROLIF = np.uint8(0)
ACT_QUIESC = np.uint8(1)


@dataclass
class Params:
    """Simulation parameters for the model of the Hybrid Cellular Automata minimal model (oxygen-only)."""
    # Lattice
    N: int = 200
    seed_radius: int = 6

    # Runtime
    steps: int = 150
    rng_seed: int = 0

    # Oxygen field (dimensionless)
    c0: float = 1.0
    Dc: float = 0.05
    Dt: float = 0.02
    d: float = 1.0

    # Uptake
    rc: float = 0.15
    q: float = 5.0  # quiescent consumes rc/q

    # Life-cycle thresholds
    c_apoptosis_threshold: float = 0.15 # apoptosis if oxygen below this (Fig. 5 behaviour)
    c_quiescence_threshold: int = 3     # quiescence if n_neigh > 3

    # Cell cycle (hours)
    Dt_age_inc: float = 16.0
    Ap: float = 16.0
    Ap_s: float = 8.0  # Ap/2

    # Mutations
    p: float = 0.01
    s: float = 0.25


def moore_neighbors_count(state: np.ndarray) -> np.ndarray:
    """Compute Moore-neighborhood alive+necrotic neighbor count for each site.
    Params: state (N,N) uint8.
    Returns: n_neigh (N,N) int16.
    """
    occ = (state != EMPTY).astype(np.int16)
    n = np.zeros_like(occ, dtype=np.int16)
    n += np.roll(np.roll(occ,1,0), 1, 1)
    n += np.roll(occ, 1, 0)
    n += np.roll(np.roll(occ,1,0), -1, 1)
    n += np.roll(occ, 1, 1)
    n += np.roll(occ, -1, 1)
    n += np.roll(np.roll(occ,-1,0), 1, 1)
    n += np.roll(occ, -1, 0)
    n += np.roll(np.roll(occ,-1,0), -1, 1)
    return n


def laplacian_5pt(u: np.ndarray, dx: float) -> np.ndarray:
    """5-point Laplacian with periodic interior; boundaries handled separately.
    Params: u (N,N), dx.
    Returns: lap (N,N).
    """
    return (
        np.roll(u,1,0) + np.roll(u,-1,0) + np.roll(u,1,1) + np.roll(u,-1,1) - 4.0 * u
    ) / (dx * dx)


def set_dirichlet_boundary(u: np.ndarray, value: float) -> None:
    """Set Dirichlet boundary on all edges in-place.
    Params: u (N,N), value.
    Returns: None.
    """
    u[0, :] = value
    u[-1, :] = value
    u[:, 0] = value
    u[:, -1] = value


def oxygen_step_explicit(c: np.ndarray, alpha: np.ndarray, p: Params) -> np.ndarray:
    """One explicit Euler step for oxygen reaction-diffusion.
    Params: c (N,N), alpha (N,N), p.
    Returns: c_new (N,N).
    """
    c_new = c + p.Dt * (p.Dc * laplacian_5pt(c, p.d) - alpha * c)
    c_new = np.maximum(c_new, 0.0)
    set_dirichlet_boundary(c_new, p.c0)
    return c_new


def transfer_sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid transfer function from the paper: T(x) = 1 / (1 + exp(-2x)).
    Params: x array.
    Returns: transformed array.
    """
    return 1.0 / (1.0 + np.exp(-2.0 * x))


def init_minimal_network_genotype(p: Params) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize a minimal 2->2->3 response network (inputs: oxygen, crowding; outputs: P/Q/A).
    Params: p.
    Returns: w_in_hidden (2,2), W_hidden_out (3,2), theta_hidden (2,), phi_out (3,).
    """
    # Input vector x = [c, n_norm], with n_norm = n_neigh / 8.
    # Hidden nodes:
    # hN: "crowding gate" active for n_norm > 3/8.
    # hA: "hypoxia gate" active for c < c_apoptosis_threshold.
    k_n = 46.0239
    k_c = 38.6537
    n0 = p.c_quiescence_threshold / 8.0

    w_in_hidden = np.array([
        [0.0, k_n],     # hN
        [-k_c, 0.0],    # hA
    ], dtype=np.float32)

    theta_hidden = np.array([
        k_n * n0,
        -k_c * p.c_apoptosis_threshold,
    ], dtype=np.float32)

    # Output pre-activations:
    # zP = -sN*hN - sA*hA + bP
    # zQ = +sN*hN - sA*hA + bQ
    # zA =          sAA*hA - bA
    sN = 3.1795
    sA = 0.1839
    bP = 2.9583
    bQ = -0.5562
    sAA = 9.3252
    bA = 1.8945

    W_hidden_out = np.array([
        [-sN, -sA],     # P
        [sN, -sA],      # Q
        [0.0, sAA],     # A
    ], dtype=np.float32)

    # Network uses O = T(W*V - phi), so phi stores output thresholds/bias shifts.
    phi_out = np.array([
        -bP,            # P
        -bQ,            # Q
        bA,             # A
    ], dtype=np.float32)

    return w_in_hidden, W_hidden_out, theta_hidden, phi_out


def policy_action_scores(
    w_in_hidden: np.ndarray,
    W_hidden_out: np.ndarray,
    theta_hidden: np.ndarray,
    phi_out: np.ndarray,
    c_val: float,
    n_neigh: int,
) -> np.ndarray:
    """Compute NN output scores (P,Q,A) given local oxygen and neighbor count.
    Params: w_in_hidden (2,2), W_hidden_out (3,2), theta_hidden (2), phi_out (3), c_val, n_neigh.
    Returns: scores (3,) in [0,1].
    """
    x = np.array([c_val, float(n_neigh) / 8.0], dtype=np.float32)
    hidden = transfer_sigmoid(w_in_hidden @ x - theta_hidden)
    out = transfer_sigmoid(W_hidden_out @ hidden - phi_out)
    return out


def sample_action(scores: np.ndarray) -> int:
    """Select the max-score action index.
    Params: scores (3,).
    Returns: action index (0=P,1=Q,2=A).
    """
    return int(np.argmax(scores))


def mutate_genotype(
    w_in_hidden: np.ndarray,
    W_hidden_out: np.ndarray,
    theta_hidden: np.ndarray,
    phi_out: np.ndarray,
    p: Params,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mutate NN genotype entries with probability p and Gaussian noise.
    Params: w_in_hidden (2,2), W_hidden_out (3,2), theta_hidden (2), phi_out (3), p, rng.
    Returns: mutated copies of all four arrays.
    """
    w_new = w_in_hidden.copy()
    W_new = W_hidden_out.copy()
    theta_new = theta_hidden.copy()
    phi_new = phi_out.copy()

    mask_w = rng.random(w_new.shape) < p.p
    w_new[mask_w] += rng.normal(0.0, p.s, size=mask_w.sum()).astype(np.float32)

    mask_W = rng.random(W_new.shape) < p.p
    W_new[mask_W] += rng.normal(0.0, p.s, size=mask_W.sum()).astype(np.float32)

    mask_theta = rng.random(theta_new.shape) < p.p
    theta_new[mask_theta] += rng.normal(0.0, p.s, size=mask_theta.sum()).astype(np.float32)

    mask_phi = rng.random(phi_new.shape) < p.p
    phi_new[mask_phi] += rng.normal(0.0, p.s, size=mask_phi.sum()).astype(np.float32)

    return w_new, W_new, theta_new, phi_new


class SimulationModel:
    """Simulation model for tumour growth in Hybrid Cellular Automata minimal model (oxygen-only)."""

    def __init__(self, p: Params):
        """Create model state and initialize tumour seed.
        Params: p.
        Returns: None.
        """
        self.p = p
        self.rng = np.random.default_rng(p.rng_seed)

        N = p.N
        self.state = np.zeros((N, N), dtype=np.uint8)         # EMPTY/ALIVE/NECROTIC
        self.activity = np.zeros((N, N), dtype=np.uint8)      # ACT_PROLIF/ACT_QUIESC (only valid if ALIVE)

        self.age_hours = np.zeros((N, N), dtype=np.float32)
        self.prolif_age_hours = np.zeros((N, N), dtype=np.float32)

        self.c = np.full((N, N), p.c0, dtype=np.float32)
        set_dirichlet_boundary(self.c, p.c0)

        # Each alive cell carries a small feed-forward response network genotype.
        # Shapes: w_in_hidden (2,2), W_hidden_out (3,2), theta_hidden (2), phi_out (3)
        self.w = np.zeros((N, N, 2, 2), dtype=np.float32)
        self.W = np.zeros((N, N, 3, 2), dtype=np.float32)
        self.theta = np.zeros((N, N, 2), dtype=np.float32)
        self.phi = np.zeros((N, N, 3), dtype=np.float32)

        w0, W0, theta0, phi0 = init_minimal_network_genotype(p)
        self._seed_tumour(w0, W0, theta0, phi0)

    def _seed_tumour(
        self,
        w0: np.ndarray,
        W0: np.ndarray,
        theta0: np.ndarray,
        phi0: np.ndarray
    ) -> None:
        """Seed a circular cluster of initial cancer cells.
        Params: w0, W0, theta0, phi0.
        Returns: None.
        """
        N = self.p.N
        cx = cy = N // 2
        rr = self.p.seed_radius
        for i in range(N):
            for j in range(N):
                if (i - cx) ** 2 + (j - cy) ** 2 <= rr ** 2:
                    self.state[i,j] = ALIVE
                    self.activity[i,j] = ACT_PROLIF
                    self.age_hours[i,j] = 0.0
                    self.prolif_age_hours[i,j] = max(
                        1e-3,
                        float(self.rng.normal(self.p.Ap, self.p.Ap_s))
                    )
                    self.w[i,j] = w0
                    self.W[i,j] = W0
                    self.theta[i,j] = theta0
                    self.phi[i,j] = phi0

    def _uptake_alpha(self) -> np.ndarray:
        """Compute local oxygen uptake coefficient alpha(x,t).
        Params: None.
        Returns: alpha (N,N).
        """
        quiescence    = (self.state == ALIVE) & (self.activity == ACT_QUIESC)
        proliferation = (self.state == ALIVE) & (self.activity == ACT_PROLIF)
        alpha = np.zeros_like(self.c, dtype=np.float32)
        alpha[proliferation] = self.p.rc
        alpha[quiescence] = self.p.rc / self.p.q
        return alpha

    def _alive_indices_shuffled(self) -> np.ndarray:
        """Return shuffled list of alive cell coordinates.
        Params: None.
        Returns: coords (M,2) int.
        """
        coords = np.argwhere(self.state == ALIVE)
        self.rng.shuffle(coords)
        return coords

    def _von_neumann_empty_neighbors(self, i: int, j: int) -> Tuple[Tuple[int, int], ...]:
        """List empty Von Neumann neighbors (4-neighborhood).
        Params: i, j.
        Returns: tuple of (ni,nj).
        """
        N = self.p.N
        neigh = []
        for di, dj in ((-1,0), (1,0), (0,-1), (0,1)):
            ni, nj = i + di, j + dj
            if (0 <= ni < N) and (0 <= nj < N) and (self.state[ni,nj] == EMPTY):
                neigh.append((ni,nj))
        return tuple(neigh)
    
    def _oxygen_demand(self) -> np.ndarray:
        """Compute per-site oxygen demand (amount requested) in the current field time-step.
        Params: None.
        Returns: demand (N,N) float32.
        """
        proliferation = (self.state == ALIVE) & (self.activity == ACT_PROLIF)
        quiescience   = (self.state == ALIVE) & (self.activity == ACT_QUIESC)
        demand = np.zeros_like(self.c, dtype=np.float32)
        demand[proliferation] = self.p.rc * self.p.Dt
        demand[quiescience]   = (self.p.rc / self.p.q) * self.p.Dt
        return demand

    def step(self) -> Dict[str, float]:
        """Advance simulation by one CA + field update.
        Params: None.
        Returns: summary dict.
        """
        # 0) Starvation necrosis: if demand > availability -> necrotic (site blocked)
        demand = self._oxygen_demand()
        nec_mask = (self.state == ALIVE) & (self.c < demand)
        self.state[nec_mask] = NECROTIC

        # 1) Update oxygen field
        alpha = self._uptake_alpha()
        self.c = oxygen_step_explicit(self.c, alpha, self.p)

        # 2) Update cells in random order
        n_neigh = moore_neighbors_count(self.state)
        coords = self._alive_indices_shuffled()

        for (i,j) in coords:
            if self.state[i,j] != ALIVE:
                continue
            
            # Get the local state in (i,j) to choose the next action
            c_ij = float(self.c[i,j])
            w_ij = self.w[i,j]
            W_ij = self.W[i,j]
            theta_ij = self.theta[i,j]
            phi_ij = self.phi[i,j]
            scores = policy_action_scores(w_ij, W_ij, theta_ij, phi_ij, c_ij, int(n_neigh[i,j]))
            action = sample_action(scores)

            # A: apoptosis frees space
            if action == 2:
                self.state[i,j] = EMPTY
                continue

            # Q: quiescent
            if action == 1:
                self.activity[i,j] = ACT_QUIESC
                self.age_hours[i,j] = 0.0
                continue

            # P: proliferate (contact inhibition if no space)
            self.activity[i,j] = ACT_PROLIF
            self.age_hours[i,j] += self.p.Dt_age_inc

            if self.age_hours[i,j] < self.prolif_age_hours[i,j]:
                continue

            empties = self._von_neumann_empty_neighbors(i, j)
            if not empties:
                self.activity[i,j] = ACT_QUIESC
                self.age_hours[i,j] = 0.0
                continue

            ni, nj = empties[self.rng.integers(0, len(empties))]
            self.state[ni,nj] = ALIVE
            self.activity[ni,nj] = ACT_PROLIF
            self.age_hours[ni,nj] = 0.0
            self.prolif_age_hours[ni,nj] = max(1e-3, float(self.rng.normal(self.p.Ap, self.p.Ap_s)))

            w_child, W_child, theta_child, phi_child = mutate_genotype(
                w_ij,
                W_ij,
                theta_ij,
                phi_ij,
                self.p,
                self.rng
            )
            self.w[ni,nj] = w_child
            self.W[ni,nj] = W_child
            self.theta[ni,nj] = theta_child
            self.phi[ni,nj] = phi_child

            self.age_hours[i,j] = 0.0
            self.prolif_age_hours[i,j] = max(
                1e-3, float(self.rng.normal(self.p.Ap, self.p.Ap_s))
            )

        return {
            "alive": int(np.sum(self.state == ALIVE)),
            "necrotic": int(np.sum(self.state == NECROTIC)),
            "empty": int(np.sum(self.state == EMPTY)),
            "c_min": float(self.c.min()),
            "c_mean": float(self.c.mean())
        }

    def run(self) -> Dict[str, np.ndarray]:
        """Run simulation for p.steps and collect basic trajectories.
        Params: None.
        Returns: dict with time series.
        """
        alive_ts = np.zeros(self.p.steps, dtype=np.int32)
        nec_ts = np.zeros(self.p.steps, dtype=np.int32)
        cmin_ts = np.zeros(self.p.steps, dtype=np.float32)

        for t in range(self.p.steps):
            out = self.step()
            alive_ts[t] = out["alive"]
            nec_ts[t] = out["necrotic"]
            cmin_ts[t] = out["c_min"]

        return {"alive": alive_ts, "necrotic": nec_ts, "c_min": cmin_ts}
