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
    Tr: float = 0.675
    k: float = 6.0

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
    """Compute Moore-neighborhood occupied-neighbor count for each site.
    Occupied means alive or necrotic.
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
    """5-point Laplacian on interior points with non-periodic boundaries.
    Params: u (N,N), dx.
    Returns: lap (N,N).
    """
    lap = np.zeros_like(u, dtype=np.float32)
    lap[1:-1, 1:-1] = (
        u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4.0 * u[1:-1, 1:-1]
    ) / (dx * dx)
    return lap


def set_dirichlet_boundary(u: np.ndarray, value: float) -> None:
    """Set Dirichlet boundary on all edges in-place.
    Params: u (N,N), value.
    Returns: None.
    """
    u[0, :] = value
    u[-1, :] = value
    u[:, 0] = value
    u[:, -1] = value


def oxygen_step_explicit(c: np.ndarray, uptake: np.ndarray, p: Params) -> np.ndarray:
    """One explicit Euler step for oxygen reaction-diffusion.
    PDE form (minimal paper model): dc/dt = Dc * Lap(c) - f_c.
    Params: c (N,N), uptake=f_c (N,N), p.
    Returns: c_new (N,N).
    """
    c_work = c.copy()
    set_dirichlet_boundary(c_work, p.c0)
    lap = laplacian_5pt(c_work, p.d)

    c_new = c_work.copy()
    c_new[1:-1, 1:-1] = c_work[1:-1, 1:-1] + p.Dt * (
        p.Dc * lap[1:-1, 1:-1] - uptake[1:-1, 1:-1]
    )
    c_new[1:-1, 1:-1] = np.maximum(c_new[1:-1, 1:-1], 0.0)
    set_dirichlet_boundary(c_new, p.c0)
    return c_new


def transfer_sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid transfer function from the paper: T(x) = 1 / (1 + exp(-2x)).
    Params: x array.
    Returns: transformed array.
    """
    return 1.0 / (1.0 + np.exp(-2.0 * x))


def F(R: float, p: Params) -> float:
    """Metabolic modulation factor from the paper: F = max(k*(R-Tr)+1, 0.25).
    Params: R response strength in [0,1], p.
    Returns: F scalar >= 0.25.
    """
    return float(max(p.k * (R - p.Tr) + 1.0, 0.25))


def init_minimal_network_genotype(p: Params) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize a minimal 2->2->3 response network (inputs: oxygen, crowding; outputs: P/Q/A).
    Params: p.
    Returns: w_in_hidden (2,2), W_hidden_out (3,2), theta_hidden (2,), phi_out (3,).
    """
    # Input vector x = [c, n], with n in [0,8] (Moore neighbors).
    # Hidden nodes:
    # hN: "crowding gate" active for n > 3.
    # hA: "hypoxia gate" active for c < c_apoptosis_threshold.
    k_n = 46.0239 / 8.0
    k_c = 38.6537
    n0 = float(p.c_quiescence_threshold)

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
    x = np.array([c_val, float(n_neigh)], dtype=np.float32)
    hidden = transfer_sigmoid(w_in_hidden @ x - theta_hidden)
    out = transfer_sigmoid(W_hidden_out @ hidden - phi_out)
    return out # [P_score, Q_score, A_score]


def mutate_genotype(
    w_in_hidden: np.ndarray,
    W_hidden_out: np.ndarray,
    theta_hidden: np.ndarray,
    phi_out: np.ndarray,
    p: Params,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mutate NN genotype with Poisson mutation count and Gaussian perturbations.
    Params: w_in_hidden (2,2), W_hidden_out (3,2), theta_hidden (2), phi_out (3), p, rng.
    Returns: mutated copies of all four arrays.
    """
    w_new = w_in_hidden.copy()
    W_new = W_hidden_out.copy()
    theta_new = theta_hidden.copy()
    phi_new = phi_out.copy()

    # Paper-consistent mutation logic:
    # 1) Draw total number of mutation events from Poisson.
    # 2) Distribute events uniformly over all genotype entries.
    # 3) Add Gaussian N(0, s) perturbation to selected entries.
    sizes = np.array([w_new.size, W_new.size, theta_new.size, phi_new.size], dtype=np.int32)
    total_entries = int(sizes.sum())
    if total_entries == 0:
        return w_new, W_new, theta_new, phi_new

    # p is interpreted as expected fraction of entries mutated per division.
    n_mut = int(rng.poisson(p.p * total_entries))
    if n_mut <= 0:
        return w_new, W_new, theta_new, phi_new

    n_mut = min(n_mut, total_entries)
    mut_idx = rng.choice(total_entries, size=n_mut, replace=False)
    noise = rng.normal(0.0, p.s, size=n_mut).astype(np.float32)

    flat = np.concatenate(
        [w_new.ravel(), W_new.ravel(), theta_new.ravel(), phi_new.ravel()]
    ).astype(np.float32, copy=False)
    flat[mut_idx] += noise

    c0 = sizes[0]
    c1 = c0 + sizes[1]
    c2 = c1 + sizes[2]
    c3 = c2 + sizes[3]
    w_new = flat[:c0].reshape(w_new.shape)
    W_new = flat[c0:c1].reshape(W_new.shape)
    theta_new = flat[c1:c2].reshape(theta_new.shape)
    phi_new = flat[c2:c3].reshape(phi_new.shape)

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

    def _uptake_alpha(self, action_map: np.ndarray, F_map: np.ndarray) -> np.ndarray:
        """Compute local oxygen uptake coefficient alpha(x,t) from chosen actions and modulation.
        Params: action_map (N,N) int8, F_map (N,N) float32.
        Returns: alpha (N,N).
        """
        alive = (self.state == ALIVE)
        quiescence = alive & (action_map == 1)
        proliferation = alive & (action_map == 0)
        alpha = np.zeros_like(self.c, dtype=np.float32)
        alpha[proliferation] = self.p.rc * F_map[proliferation]
        alpha[quiescence] = (self.p.rc / self.p.q) * F_map[quiescence]
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

    def _moore_neighbor_count_at(self, i: int, j: int) -> int:
        """Count occupied Moore neighbors, which are all the positions around the cell (i,j).
        Params: i, j.
        Returns: integer in [0,8].
        """
        N = self.p.N
        c = 0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni = (i + di) % N
                nj = (j + dj) % N
                if self.state[ni, nj] != EMPTY:
                    c += 1
        return c

    def step(self) -> Dict[str, float]:
        """Advance simulation by one CA + field update.
        Params: None.
        Returns: summary dict.
        """
        N = self.p.N
        # Lattice with the chosen action per cell
        action_map = np.full((N, N), -1, dtype=np.int8)
        # Lattice with the returned metabolic modulation factor value per cell
        F_map = np.zeros((N, N), dtype=np.float32)

        # Iterates over each ALIVE and shuffled cells coordinates
        # Steps per iteration: scores computation -> consumption check (necrosis) -> death/action -> division.
        coords = self._alive_indices_shuffled()
        for (i,j) in coords:
            if self.state[i,j] != ALIVE:
                continue

            # Compute the number of neighbors (non-empty cells in Moore neighborhood)
            n_ij = self._moore_neighbor_count_at(i,j)

            # Compute the forward pass to get actions' scores (to choose the action with highest score)
            scores = policy_action_scores(
                self.w[i,j],
                self.W[i,j],
                self.theta[i,j],
                self.phi[i,j],
                float(self.c[i,j]),
                n_ij
            )
            action = int(np.argmax(scores))

            # Apply the metabolic modulation factor F to the chosen action's score
            F_ij = F(float(scores[action]), self.p)
            action_map[i,j] = np.int8(action)
            F_map[i,j] = np.float32(F_ij)

            # Demand of oxygen depends on the action taken
            demand_ij = 0.0
            if action == 0: # action = P
                demand_ij = self.p.rc * F_ij * self.p.Dt
            elif action == 1: # action = Q
                demand_ij = (self.p.rc / self.p.q) * F_ij * self.p.Dt

            # Necrotic case (oxygen available < demand)
            if action != 2 and float(self.c[i,j]) < demand_ij:
                self.state[i,j] = NECROTIC
                action_map[i,j] = np.int8(-1)
                F_map[i,j] = np.float32(0.0)
                continue

            # Apoptosis case (action = A)
            if action == 2:
                self.state[i,j] = EMPTY
                action_map[i,j] = np.int8(-1)
                F_map[i,j] = np.float32(0.0)
                continue

            # Quiescence case (action = Q)
            if action == 1:
                self.activity[i,j] = ACT_QUIESC
                continue

            # Proliferation case (action = P)
            self.activity[i,j] = ACT_PROLIF

            # Proliferation-age increment
            self.age_hours[i,j] += self.p.Dt_age_inc * F_ij
            if self.age_hours[i,j] < self.prolif_age_hours[i,j]:
                continue
            
            # Division phase - if no empty Von Neumann neighbors, cell becomes Quiescent instead of Proliferating
            empties = self._von_neumann_empty_neighbors(i, j)
            if not empties:
                self.activity[i,j] = ACT_QUIESC
                self.age_hours[i,j] = 0.0
                continue

            # Choose randomly an empty neighbor to place the daughter cell
            ni, nj = empties[self.rng.integers(0, len(empties))]
            self.state[ni,nj] = ALIVE
            self.activity[ni,nj] = ACT_PROLIF
            self.age_hours[ni,nj] = 0.0
            self.prolif_age_hours[ni,nj] = max(1e-3, float(self.rng.normal(self.p.Ap, self.p.Ap_s)))

            # Mutation of child's genotype (with probability p)
            w_ij = self.w[i,j]
            W_ij = self.W[i,j]
            theta_ij = self.theta[i,j]
            phi_ij = self.phi[i,j]
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
            self.prolif_age_hours[i,j] = max(1e-3, float(self.rng.normal(self.p.Ap, self.p.Ap_s)))

        # Oxygen PDE update from effective actions taken in this CA step.
        alpha = self._uptake_alpha(action_map, F_map)
        self.c = oxygen_step_explicit(self.c, alpha, self.p)

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
