import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

# Cell states
EMPTY = np.uint8(0)
PROLIFERATING = np.uint8(1)
QUIESCENT = np.uint8(2)
NECROTIC = np.uint8(3)
DEAD = np.uint8(4)


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
    Occupied means proliferating, quiescent, or necrotic.
    Params: state (N,N) uint8.
    Returns: n_neigh (N,N) int16.
    """
    occ = (
        (state == PROLIFERATING) |
        (state == QUIESCENT) |
        (state == NECROTIC)
    ).astype(np.int16)
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


@dataclass
class ArtificialNN:
    """Per-cell ANN genotype and behaviour (2->2->3 network)."""
    w_in_hidden: np.ndarray
    W_hidden_out: np.ndarray
    theta_hidden: np.ndarray
    phi_out: np.ndarray

    @classmethod
    def from_params(cls, p: Params) -> "ArtificialNN":
        """Build a minimal-model ANN instance from default paper-aligned parameters."""
        w, W, theta, phi = init_minimal_network_genotype(p)
        return cls(w, W, theta, phi)

    def copy(self) -> "ArtificialNN":
        """Deep-copy ANN parameters."""
        return ArtificialNN(
            self.w_in_hidden.copy(),
            self.W_hidden_out.copy(),
            self.theta_hidden.copy(),
            self.phi_out.copy(),
        )

    def forward(self, c_val: float, n_neigh: int) -> np.ndarray:
        """Compute action scores [P,Q,A] in [0,1] from local oxygen and crowding."""
        x = np.array([c_val, float(n_neigh)], dtype=np.float32)
        hidden = transfer_sigmoid(self.w_in_hidden @ x - self.theta_hidden)
        out = transfer_sigmoid(self.W_hidden_out @ hidden - self.phi_out)
        return out

    def mutate_inplace(self, p: Params, rng: np.random.Generator) -> None:
        """Apply paper-consistent mutations directly to this ANN genotype."""
        sizes = np.array(
            [
                self.w_in_hidden.size,
                self.W_hidden_out.size,
                self.theta_hidden.size,
                self.phi_out.size,
            ],
            dtype=np.int32,
        )
        total_entries = int(sizes.sum())
        if total_entries == 0:
            return

        # p is interpreted as expected fraction of entries mutated per division.
        n_mut = int(rng.poisson(p.p * total_entries))
        if n_mut <= 0:
            return

        n_mut = min(n_mut, total_entries)
        mut_idx = rng.choice(total_entries, size=n_mut, replace=False)
        noise = rng.normal(0.0, p.s, size=n_mut).astype(np.float32)

        flat = np.concatenate(
            [
                self.w_in_hidden.ravel(),
                self.W_hidden_out.ravel(),
                self.theta_hidden.ravel(),
                self.phi_out.ravel(),
            ]
        ).astype(np.float32, copy=False)
        flat[mut_idx] += noise

        c0 = sizes[0]
        c1 = c0 + sizes[1]
        c2 = c1 + sizes[2]
        c3 = c2 + sizes[3]
        self.w_in_hidden = flat[:c0].reshape(self.w_in_hidden.shape)
        self.W_hidden_out = flat[c0:c1].reshape(self.W_hidden_out.shape)
        self.theta_hidden = flat[c1:c2].reshape(self.theta_hidden.shape)
        self.phi_out = flat[c2:c3].reshape(self.phi_out.shape)

    def mutated_copy(self, p: Params, rng: np.random.Generator) -> "ArtificialNN":
        """Return a mutated copy of this ANN genotype."""
        child = self.copy()
        child.mutate_inplace(p, rng)
        return child


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
        self.state = np.zeros((N, N), dtype=np.uint8)         # EMPTY/PROLIFERATING/QUIESCENT/NECROTIC/DEAD

        self.age_hours = np.zeros((N, N), dtype=np.float32)
        self.prolif_age_hours = np.zeros((N, N), dtype=np.float32)

        self.c = np.full((N, N), p.c0, dtype=np.float32)
        set_dirichlet_boundary(self.c, p.c0)

        # Each living cell can carry one ANN genotype instance (None for non-living cells).
        self.ann = np.empty((N, N), dtype=object)
        self.ann.fill(None)

        ann_proto = ArtificialNN.from_params(p)
        self._seed_tumour(ann_proto)

    def _seed_tumour(
        self,
        ann_proto: ArtificialNN,
    ) -> None:
        """Seed a circular cluster of initial cancer cells.
        Params: ann_proto.
        Returns: None.
        """
        N = self.p.N
        cx = cy = N // 2
        rr = self.p.seed_radius
        for i in range(N):
            for j in range(N):
                if (i - cx) ** 2 + (j - cy) ** 2 <= rr ** 2:
                    self.state[i,j] = PROLIFERATING
                    self.age_hours[i,j] = 0.0
                    self.prolif_age_hours[i,j] = max(
                        1e-3,
                        float(self.rng.normal(self.p.Ap, self.p.Ap_s))
                    )
                    self.ann[i,j] = ann_proto.copy()

    def _uptake_alpha(self, action_map: np.ndarray, F_map: np.ndarray) -> np.ndarray:
        """Compute local oxygen uptake coefficient alpha(x,t) from chosen actions and modulation.
        Params: action_map (N,N) int8, F_map (N,N) float32.
        Returns: alpha (N,N).
        """
        living = (self.state == PROLIFERATING) | (self.state == QUIESCENT)
        quiescence = living & (action_map == 1)
        proliferation = living & (action_map == 0)
        alpha = np.zeros_like(self.c, dtype=np.float32)
        alpha[proliferation] = self.p.rc * F_map[proliferation]
        alpha[quiescence] = (self.p.rc / self.p.q) * F_map[quiescence]
        return alpha

    def _alive_indices_shuffled(self) -> np.ndarray:
        """Return shuffled list of alive cell coordinates.
        Params: None.
        Returns: coords (M,2) int.
        """
        coords = np.argwhere((self.state == PROLIFERATING) | (self.state == QUIESCENT))
        self.rng.shuffle(coords)
        return coords

    def _von_neumann_empty_neighbors(self, i: int, j: int) -> Tuple[Tuple[int, int], ...]:
        """List available Von Neumann neighbors (4-neighborhood).
        Available means EMPTY or DEAD (apoptotic debris cleared on recolonization).
        Params: i, j.
        Returns: tuple of (ni,nj).
        """
        N = self.p.N
        neigh = []
        for di, dj in ((-1,0), (1,0), (0,-1), (0,1)):
            ni, nj = i + di, j + dj
            if (
                (0 <= ni < N)
                and (0 <= nj < N)
                and (self.state[ni,nj] == EMPTY or self.state[ni,nj] == DEAD)
            ):
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
                if (
                    self.state[ni, nj] == PROLIFERATING
                    or self.state[ni, nj] == QUIESCENT
                    or self.state[ni, nj] == NECROTIC
                ):
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

        # Iterates over each living (PROLIFERATING/QUIESCENT) and shuffled cells coordinates
        # Steps per iteration: scores computation -> consumption check (necrosis) -> death/action -> division.
        coords = self._alive_indices_shuffled()
        for (i,j) in coords:
            if self.state[i,j] != PROLIFERATING and self.state[i,j] != QUIESCENT:
                continue

            # Defensive guard: alive cells should always have an ANN
            ann_ij = self.ann[i,j]
            if ann_ij is None:
                continue

            # Compute the number of neighbors (non-empty cells in Moore neighborhood)
            n_ij = self._moore_neighbor_count_at(i,j)

            # Compute the forward pass to get actions' scores (to choose the action with highest score)
            scores = ann_ij.forward(float(self.c[i,j]), n_ij)
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
                self.ann[i,j] = None
                action_map[i,j] = np.int8(-1)
                F_map[i,j] = np.float32(0.0)
                continue

            # Apoptosis case (action = A)
            if action == 2:
                self.state[i,j] = DEAD
                self.ann[i,j] = None
                action_map[i,j] = np.int8(-1)
                F_map[i,j] = np.float32(0.0)
                continue

            # Quiescence case (action = Q)
            if action == 1:
                self.state[i,j] = QUIESCENT
                continue

            # Proliferation case (action = P)
            self.state[i,j] = PROLIFERATING

            # Proliferation-age increment
            self.age_hours[i,j] += self.p.Dt_age_inc * F_ij
            if self.age_hours[i,j] < self.prolif_age_hours[i,j]:
                continue
            
            # Division phase - if no empty Von Neumann neighbors, cell becomes Quiescent instead of Proliferating
            empties = self._von_neumann_empty_neighbors(i, j)
            if not empties:
                self.state[i,j] = QUIESCENT
                self.age_hours[i,j] = 0.0
                continue

            # Choose randomly an empty neighbor to place the daughter cell
            ni, nj = empties[self.rng.integers(0, len(empties))]
            self.state[ni,nj] = PROLIFERATING
            self.ann[ni,nj] = ann_ij.mutated_copy(self.p, self.rng)
            self.age_hours[ni,nj] = 0.0
            self.prolif_age_hours[ni,nj] = max(1e-3, float(self.rng.normal(self.p.Ap, self.p.Ap_s)))

            self.age_hours[i,j] = 0.0
            self.prolif_age_hours[i,j] = max(1e-3, float(self.rng.normal(self.p.Ap, self.p.Ap_s)))

        # Oxygen PDE update from effective actions taken in this CA step.
        alpha = self._uptake_alpha(action_map, F_map)
        self.c = oxygen_step_explicit(self.c, alpha, self.p)
        
        return {
            "proliferating": int(np.sum(self.state == PROLIFERATING)),
            "quiescent": int(np.sum(self.state == QUIESCENT)),
            "necrotic": int(np.sum(self.state == NECROTIC)),
            "dead": int(np.sum(self.state == DEAD)),
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
        dead_ts = np.zeros(self.p.steps, dtype=np.int32)
        cmin_ts = np.zeros(self.p.steps, dtype=np.float32)

        for t in range(self.p.steps):
            out = self.step()
            alive_ts[t] = out["alive"]
            nec_ts[t] = out["necrotic"]
            dead_ts[t] = out["dead"]
            cmin_ts[t] = out["c_min"]

        return {"alive": alive_ts, "necrotic": nec_ts, "dead": dead_ts, "c_min": cmin_ts}
