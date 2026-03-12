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
    '''
    Store simulation parameters for the oxygen-only Hybrid Cellular Automata model.
    '''
    # Lattice
    N: int = 400                            # Project grid side length; in the paper this is the NxN CA/PDE computational domain.
    seed_radius: int = 4                    # Project-only initialization radius (in lattice sites) for the central seeded tumour.

    # Runtime
    steps: int = 200                        # Project-only number of global CA/PDE iterations to run.
    rng_seed: int = 0                       # Project-only random seed used for reproducible stochastic events.

    # Oxygen fields
    c0: float = 1.0                         # Paper c_0: fixed oxygen value at the domain boundary (main environmental control parameter).
    Dc: float = 1.04                        # Paper D_c: oxygen diffusion coefficient in the reaction-diffusion equation.
    Dt: float = 5e-4                        # Paper Delta t: simulation time step used in the explicit oxygen update.
    d: float = 0.05                         # Paper d: lattice spacing used by the finite-difference Laplacian.

    # Uptake
    rc: float = 124.7                       # Paper r_c: baseline oxygen consumption rate.
    q: float = 5.0                          # Project simplification: quiescent-cell uptake scaling factor (uptake = r_c / q).
    Tr: float = 0.675                       # Paper T_r: response target/threshold in metabolic modulation F(R).
    k: float = 6.0                          # Paper k: slope (gain) of metabolic modulation F(R).

    # Life-cycle thresholds
    c_apoptosis_threshold: float = 0.15     # Project threshold for the hypoxia gate that drives apoptosis preference.
    c_quiescence_threshold: float = 3       # Project crowding threshold (on Moore-neighbour count) used by the quiescence gate.

    # Cell cycle (hours)
    Dt_age_inc: float = 16.0                # Project conversion from one CA update to biological hours for age accumulation.
    Ap: float = 16.0                        # Paper A_p: mean proliferation age (cell-cycle duration) required before division.
    Ap_s: float = 8.0                       # Project heterogeneity term: std dev used to sample per-cell proliferation age around A_p.

    # Mutations
    p: float = 0.01                         # Paper p: mutation probability/rate of division (implemented as expected mutated fraction).
    s: float = 0.25                         # Paper sigma: std of Gaussian perturbation applied to mutated ANN entries.


def moore_neighbors_count(state: np.ndarray) -> np.ndarray:
    '''
    Compute occupied-neighbor counts in Moore neighborhood for all lattice sites.

    Params
    -------
    - state (np.ndarray) : Cellular automaton state map with shape `(N, N)`.

    Returns
    --------
    - n_neigh (np.ndarray) : Occupied-neighbor counts in `[0, 8]` for each lattice site.
    '''
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
    '''
    Compute the 5-point Laplacian on interior lattice points.

    Params
    -------
    - u (np.ndarray) : Scalar field with shape `(N, N)`.
    - dx (float) : Lattice spacing.

    Returns
    --------
    - lap (np.ndarray) : Laplacian field with zeroed boundaries.
    '''
    lap = np.zeros_like(u, dtype=np.float32)
    lap[1:-1, 1:-1] = (
        u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4.0 * u[1:-1, 1:-1]
    ) / (dx * dx)
    return lap


def set_dirichlet_boundary(u: np.ndarray, value: float) -> None:
    '''
    Apply constant Dirichlet boundary conditions to all field edges.

    Params
    -------
    - u (np.ndarray) : Scalar field with shape `(N, N)` to modify in place.
    - value (float) : Constant boundary value.
    '''
    u[0, :] = value
    u[-1, :] = value
    u[:, 0] = value
    u[:, -1] = value


def oxygen_step_explicit(c: np.ndarray, uptake: np.ndarray, p: Params) -> np.ndarray:
    '''
    Execute one explicit Euler update of oxygen reaction-diffusion dynamics.

    Params
    -------
    - c (np.ndarray) : Current oxygen field with shape `(N, N)`.
    - uptake (np.ndarray) : Oxygen consumption field with shape `(N, N)`.
    - p (Params) : Model parameters.

    Returns
    --------
    - c_new (np.ndarray) : Updated oxygen field after one explicit step.
    '''
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
    '''
    Apply the ANN transfer function T(x)=1/(1+exp(-2x)).

    Params
    -------
    - x (np.ndarray) : Pre-activation values.

    Returns
    --------
    - y (np.ndarray) : Transformed values in `[0, 1]`.
    '''
    return 1.0 / (1.0 + np.exp(-2.0 * x))


def F(R: float, p: Params) -> float:
    '''
    Compute metabolic modulation factor from selected ANN response strength.

    Params
    -------
    - R (float) : Selected action response in `[0, 1]`.
    - p (Params) : Model parameters.

    Returns
    --------
    - modulation (float) : Metabolic modulation factor with lower bound `0.25`.
    '''
    return float(max(p.k * (R - p.Tr) + 1.0, 0.25))



@dataclass
class ArtificialNN:
    '''
    Represent per-cell ANN genotype and behavior in the minimal model.
    '''
    w_in_hidden: np.ndarray
    W_hidden_out: np.ndarray
    theta_hidden: np.ndarray
    phi_out: np.ndarray

    def __init__(self, p: Params, ann_params_dict: dict = None):
        '''
        Initialize a baseline ANN genotype from model parameters.

        Params
        -------
        - p (Params) : Model parameters.
        - ann_params_dict (dict): parameters used to initialize the weights and biases of the ANN.
        '''

        # Input vector x = [c, n], with n in [0,8] (Moore neighbors).
        # Hidden nodes:
        # hN: "crowding gate" active for n > 3.
        # hA: "hypoxia gate" active for c < c_apoptosis_threshold.
        # Output pre-activations:
        # P = -sN*hN - sA*hA + bP
        # Q = +sN*hN - sA*hA + bQ
        # A =          sAA*hA - bA
        k_n = float(ann_params_dict["k_n"])
        k_c = float(ann_params_dict["k_c"])
        sN = float(ann_params_dict["sN"])
        sA = float(ann_params_dict["sA"])
        sAA = float(ann_params_dict["sAA"])
        bP = float(ann_params_dict["bP"])
        bQ = float(ann_params_dict["bQ"])
        bA = float(ann_params_dict["bA"])

        # Project crowding pivot for hN; derived from neighbour threshold used in this implementation
        n0 = float(p.c_quiescence_threshold)

        # Input-to-hidden weight matrix (w) for the minimal response network.
        self.w_in_hidden = np.array([
            [0.0, k_n],     # hN
            [-k_c, 0.0],    # hA
        ], dtype=np.float32)

        # Hidden thresholds (theta) that set activation pivots for hN and hA.
        self.theta_hidden = np.array([
            k_n * n0,
            -k_c * p.c_apoptosis_threshold,
        ], dtype=np.float32)

        # Hidden-to-output weight matrix (W) for the three life-cycle responses [P,Q,A].
        self.W_hidden_out = np.array([
            [-sN, -sA],     # P
            [sN, -sA],      # Q
            [0.0, sAA],     # A
        ], dtype=np.float32)

        # Output thresholds (phi), encoded as bias shifts in O = T(WV - phi).
        self.phi_out = np.array([
            -bP,            # P
            -bQ,            # Q
            bA,             # A
        ], dtype=np.float32)

        return
    


    def copy(self) -> "ArtificialNN":
        '''
        Create a deep copy of ANN genotype arrays.

        Returns
        --------
        - ann_copy (ArtificialNN) : Deep copy of the current ANN genotype.
        '''
        ann_copy = ArtificialNN.__new__(ArtificialNN)
        ann_copy.w_in_hidden = self.w_in_hidden.copy()
        ann_copy.W_hidden_out = self.W_hidden_out.copy()
        ann_copy.theta_hidden = self.theta_hidden.copy()
        ann_copy.phi_out = self.phi_out.copy()
        return ann_copy

    def forward(self, c_val: float, n_neigh: int) -> np.ndarray:
        '''
        Compute ANN action scores for one cellular microenvironment.

        Params
        -------
        - c_val (float) : Local oxygen concentration.
        - n_neigh (int) : Local occupied-neighbor count.

        Returns
        --------
        - out (np.ndarray) : Action scores for `[P, Q, A]` in `[0, 1]`.
        '''
        x = np.array([c_val, float(n_neigh)], dtype=np.float32)
        x = transfer_sigmoid(self.w_in_hidden @ x - self.theta_hidden)
        x = transfer_sigmoid(self.W_hidden_out @ x - self.phi_out)
        return x

    def mutate_inplace(self, p: Params, rng: np.random.Generator) -> None:
        '''
        Apply Poisson-Gaussian mutations to ANN genotype in place.

        Params
        -------
        - p (Params) : Model parameters controlling mutation rate and variance.
        - rng (np.random.Generator) : Random generator used for mutation sampling.
        '''
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
        '''
        Build a mutated ANN copy for daughter-cell inheritance.

        Params
        -------
        - p (Params) : Model parameters controlling mutation rate and variance.
        - rng (np.random.Generator) : Random generator used for mutation sampling.

        Returns
        --------
        - child (ArtificialNN) : Mutated ANN clone.
        '''
        child = self.copy()
        child.mutate_inplace(p, rng)
        return child


class SimulationModel:
    '''
    Simulate tumour growth with coupled cellular automaton and oxygen PDE dynamics.
    '''

    def __init__(self, p: Params, ann_params_dict: dict = None):
        '''
        Create simulation state and initialize the tumour seed.

        Params
        -------
        - p (Params) : Model parameters.
        - ann_params_dict (dict) : Optional ANN initialization parameters passed to `ArtificialNN`.
        '''
        self.p = p
        self.rng = np.random.default_rng(p.rng_seed)

        N = p.N
        self.state = np.zeros((N, N), dtype=np.uint8)

        self.age_hours = np.zeros((N, N), dtype=np.float32)
        self.prolif_age_hours = np.zeros((N, N), dtype=np.float32)

        self.c = np.full((N, N), p.c0, dtype=np.float32)
        set_dirichlet_boundary(self.c, p.c0)

        # Each living cell can carry one ANN genotype instance (None for non-living cells).
        self.ann = np.empty((N, N), dtype=object)
        self.ann.fill(None)

        ann = ArtificialNN(p, ann_params_dict)
        self._seed_tumour(ann)

    def _seed_tumour(
        self,
        ann: ArtificialNN,
    ) -> None:
        '''
        Seed a circular initial tumour at lattice center.

        Params
        -------
        - ann (ArtificialNN) : Baseline ANN genotype copied to seeded cells.
        '''
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
                    self.ann[i,j] = ann.copy()

    def _uptake_alpha(self, action_map: np.ndarray, F_map: np.ndarray) -> np.ndarray:
        '''
        Compute oxygen uptake field from selected cellular actions.

        Params
        -------
        - action_map (np.ndarray) : Selected action labels with shape `(N, N)`.
        - F_map (np.ndarray) : Metabolic modulation factors with shape `(N, N)`.

        Returns
        --------
        - alpha (np.ndarray) : Oxygen uptake coefficients for the PDE update.
        '''
        living = (self.state == PROLIFERATING) | (self.state == QUIESCENT)
        quiescence = living & (action_map == 1)
        proliferation = living & (action_map == 0)
        alpha = np.zeros_like(self.c, dtype=np.float32)
        alpha[proliferation] = self.p.rc * F_map[proliferation]
        alpha[quiescence] = (self.p.rc / self.p.q) * F_map[quiescence]
        return alpha

    def _alive_indices_shuffled(self) -> np.ndarray:
        '''
        Return shuffled coordinates of living cells.

        Params
        -------
        - None (None) : This method does not require additional parameters.

        Returns
        --------
        - coords (np.ndarray) : Shuffled living-cell coordinates with shape `(M, 2)`.
        '''
        coords = np.argwhere((self.state == PROLIFERATING) | (self.state == QUIESCENT))
        self.rng.shuffle(coords)
        return coords

    def _von_neumann_empty_neighbors(self, i: int, j: int) -> Tuple[Tuple[int, int], ...]:
        '''
        List available Von Neumann neighbors around one site.

        Params
        -------
        - i (int) : Row index of the reference cell.
        - j (int) : Column index of the reference cell.

        Returns
        --------
        - neighbors (Tuple[Tuple[int, int], ...]) : Available neighbor coordinates.
        '''
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
        '''
        Count occupied Moore neighbors around one lattice site.

        Params
        -------
        - i (int) : Row index of the reference cell.
        - j (int) : Column index of the reference cell.

        Returns
        --------
        - count (int) : Occupied-neighbor count in `[0, 8]`.
        '''
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
        '''
        Advance simulation step of alive cells (PROLIFERATING and QUIESCENT) in random order,
        following this sequence: score computation, consumption check (necrosis), death/action,
        and division.

        Returns
        --------
        - summary_dict (Dict[str, float]) : Summary statistics of the current state with keys
          `"proliferating"`, `"quiescent"`, `"necrotic"`, `"dead"`, `"empty"`, `"c_min"`, `"c_mean"`.
        '''
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

    def run(self, track_diversity: bool = False) -> Dict[str, np.ndarray]:
        '''
        Run simulation for all configured steps and collect trajectories.

        Params
        -------
        - track_diversity (bool) : If True, compute Shannon diversity over living-cell genotypes.

        Returns
        --------
        - trajectories (Dict[str, np.ndarray]) : Time-series arrays for state counts and
          diagnostics, including `"invasive"` and `"shannon"`.
        '''
        alive_ts = np.zeros(self.p.steps, dtype=np.int32)
        prolif_ts = np.zeros(self.p.steps, dtype=np.int32)
        quiesc_ts = np.zeros(self.p.steps, dtype=np.int32)
        nec_ts = np.zeros(self.p.steps, dtype=np.int32)
        dead_ts = np.zeros(self.p.steps, dtype=np.int32)
        total_ts = np.zeros(self.p.steps, dtype=np.int32)
        empty_ts = np.zeros(self.p.steps, dtype=np.int32)
        cmean_ts = np.zeros(self.p.steps, dtype=np.float32)
        cmin_ts = np.zeros(self.p.steps, dtype=np.float32)
        invasive_ts = np.zeros(self.p.steps, dtype=np.float32)
        shannon_ts = np.full(self.p.steps, np.nan, dtype=np.float32)

        cx = cy = self.p.N // 2

        def _genotype_key(ann: ArtificialNN) -> Tuple[bytes, bytes, bytes, bytes]:
            return (
                ann.w_in_hidden.tobytes(),
                ann.W_hidden_out.tobytes(),
                ann.theta_hidden.tobytes(),
                ann.phi_out.tobytes(),
            )

        for t in range(self.p.steps):
            out = self.step()
            alive_ts[t] = out["proliferating"] + out["quiescent"]
            prolif_ts[t] = out["proliferating"]
            quiesc_ts[t] = out["quiescent"]
            nec_ts[t] = out["necrotic"]
            dead_ts[t] = out["dead"]
            empty_ts[t] = out["empty"]
            cmean_ts[t] = out["c_mean"]
            cmin_ts[t] = out["c_min"]

            occupied = (
                (self.state == PROLIFERATING)
                | (self.state == QUIESCENT)
                | (self.state == NECROTIC)
            )
            if alive_ts[t] > 0 or nec_ts[t] > 0:
                ii, jj = np.where(occupied)
                invasive_ts[t] = float(np.sqrt((ii - cx) ** 2 + (jj - cy) ** 2).max())

            # Case of computing the diversity level (Shannon Index) in the lattice
            if track_diversity:
                alive_coords = np.argwhere((self.state == PROLIFERATING) | (self.state == QUIESCENT))
                # Case of no existing alive cells
                if alive_coords.size == 0:
                    shannon_ts[t] = 0.0
                # Case of existing alive cells (are counted the number of cells for each different genotype)
                else:
                    counts = {}
                    for i, j in alive_coords:
                        ann_ij = self.ann[i,j]
                        if ann_ij is None:
                            continue
                        key = _genotype_key(ann_ij)
                        counts[key] = counts.get(key, 0) + 1
                    if not counts:
                        shannon_ts[t] = 0.0
                    else:
                        freq = np.array(list(counts.values()), dtype=np.float64)
                        probs = freq / freq.sum()
                        shannon_ts[t] = float(-np.sum(probs * np.log(probs + 1e-12)))  # Shannon Index (H) formula

        return {
            "alive": alive_ts,
            "proliferating": prolif_ts,
            "quiescent": quiesc_ts,
            "necrotic": nec_ts,
            "dead": dead_ts,
            "empty": empty_ts,
            "c_mean": cmean_ts,
            "c_min": cmin_ts,
            "invasive": invasive_ts,
            "shannon": shannon_ts,
        }
