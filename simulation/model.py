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
    Dc: float = 0.05                      # Paper D_c: oxygen diffusion coefficient in the reaction-diffusion equation.
    Dt: float = 0.2                        # Paper Delta t: simulation time step used in the explicit oxygen update.
    d: float = 1.0                       # Paper d: lattice spacing used by the finite-difference Laplacian.

    # Uptake
    rc: float = 0.4                       # Paper r_c: baseline oxygen consumption rate.
    q: float = 5.0                          # Project simplification: quiescent-cell uptake scaling factor (uptake = r_c / q).
    Tr: float = 0.675                       # Paper T_r: response target/threshold in metabolic modulation F(R).
    k: float = 6.0                          # Paper k: slope (gain) of metabolic modulation F(R).

    # Life-cycle thresholds
    c_apoptosis_threshold: float = 0.15     # Project threshold for the hypoxia gate that drives apoptosis preference.
    c_quiescence_threshold: float = 3.0     # Project crowding threshold (on Moore-neighbour count) used by the quiescence gate.

    # Cell cycle (hours)
    Dt_age_inc: float = 1.0                # Project conversion from one CA update to biological hours for age accumulation.
    Ap: float = 1.0                        # Paper A_p: mean proliferation age (cell-cycle duration) required before division.
    Ap_s: float = 0.5                       # Project heterogeneity term: std dev used to sample per-cell proliferation age around A_p.

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
    output_raw_order: Tuple[str, str, str]

    def __init__(self, ann_params_dict: dict = None):
        '''
        Initialize a baseline ANN genotype from matrix-form ANN parameters.

        Params
        -------
        - ann_params_dict (dict): ANN matrix parameters with keys `w`, `W`, `theta`, `phi`.
        '''
        self._init_from_matrix_params(ann_params_dict)

    def _default_ann_params(self) -> Dict[str, np.ndarray]:
        '''
        Provide default ANN matrices for the oxygen-only reduced ANN.

        Architecture follows Appendix A equations with reduced dimensionality:
        `epsilon = [n_neighbors, oxygen]`, `|hidden| = 3`, `|output| = 3`.
        Raw output order is `[A, P, Q]`.

        Returns
        --------
        - defaults (Dict[str, np.ndarray]) : Default ANN parameter tensors.
        '''
        return {
            "w": np.array(
                [
                    [1.0, 0.0],
                    [0.5, 0.0],
                    [0.0, -2.0],
                ],
                dtype=np.float32,
            ),
            "W": np.array(
                [
                    [0.0, 0.0, 2.0],      # A
                    [-0.5, 1.0, -0.5],    # P
                    [0.0, 0.55, -0.5],    # Q
                ],
                dtype=np.float32,
            ),
            "theta": np.array([0.55, 0.0, 0.7], dtype=np.float32),
            "phi": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        }

    def _init_from_matrix_params(self, ann_params_dict: dict = None) -> None:
        '''
        Validate and load ANN matrices from `(w, W, theta, phi)` parameters.

        Params
        -------
        - ann_params_dict (dict): ANN matrix parameters with keys `w`, `W`, `theta`, `phi`.
        '''
        params = ann_params_dict if ann_params_dict is not None else self._default_ann_params()

        required_keys = ("w", "W", "theta", "phi")
        missing_keys = [k for k in required_keys if k not in params]
        if missing_keys:
            raise ValueError(f"Missing ANN params {missing_keys}. Required keys: {required_keys}.")

        w = np.asarray(params["w"], dtype=np.float32)
        W = np.asarray(params["W"], dtype=np.float32)
        theta = np.asarray(params["theta"], dtype=np.float32)
        phi = np.asarray(params["phi"], dtype=np.float32)

        if w.ndim != 2:
            raise ValueError(f"`w` must be a 2D matrix. Got shape {w.shape}.")
        if W.ndim != 2:
            raise ValueError(f"`W` must be a 2D matrix. Got shape {W.shape}.")
        if theta.ndim != 1:
            raise ValueError(f"`theta` must be a 1D vector. Got shape {theta.shape}.")
        if phi.ndim != 1:
            raise ValueError(f"`phi` must be a 1D vector. Got shape {phi.shape}.")

        expected_w_shape = (3, 2)
        expected_W_shape = (3, 3)
        expected_theta_shape = (3,)
        expected_phi_shape = (3,)

        if w.shape != expected_w_shape:
            raise ValueError(f"`w` must have shape {expected_w_shape}. Got {w.shape}.")
        if W.shape != expected_W_shape:
            raise ValueError(f"`W` must have shape {expected_W_shape}. Got {W.shape}.")
        if theta.shape != expected_theta_shape:
            raise ValueError(f"`theta` must have shape {expected_theta_shape}. Got {theta.shape}.")
        if phi.shape != expected_phi_shape:
            raise ValueError(f"`phi` must have shape {expected_phi_shape}. Got {phi.shape}.")

        output_raw_order = tuple(params.get("output_order", ("A", "P", "Q")))
        valid_labels = {"A", "P", "Q"}
        if len(output_raw_order) != 3 or set(output_raw_order) != valid_labels:
            raise ValueError("Invalid `output_order`: expected a permutation of ('A', 'P', 'Q').")

        if w.shape[0] != theta.shape[0]:
            raise ValueError(f"Inconsistent ANN shapes: number of hidden units in `w` and `theta` do not match ({w.shape[0]} vs {theta.shape[0]}).")
        if W.shape[1] != w.shape[0]:
            raise ValueError(f"Inconsistent ANN shapes: `W` input width must match `w` hidden size ({W.shape[1]} vs {w.shape[0]}).")
        if W.shape[0] != phi.shape[0]:
            raise ValueError(f"Inconsistent ANN shapes: number of outputs in `W` and `phi` do not match ({W.shape[0]} vs {phi.shape[0]}).")
            
        self.w_in_hidden = w.copy()
        self.W_hidden_out = W.copy()
        self.theta_hidden = theta.copy()
        self.phi_out = phi.copy()
        self.output_raw_order = output_raw_order

    def _build_input_vector(self, c_val: float, n_neigh: int) -> np.ndarray:
        '''
        Build reduced ANN input vector `epsilon = [n_neighbors, oxygen]`.

        Params
        -------
        - c_val (float) : Local oxygen concentration.
        - n_neigh (int) : Local occupied-neighbor count.

        Returns
        --------
        - x (np.ndarray) : Reduced ANN input vector of shape `(2,)`.
        '''
        return np.array([float(n_neigh), c_val], dtype=np.float32)
    


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
        ann_copy.output_raw_order = tuple(self.output_raw_order)
        return ann_copy

    def forward(self, c_val: float, n_neigh: int) -> np.ndarray:
        '''
        Compute ANN action scores for one cellular microenvironment.

        Appendix A reduced equations:
        - `V_j = T(sum_k w_jk * epsilon_k - theta_j)`
        - `O_i = T(sum_j W_ij * V_j - phi_i)`
        with `T(x) = 1 / (1 + exp(-2x))`.
        Raw ANN output order is read from `output_order` (default `[A, P, Q]`)
        and then reordered to `[P, Q, A]` to match `SimulationModel.step()`.

        Params
        -------
        - c_val (float) : Local oxygen concentration.
        - n_neigh (int) : Local occupied-neighbor count.

        Returns
        --------
        - out (np.ndarray) : Action scores for `[P, Q, A]` in `[0, 1]`.
        '''
        epsilon = self._build_input_vector(c_val, n_neigh)
        hidden = transfer_sigmoid(self.w_in_hidden @ epsilon - self.theta_hidden)
        out_raw = transfer_sigmoid(self.W_hidden_out @ hidden - self.phi_out)

        raw_by_label = {label: float(out_raw[i]) for i, label in enumerate(self.output_raw_order)}
        out = np.array(
            [raw_by_label["P"], raw_by_label["Q"], raw_by_label["A"]],
            dtype=np.float32,
        )
        return out

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.05,
        epochs: int = 200,
        batch_size: int = 64,
        rng_seed: int = 0,
        verbose: bool = False,
        early_stopping: bool = False,
        es_threshold: float = 0.0001,
        es_count_max: int = 10
    ) -> Dict[str, np.ndarray]:
        '''
        Train ANN parameters on a labeled dataset using mini-batch gradient descent.

        Input feature convention:
        - `X[:, 0] = oxygen_concentration`
        - `X[:, 1] = n_neighbors`

        Label convention:
        - `0 -> P`, `1 -> Q`, `2 -> A`

        Training uses independent binary cross-entropy on raw outputs with
        transfer `T(x)=1/(1+exp(-2x))` and backpropagation through all ANN
        parameters `w`, `W`, `theta`, `phi`.

        Params
        -------
        - X (np.ndarray) : Feature matrix with shape `(N, 2)`.
        - y (np.ndarray) : Integer labels with shape `(N,)`.
        - lr (float) : Learning rate.
        - epochs (int) : Number of optimization epochs.
        - batch_size (int) : Mini-batch size.
        - rng_seed (int) : Shuffle seed for reproducibility.
        - verbose (bool) : If True, print periodic loss values.
        - early_stopping (bool): performs the early stopping (default False).
        - es_count_max (int): is the max number of iteration in which the loss gain is lower than es_theshold.
        - es_threshold (float): is the amount of minimum gain need for the loss to avoid the early stopping case.

        Returns
        --------
        - history (Dict[str, np.ndarray]) : Training history with key `"loss"`.
        '''
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64).reshape(-1)

        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError(f"`X` must have shape (N, 2). Got {X.shape}.")
        if y.ndim != 1:
            raise ValueError(f"`y` must be 1D. Got shape {y.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Inconsistent dataset sizes: X has {X.shape[0]} rows while y has {y.shape[0]}."
            )
        if np.any((y < 0) | (y > 2)):
            raise ValueError("`y` labels must be in {0, 1, 2} for [P, Q, A].")
        if lr <= 0.0:
            raise ValueError(f"`lr` must be > 0. Got {lr}.")
        if epochs <= 0:
            raise ValueError(f"`epochs` must be > 0. Got {epochs}.")
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be > 0. Got {batch_size}.")

        n_samples = X.shape[0]
        batch_size = int(min(batch_size, n_samples))
        rng = np.random.default_rng(rng_seed)
        losses = np.zeros(epochs, dtype=np.float32)

        label_to_output = {0: "P", 1: "Q", 2: "A"}
        raw_idx = {label: i for i, label in enumerate(self.output_raw_order)}
        eps_loss = 1e-7

        for epoch in range(epochs):
            order = rng.permutation(n_samples)
            epoch_loss = 0.0

            for start in range(0, n_samples, batch_size):
                idx = order[start:start + batch_size]
                xb = X[idx]
                yb = y[idx]
                bsz = xb.shape[0]

                grad_w = np.zeros_like(self.w_in_hidden, dtype=np.float32)
                grad_W = np.zeros_like(self.W_hidden_out, dtype=np.float32)
                grad_theta = np.zeros_like(self.theta_hidden, dtype=np.float32)
                grad_phi = np.zeros_like(self.phi_out, dtype=np.float32)
                batch_loss = 0.0
                batch_loss_prec = 0.0

                for sample, label_int in zip(xb, yb):
                    c_val = float(sample[0])
                    n_neigh = int(sample[1])

                    epsilon = self._build_input_vector(c_val, n_neigh)
                    z_hidden = self.w_in_hidden @ epsilon - self.theta_hidden
                    hidden = transfer_sigmoid(z_hidden)

                    z_out = self.W_hidden_out @ hidden - self.phi_out
                    out_raw = transfer_sigmoid(z_out)

                    target = np.zeros(3, dtype=np.float32)
                    target_idx = raw_idx[label_to_output[int(label_int)]]
                    target[target_idx] = 1.0

                    batch_loss_prec = batch_loss
                    batch_loss += float(-np.sum(target * np.log(out_raw + eps_loss) + (1.0 - target) * np.log(1.0 - out_raw + eps_loss)))

                    dL_dz_out = out_raw - target
                    grad_W += np.outer(dL_dz_out, hidden)
                    grad_phi += -dL_dz_out

                    dL_dhidden = self.W_hidden_out.T @ dL_dz_out
                    dT_hidden = 2.0 * hidden * (1.0 - hidden)
                    dL_dz_hidden = dL_dhidden * dT_hidden

                    grad_w += np.outer(dL_dz_hidden, epsilon)
                    grad_theta += -dL_dz_hidden

                    if early_stopping and abs(batch_loss_prec - batch_loss) < es_threshold:
                        es_count += 1
                    else:
                        es_count = 0
                    if es_count > es_count_max:
                        break

                inv_bsz = 1.0 / float(bsz)
                self.w_in_hidden -= lr * grad_w * inv_bsz
                self.W_hidden_out -= lr * grad_W * inv_bsz
                self.theta_hidden -= lr * grad_theta * inv_bsz
                self.phi_out -= lr * grad_phi * inv_bsz

                epoch_loss += batch_loss

                if es_count > es_count_max:
                        break

            losses[epoch] = np.float32(epoch_loss / float(n_samples))
            if es_count > es_count_max:
                print(f"Epoch {epoch + 1:4d}/{epochs}, loss={losses[epoch]:.6f} [early stopping]")
                break
            if verbose and (epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch + 1:4d}/{epochs}, loss={losses[epoch]:.6f}")

        return {"loss": losses}

    def mutate_inplace(self, p: Params, rng: np.random.Generator) -> None:
        '''
        Apply Appendix A per-parameter mutations to ANN genotype in place.

        For each parameter `omega_i`, mutation is:
        - `omega_i' = omega_i + m_i * xi_i`
        - `m_i ~ Bernoulli(p)` and `xi_i ~ N(0, sigma^2)`.

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

        flat = np.concatenate(
            [
                self.w_in_hidden.ravel(),
                self.W_hidden_out.ravel(),
                self.theta_hidden.ravel(),
                self.phi_out.ravel(),
            ]
        ).astype(np.float32, copy=False)

        mut_mask = rng.random(total_entries) < p.p
        if not np.any(mut_mask):
            return

        noise = rng.normal(0.0, p.s, size=total_entries).astype(np.float32)
        flat[mut_mask] += noise[mut_mask]

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

        # Each alive cell can carry one ANN genotype instance (None for non-alive cells).
        self.ann = np.empty((N, N), dtype=object)
        self.ann.fill(None)

        ann = ArtificialNN(ann_params_dict)
        self._seed_tumour(ann)

    def _seed_tumour(
        self,
        ann: ArtificialNN,
    ) -> None:
        '''
        Seed a circular initial tumour at lattice center of 2x2 dim.

        Params
        -------
        - ann (ArtificialNN) : Baseline ANN genotype copied to seeded cells.
        '''
        N = self.p.N
        cx = cy = (N // 2) - 1
        for i in range(2):
            for j in range(2):
                row = cx+i
                col = cy+j
                self.state[row,col] = PROLIFERATING
                self.age_hours[row,col] = 0.0
                self.prolif_age_hours[row,col] = max(
                    1e-3,
                    float(self.rng.normal(self.p.Ap, self.p.Ap_s))
                )
                self.ann[row,col] = ann.copy()

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
        alive = (self.state == PROLIFERATING) | (self.state == QUIESCENT)
        quiescence = alive & (action_map == 1)
        proliferation = alive & (action_map == 0)
        alpha = np.zeros_like(self.c, dtype=np.float32)
        alpha[proliferation] = self.p.rc * F_map[proliferation]
        alpha[quiescence] = (self.p.rc / self.p.q) * F_map[quiescence]
        return alpha

    def _alive_indices_shuffled(self) -> np.ndarray:
        '''
        Return shuffled coordinates of alive cells.

        Params
        -------
        - None (None) : This method does not require additional parameters.

        Returns
        --------
        - coords (np.ndarray) : Shuffled alive-cell coordinates with shape `(M, 2)`.
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
        neighborhood = []
        for di, dj in ((-1,0), (1,0), (0,-1), (0,1)):
            ni, nj = i + di, j + dj
            if (0 <= ni < N) and (0 <= nj < N) and (self.state[ni,nj] == EMPTY or self.state[ni,nj] == DEAD):
                neighborhood.append((ni,nj))
        return tuple(neighborhood)

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
        count = 0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni = (i + di) % N
                nj = (j + dj) % N
                if self.state[ni,nj] == PROLIFERATING or self.state[ni,nj] == QUIESCENT or self.state[ni,nj] == NECROTIC:
                    count += 1
        return count
    
    def _von_neumann_neighbors_count(self, i: int, j: int):
        '''
        Count occupied Von Neumann neighbors around one lattice site.

        Params
        -------
        - i (int) : Row index of the reference cell.
        - j (int) : Column index of the reference cell.

        Returns
        --------
        - count (int) : Occupied-neighbor count in `[0, 3]`.
        '''
        N = self.p.N
        count = 0
        for di, dj in ((-1,0), (1,0), (0,-1), (0,1)):
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if (0 <= ni < N) and (0 <= nj < N) and (self.state[ni,nj] == PROLIFERATING or self.state[ni,nj] == QUIESCENT or self.state[ni,nj] == NECROTIC):
                count += 1
        return count

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

        # Iterates over each alive (PROLIFERATING/QUIESCENT) and shuffled cells coordinates
        # Steps per iteration: scores computation -> consumption check (necrosis) -> death/action -> mitosis.
        coords = self._alive_indices_shuffled()
        for (i,j) in coords:
            if self.state[i,j] != PROLIFERATING and self.state[i,j] != QUIESCENT:
                continue

            # Defensive guard: alive cells should always have an ANN
            ann_ij: ArtificialNN = self.ann[i,j]
            if ann_ij is None:
                continue

            # Compute the inputs for the ANN
            n_ij = self._von_neumann_neighbors_count(i,j)
            c_ij = float(self.c[i,j])

            # 1. Compute the forward pass to get the action as output
            scores = ann_ij.forward(c_ij, n_ij)
            action_idx = int(np.argmax(scores))
            action = 'P' if action_idx == 0 else 'Q' if action_idx == 1 else 'A'

            R = float(scores[action_idx]) # response value
            F_ij = F(R, self.p) # metabolic modulation factor F
            action_map[i,j] = np.int8(action_idx)
            F_map[i,j] = np.float32(F_ij)

            # 2. Nutrients consumption
            self.age_hours[i,j] += self.p.Dt_age_inc * F_ij # age increment
            oxygen_demand_ij = 0.0
            if action == 'P':
                oxygen_demand_ij = self.p.rc * F_ij * self.p.Dt
            elif action == 'Q':
                oxygen_demand_ij = (self.p.rc / self.p.q) * F_ij * self.p.Dt
            
            # 3. Apoptosis case
            if action == 'A':
                self.state[i,j] = DEAD
                self.ann[i,j] = None
                action_map[i,j] = np.int8(-1)
                F_map[i,j] = np.float32(0.0)
                continue
            
            # 4. Necrosis case (oxygen available < oxygen demand)
            if c_ij < oxygen_demand_ij:
                self.state[i,j] = NECROTIC
                self.ann[i,j] = None
                action_map[i,j] = np.int8(-1)
                F_map[i,j] = np.float32(0.0)
                continue

            # 6. Proliferation case (mitosis)
            if action == 'P' and self.age_hours[i,j] >= self.prolif_age_hours[i,j] and n_ij < 4:
                self.state[i,j] = QUIESCENT
                empties = self._von_neumann_empty_neighbors(i, j)
                # Create the daughter cell
                ni, nj = empties[self.rng.integers(0, len(empties))]
                self.state[ni,nj] = PROLIFERATING
                self.ann[ni,nj] = ann_ij.mutated_copy(self.p, self.rng)
                self.age_hours[ni,nj] = 0.0
                self.prolif_age_hours[ni,nj] = max(1e-3, float(self.rng.normal(self.p.Ap, self.p.Ap_s)))
                continue

            # 5. Quiescence case
            if action == 'Q':
                self.state[i,j] = QUIESCENT
                continue

        # Oxygen PDE update from effective actions taken in this CA step
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
        - track_diversity (bool) : If True, compute Shannon diversity over alive-cell genotypes.

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
