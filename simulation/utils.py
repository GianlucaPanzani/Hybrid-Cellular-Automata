from pathlib import Path
from dataclasses import fields
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from IPython.display import HTML
import numpy as np
import matplotlib.pyplot as plt
from model import (
    Params,
    SimulationModel,
    EMPTY,
    PROLIFERATING,
    QUIESCENT,
    NECROTIC,
    DEAD,
)
from model import Params




STATE_COLORS = {
    "empty": "#f3f1f1",
    "proliferating": "#ed2f2f",
    "quiescent": "#239251",
    "necrotic": "#ffbf00",
    "dead": "#2020a8",
}




def _find_env_path(filepath: str = '.env') -> Path:
    '''
    Resolve the environment file path from known simulation locations.

    Params
    -------
    - filepath (str) : Preferred .env filename or path.

    Returns
    --------
    - env_path (Path) : First existing .env candidate path.
    '''

    # Build candidate search paths ordered by priority.
    candidates = [
        Path(filepath),
        Path('simulation/.env'),
        Path('Hybrid-Cellular-Automata/simulation/.env'),
    ]  # candidates: ordered list of possible environment file locations.

    # Return first existing candidate path.
    for path in candidates:  # path: current candidate filesystem path.
        if path.exists():
            return path

    # Raise explicit error when no candidate exists.
    raise FileNotFoundError('.env not found in expected locations')


def _load_env(path: Path) -> dict:
    '''
    Parse key-value pairs from a .env text file.

    Params
    -------
    - path (Path) : Filesystem path to the environment file.

    Returns
    --------
    - env (dict) : Mapping of parsed keys to raw string values.
    '''

    # Initialize output mapping for parsed environment variables.
    env = {}  # env: dictionary of parsed key -> raw string value.

    # Parse file line by line while skipping blanks and comments.
    for raw in path.read_text().splitlines():  # raw: unprocessed line content.
        line = raw.strip()  # line: whitespace-trimmed line.
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            continue

        # Split key/value pair once and normalize surrounding spaces.
        key, value = line.split('=', 1)  # key,value: raw split components.
        env[key.strip()] = value.strip()

    # Return raw environment mapping.
    return env


def _cast_env_value(value: str, target_type):
    '''
    Cast .env string values to expected dataclass field types.

    Params
    -------
    - value (str) : Raw string value read from `.env`.
    - target_type (type) : Expected Python type for the parameter.

    Returns
    --------
    - typed_value (object) : Converted value for supported types, otherwise the original string.
    '''

    # Cast integer-valued fields.
    if target_type is int:
        return int(value)

    # Cast floating-valued fields.
    if target_type is float:
        return float(value)

    # Keep original string for non-int/float fields.
    return value


def get_params_from_env(filepath: str) -> dict:
    '''
    Build Params keyword arguments from .env values plus dataclass defaults.

    Params
    -------
    - filepath (str) : path to .env-style file with parameter overrides. If None is returned an empty dict.

    Returns
    --------
    - params_kwargs (dict) : Mapping of `Params` field names to typed values.
    '''
    if filepath is None:
        return {}

    # Resolve and load environment key-value pairs.
    env_path = _find_env_path(filepath)  # env_path: resolved .env filesystem path.
    env_values = _load_env(env_path)  # env_values: raw .env mapping.

    # Fill parameter dictionary using .env overrides and Params defaults.
    params_kwargs = {}  # params_kwargs: output mapping for Params(**params_kwargs).
    for field in fields(Params):  # field: dataclass field metadata.
        if field.name in env_values:
            params_kwargs[field.name] = _cast_env_value(env_values[field.name], field.type)
        else:
            params_kwargs[field.name] = field.default

    # Return complete parameter mapping.
    return params_kwargs


def create_animation(sim: SimulationModel, p: Params):

    # Pre-create figure with more balanced proportions
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), dpi=110)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.90, bottom=0.14, wspace=0.10)

    # Left animation: cell state visualization
    state_values = [int(EMPTY), int(PROLIFERATING), int(QUIESCENT), int(NECROTIC), int(DEAD)]
    state_labels = ["empty", "proliferating", "quiescent", "necrotic", "dead"]
    state_colors = [
        STATE_COLORS["empty"],
        STATE_COLORS["proliferating"],
        STATE_COLORS["quiescent"],
        STATE_COLORS["necrotic"],
        STATE_COLORS["dead"],
    ]
    state_cmap = ListedColormap(state_colors)
    state_norm = BoundaryNorm(np.arange(len(state_values) + 1) - 0.5, state_cmap.N)
    im_state = ax[0].imshow(sim.state, cmap=state_cmap, norm=state_norm, interpolation="nearest")
    ax[0].set_title("Cell states", fontsize=12, weight="semibold", pad=10)
    ax[0].set_axis_off()
    legend_handles = [
        Patch(facecolor=state_colors[i], edgecolor="none", label=f"{state_values[i]} {state_labels[i]}")
        for i in range(len(state_values))
    ]
    ax[0].legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        fontsize=8,
        framealpha=0.90,
        handlelength=1.1,
        columnspacing=0.9,
    )

    # Right animation: oxygen field
    im_c = ax[1].imshow(sim.c, interpolation="nearest", cmap="viridis")
    im_c.set_clim(0.0, p.c0)
    ax[1].set_title("Oxygen c(x,t)", fontsize=12, weight="semibold", pad=10)
    ax[1].set_axis_off()
    cb = fig.colorbar(im_c, ax=ax[1], fraction=0.050, pad=0.020, shrink=0.90, aspect=30)
    cb.ax.tick_params(labelsize=9)
    cb.set_label("oxygen", fontsize=10)
    txt = fig.text(
        0.5,
        0.045,
        "",
        ha="center",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.88},
    )

    def update_frame(frame_idx: int):
        out = sim.step()
        im_state.set_data(sim.state)
        im_c.set_data(sim.c)
        txt.set_text(
            f"step={frame_idx} | P={out['proliferating']} | Q={out['quiescent']} | N={out['necrotic']} | D={out['dead']} | c_min={out['c_min']:.3f}"
        )
        return (im_state, im_c, txt)

    # Create the animation
    ani = FuncAnimation(fig, update_frame, frames=p.steps, interval=60, blit=False)
    plt.close(fig)
    return HTML(ani.to_jshtml())


def plot_stability_curve(p: Params):
    Dt_max = (p.d * p.d) / (4.0 * p.Dc)
    Dc_max = (p.d * p.d) / (4.0 * p.Dt)

    Ds = np.linspace(0.005, 0.2, 200)
    Dt_max_curve = (p.d * p.d) / (4.0 * Ds)

    # PDE stability (CFL check)
    ok_nok_str = 'OK' if p.Dt <= Dt_max else f'UNSTABLE'
    print(f"PDE stability: Dt <= d^2/(4Dc) --> {ok_nok_str}")
    if ok_nok_str == 'UNSTABLE':
        print(f"\t{p.Dt} <= {p.d * p.d / (4.0 * p.Dc)}")
    elif ok_nok_str == 'OK':
        print(f"\t{p.Dt} <= {Dt_max:.3g}")

    # Show stability margin curve
    plt.figure()
    plt.plot(Ds, Dt_max_curve)
    plt.axhline(p.Dt, linestyle="--")
    plt.xlabel("Dc")
    plt.ylabel("Dt max (CFL)")
    plt.title("PDE stability region: Dt <= d^2/(4Dc)")
    plt.show()