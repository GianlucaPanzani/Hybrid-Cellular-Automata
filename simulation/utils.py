from pathlib import Path
from dataclasses import fields
from model import Params


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


def get_params_from_env(filepath: str = '.env') -> dict:
    '''
    Build Params keyword arguments from .env values plus dataclass defaults.

    Params
    -------
    - filepath (str) : Preferred .env filename or path.

    Returns
    --------
    - params_kwargs (dict) : Mapping of `Params` field names to typed values.
    '''

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
