from pathlib import Path
from dataclasses import fields
from model import Params

def _find_env_path() -> Path:
    candidates = [
        Path(".env"),
        Path("simulation/.env"),
        Path("Hybrid-Cellular-Automata/simulation/.env"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(".env not found in expected locations")


def _load_env(path: Path) -> dict:
    env = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def _cast_env_value(value: str, target_type):
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value


def get_params_from_env() -> dict:
    env_path = _find_env_path()
    env_values = _load_env(env_path)
    params_kwargs = {}
    for field in fields(Params):
        if field.name in env_values:
            params_kwargs[field.name] = _cast_env_value(env_values[field.name], field.type)
        else:
            params_kwargs[field.name] = field.default
    return params_kwargs