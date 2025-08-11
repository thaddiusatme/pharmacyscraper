import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dep, tests will cover when installed
    yaml = None


def _substitute_env(value: Any, env: Mapping[str, str]) -> Any:
    """
    Recursively substitute environment variables in strings using
    ${VAR} or ${VAR:-default} syntax. Non-strings are returned as-is,
    and dicts/lists are processed recursively.
    """
    if isinstance(value, str):
        # Support default syntax ${VAR:-default}
        # We'll process repeatedly until no patterns remain or guard against infinite loops.
        def replace_once(s: str) -> str:
            out = s
            start = out.find("${")
            while start != -1:
                end = out.find("}", start + 2)
                if end == -1:
                    break
                token = out[start + 2 : end]
                default = None
                if ":-" in token:
                    var, default = token.split(":-", 1)
                else:
                    var = token
                repl = env.get(var, default if default is not None else "")
                out = out[:start] + str(repl) + out[end + 1 :]
                start = out.find("${")
            return out

        prev = value
        for _ in range(5):  # avoid pathological nesting
            cur = replace_once(prev)
            if cur == prev:
                break
            prev = cur
        return prev
    elif isinstance(value, list):
        return [_substitute_env(v, env) for v in value]
    elif isinstance(value, dict):
        return {k: _substitute_env(v, env) for k, v in value.items()}
    return value


def _validate_and_defaults(cfg: MutableMapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(cfg)

    # Remove env scaffolding if present in the effective config
    out.pop("env", None)
    out.pop("environments", None)

    # Stricter schema: only allow known top-level keys
    allowed_keys = {
        "api_keys",
        "max_results_per_query",
        "output_dir",
        "cache_dir",
        "classification_cache_dir",
        "classification_threshold",
        "verify_places",
        "verification_confidence_threshold",
        "max_budget",
        "api_cost_limits",
        "locations",
        "plugin_mode",
        "plugins",
        "plugin_config",
    }
    unknown = set(out.keys()) - allowed_keys
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")

    # defaults
    out.setdefault("output_dir", "output")
    out.setdefault("cache_dir", "cache")
    out.setdefault("verify_places", True)

    # base type checks
    if not isinstance(out.get("output_dir"), str):
        raise ValueError("output_dir must be a string path")
    if not isinstance(out.get("cache_dir"), str):
        raise ValueError("cache_dir must be a string path")
    if "api_keys" in out and not isinstance(out["api_keys"], dict):
        raise ValueError("api_keys must be a dict if provided")

    # plugin mode structural checks
    if out.get("plugin_mode"):
        plugins = out.get("plugins")
        plugin_config = out.get("plugin_config")
        if plugins is not None and not isinstance(plugins, dict):
            raise ValueError("plugins must be a dict when plugin_mode is true")
        if plugin_config is not None and not isinstance(plugin_config, dict):
            raise ValueError("plugin_config must be a dict when plugin_mode is true")

    return out


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)  # type: ignore[arg-type]
        else:
            result[k] = v
    return result


def load_config(path: str, env: Mapping[str, str] | None = None) -> Dict[str, Any]:
    """
    Load configuration from a JSON file, apply environment variable substitution,
    set defaults, and perform minimal schema validation.

    Parameters:
        path: Path to JSON config file.
        env: Mapping of environment variables to substitute; defaults to os.environ.

    Returns:
        Dict with validated and default-applied configuration.
    """
    env = env or os.environ

    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    with open(path, "r", encoding="utf-8") as f:
        if suffix in (".yaml", ".yml"):
            if yaml is None:
                raise ValueError("PyYAML is required to load YAML config files")
            raw = yaml.safe_load(f)
        else:
            raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("Top-level config must be a JSON object")

    # Apply environment-specific inheritance if present
    base_cfg = dict(raw)
    env_name = base_cfg.get("env")
    env_map = base_cfg.get("environments")
    if env_name is not None:
        if not isinstance(env_map, dict) or env_name not in env_map:
            raise ValueError("env specified but matching entry not found in environments")
        if not isinstance(env_map[env_name], dict):
            raise ValueError("Selected environment override must be a mapping")
        base_cfg = _deep_merge(base_cfg, env_map[env_name])

    substituted = _substitute_env(base_cfg, env)
    validated = _validate_and_defaults(substituted)
    return validated
