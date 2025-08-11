import json
import os
from typing import Any, Dict, Mapping, MutableMapping


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

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("Top-level config must be a JSON object")

    substituted = _substitute_env(raw, env)
    validated = _validate_and_defaults(substituted)
    return validated
