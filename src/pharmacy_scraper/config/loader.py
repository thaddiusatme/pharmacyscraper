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
        # Unified schema optional fields
        "search_terms",  # list[str]
        "regions",       # list[str] or list[dict]
    }
    unknown = set(out.keys()) - allowed_keys
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")

    # defaults
    out.setdefault("output_dir", "output")
    out.setdefault("cache_dir", "cache")
    out.setdefault("verify_places", True)

    # base type checks (only when present)
    if not isinstance(out.get("output_dir"), str):
        raise ValueError("output_dir must be a string path")
    if not isinstance(out.get("cache_dir"), str):
        raise ValueError("cache_dir must be a string path")
    if "api_keys" in out and not isinstance(out["api_keys"], dict):
        raise ValueError("api_keys must be a dict if provided")

    if "max_results_per_query" in out and not isinstance(out["max_results_per_query"], int):
        raise ValueError("max_results_per_query must be an integer if provided")
    if "classification_threshold" in out:
        ct = out["classification_threshold"]
        if not isinstance(ct, (int, float)):
            raise ValueError("classification_threshold must be a number if provided")
        if not (0.0 <= float(ct) <= 1.0):
            raise ValueError("classification_threshold must be between 0.0 and 1.0")
    if "verification_confidence_threshold" in out:
        vt = out["verification_confidence_threshold"]
        if not isinstance(vt, (int, float)):
            raise ValueError("verification_confidence_threshold must be a number if provided")
        if not (0.0 <= float(vt) <= 1.0):
            raise ValueError("verification_confidence_threshold must be between 0.0 and 1.0")
    if "max_budget" in out:
        mb = out["max_budget"]
        if not isinstance(mb, (int, float)):
            raise ValueError("max_budget must be a number if provided")
        if float(mb) < 0:
            raise ValueError("max_budget cannot be negative")
    if "api_cost_limits" in out:
        limits = out["api_cost_limits"]
        if not isinstance(limits, dict):
            raise ValueError("api_cost_limits must be a dict if provided")
        for k, v in limits.items():
            if not isinstance(v, (int, float)):
                raise ValueError(f"api_cost_limits['{k}'] must be a number")
            if float(v) < 0:
                raise ValueError(f"api_cost_limits['{k}'] cannot be negative")
    if "locations" in out:
        locs = out["locations"]
        if not isinstance(locs, list):
            raise ValueError("locations must be a list if provided")
        # Back-compat: accept list of strings OR list of dicts and normalize
        normalized: list[str] = []
        for item in locs:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                # Try common legacy shapes
                city = item.get("city")
                state = item.get("state")
                cities = item.get("cities")
                name = item.get("name")
                query = item.get("query") or item.get("q")
                if cities and state and isinstance(cities, list):
                    for c in cities:
                        if isinstance(c, str) and c.strip():
                            normalized.append(f"{c}, {state}")
                    continue
                if city and state:
                    normalized.append(f"{city}, {state}")
                elif name:
                    normalized.append(str(name))
                elif query:
                    normalized.append(str(query))
                else:
                    raise ValueError(
                        "location dict must include at least one of: (city and state), cities+state, name, or query"
                    )
            else:
                raise ValueError("locations must be a list of strings or dicts")
        out["locations"] = normalized

    # Unified schema optional fields
    if "search_terms" in out:
        terms = out["search_terms"]
        if not isinstance(terms, list) or not all(isinstance(x, str) and x.strip() for x in terms):
            raise ValueError("search_terms must be a non-empty list of strings if provided")
    if "regions" in out:
        regions = out["regions"]
        if isinstance(regions, list):
            # Accept list of strings or list of dicts with minimal structure
            for r in regions:
                if isinstance(r, str):
                    continue
                if isinstance(r, dict):
                    # allow keys like {state, city} or {name}
                    if not any(k in r for k in ("state", "city", "name")):
                        raise ValueError("region dict must include at least one of: state, city, name")
                    continue
                raise ValueError("regions must be a list of strings or dicts")
        else:
            raise ValueError("regions must be a list if provided")

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
