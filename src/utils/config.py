import yaml
from pathlib import Path

def _check_exists(paths_dict):
    missing = []

    for group in ["signal", "background"]:
        for name, flist in paths_dict.get(group, {}).items():
            for p in flist:
                if not p.exists():
                    missing.append(str(p))

    if missing:
        raise FileNotFoundError(
            "Missing input files:\n" + "\n".join(f"  - {m}" for m in missing)
        )

def load_paths(path_file="configs/path.yaml"):
    with open(path_file, "r") as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg["data_root"])

    if not base.exists():
        alt = cfg.get("fallback_data_root", None)
        if alt:
            alt = Path(alt)
            if alt.exists():
                print(f"[config] data_root '{base}' not found; using fallback_data_root '{alt}'")
                base = alt
            else:
                raise FileNotFoundError(
                    f"Neither data_root '{base}' nor fallback_data_root '{alt}' exists"
                )
        else:
            raise FileNotFoundError(
                f"data_root '{base}' does not exist and no fallback_data_root provided"
            )


    paths = {
        "signal": {
            k: [base / p for p in v]
            for k, v in cfg["signal"].items()
        },
        "background": {
            k: [base / p for p in v]
            for k, v in cfg["background"].items()
        },
    }
    
    _check_exists(paths)

    return paths

