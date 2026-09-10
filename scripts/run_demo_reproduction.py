from __future__ import annotations

import contextlib
import datetime as dt
import os
import platform
import subprocess
import sys
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[1]
    log_path = repo / "logs" / "reproduction_demo.log"
    env_log = repo / "logs" / "environment_info.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env_text = [
        "=== Environment information ===",
        f"timestamp_utc={dt.datetime.utcnow().isoformat()}Z",
        f"python={sys.version.replace(chr(10), ' ')}",
        f"platform={platform.platform()}",
        f"cwd={repo}",
    ]
    for pkg in ["numpy", "scipy", "sklearn", "torch", "matplotlib"]:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            version = f"NOT_AVAILABLE: {exc}"
        env_text.append(f"{pkg}={version}")
    env_log.write_text("\n".join(env_text) + "\n", encoding="utf-8")

    cmd = [sys.executable, "-m", "oiw_repro.train", "--config", "configs/demo_config.json", "--synthetic-demo"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    with log_path.open("w", encoding="utf-8") as f:
        f.write("=== Reproduction demo smoke-test log ===\n")
        f.write(f"timestamp_utc={dt.datetime.utcnow().isoformat()}Z\n")
        f.write("This demo uses deterministic synthetic data and does not claim to be the original paper training log.\n")
        f.write("command=" + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=repo, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
        f.write(f"\nexit_code={proc.returncode}\n")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    print(f"Wrote {log_path}")
    print(f"Wrote {env_log}")


if __name__ == "__main__":
    main()
