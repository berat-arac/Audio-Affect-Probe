from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path


def read_metrics(run_dir: Path):
    return json.loads((run_dir / "test_metrics.json").read_text())


def newest(prefix: str) -> Path:
    xs = sorted(Path("results").glob(prefix + "_*"), key=lambda p: p.stat().st_mtime)
    if not xs:
        raise RuntimeError(f"No run directory found for {prefix}")
    return xs[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zip-results", action="store_true", default=True)
    args = ap.parse_args()

    rows=[]
    for kind in ("dual_split", "affect_split"):
        cmd=[sys.executable, "-m", "audio_affect_probe.train", "--config", f"configs/{kind}.yaml", "--device", args.device, "--seed", str(args.seed), "--zip-results"]
        print("\n===", kind, "seed", args.seed, "===", flush=True)
        subprocess.run(cmd, check=True)
        rd=newest(kind)
        m=read_metrics(rd)
        rows.append({"model":kind,"seed":args.seed,"run_dir":str(rd),**{k:m[k] for k in ("valence_ccc","arousal_ccc","mean_ccc","valence_rmse","arousal_rmse")}})

    print("\n=== HEAD PROBE COMPLETE ===")
    for r in rows:
        print(f"{r['model']:12s} val={r['valence_ccc']:.6f} ar={r['arousal_ccc']:.6f} mean={r['mean_ccc']:.6f}")
    Path("exports/head_probe_summary.json").write_text(json.dumps(rows, indent=2))
    print("SUMMARY=exports/head_probe_summary.json")

if __name__ == "__main__":
    main()
