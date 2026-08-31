import argparse
from audio_affect_probe.utils import package_run

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--out", default="exports")
args = ap.parse_args()
print(package_run(args.run_dir, args.out))
