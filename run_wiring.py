"""Run the supplied desktop router after validating input filenames."""
import argparse
from pathlib import Path
import os
import runpy
import random

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--workdir", required=True, help="Directory containing input/*.png and matching .txt files")
parser.add_argument("--seed", type=int, default=None, help="Optional random seed for a new run")
args = parser.parse_args()
base = Path(__file__).resolve().parent
work = Path(args.workdir).resolve()
images = sorted((work / "input").glob("*.png"))
if not images:
    parser.error("No PNG plans found under workdir/input")
missing = [p.with_suffix(".txt").name for p in images if not p.with_suffix(".txt").is_file()]
if missing:
    parser.error("Missing detection labels: " + ", ".join(missing))
if len(images) != 1:
    parser.error("Use one plan per work directory to preserve shared output filenames")
if args.seed is not None:
    random.seed(args.seed)
os.chdir(work)
runpy.run_path(str(base / "code" / "custom_astar.py"), run_name="__main__")
