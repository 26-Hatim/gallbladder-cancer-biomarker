"""Run the pipeline steps in order."""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = {1: "01_download_and_clean.py", 2: "02_harmonize.py", 3: "03_differential_expression.py",
         4: "04_pathways.py", 5: "05_model_and_validation.py", 6: "06_gene_lookup_and_literature.py",
         7: "07_figures.py"}
OPTIONAL = {4, 6, 7}

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="start", type=int, default=1)
ap.add_argument("--skip", type=int, nargs="*", default=[])
args = ap.parse_args()

(HERE / "results").mkdir(exist_ok=True)
log = open(HERE / "results" / "run_log.txt", "a", encoding="utf-8")
for n, script in STEPS.items():
    if n < args.start or n in args.skip:
        continue
    print(f"\n######## STEP {n}: {script}", flush=True)
    log.write(f"\n######## STEP {n}: {script}\n")
    proc = subprocess.Popen([sys.executable, str(HERE / script)], cwd=HERE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
        log.write(line)
    proc.wait()
    if proc.returncode != 0:
        msg = f"STEP {n} failed (exit code {proc.returncode})."
        if n in OPTIONAL:
            print(msg + " It is optional, continuing.")
            continue
        print(msg + " Fix it, then run:  python run_all.py --from %d" % n)
        sys.exit(proc.returncode)
print("\nAll done. See results/ and figures/.")
