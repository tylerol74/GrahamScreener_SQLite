import os
import sys
import subprocess
from datetime import datetime


# -------------------------
# SETTINGS
# -------------------------

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

GRAHAM_SCRIPT = os.path.join(PROJECT_DIR, "graham_screener.py")
TECHNICAL_SCRIPT = os.path.join(PROJECT_DIR, "technical_scanner.py")
MERGE_SCRIPT = os.path.join(PROJECT_DIR, "merge_opportunities.py")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

os.makedirs(LOG_DIR, exist_ok=True)


# -------------------------
# HELPERS
# -------------------------

def run_script(script_path, batch_number=None):
    script_name = os.path.basename(script_path)

    command = [sys.executable, script_path]

    if batch_number is not None:
        command.append(str(batch_number))

    print()
    print("=" * 60)
    print(f"Running {script_name}")
    if batch_number is not None:
        print(f"Batch number: {batch_number}")
    print("=" * 60)

    result = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        text=True
    )

    if result.returncode != 0:
        print()
        print(f"ERROR: {script_name} failed.")
        print(f"Exit code: {result.returncode}")
        sys.exit(result.returncode)

    print()
    print(f"Finished {script_name}")


# -------------------------
# MAIN
# -------------------------

if len(sys.argv) > 1:
    batch_number = int(sys.argv[1])
else:
    batch_number = 1

start_time = datetime.now()

print()
print("Starting value screener batch run")
print(f"Batch: {batch_number}")
print(f"Started: {start_time}")

run_script(GRAHAM_SCRIPT, batch_number)
run_script(TECHNICAL_SCRIPT, batch_number)
run_script(MERGE_SCRIPT)

end_time = datetime.now()
duration = end_time - start_time

print()
print("=" * 60)
print("Batch run complete")
print(f"Batch: {batch_number}")
print(f"Finished: {end_time}")
print(f"Duration: {duration}")
print("=" * 60)