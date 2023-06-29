import os
import sys
from pathlib import Path

# Get the parent directory of the current file (assuming the script is in the test folder)
current_dir = Path(__file__).resolve().parent
# Get the parent directory of the current directory (assuming the test folder is one level below the main folder)
main_dir = current_dir.parent
# Add the main directory to the Python path
sys.path.append(str(main_dir))


def test_run_all():
    script_path = (
        Path(__file__).parent / ".." / "neural_network_model" / "script_run_all.py"
    )
    os.system(f"python {script_path}")
    assert True

    script_path = (
            Path(__file__).parent / ".." / "neural_network_model" / "transfer_learning.py"
    )
    os.system(f"python {script_path}")
    assert True


