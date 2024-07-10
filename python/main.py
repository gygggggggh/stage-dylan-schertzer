import subprocess
import logging
from typing import List


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(script_path: str) -> None:
    """
    Run a Python script using subprocess.
    
    Args:
    script_path (str): Path to the Python script to run.
    """
    logger.info(f"Running script: {script_path}")
    try:
        result = subprocess.run(["python", script_path], check=True, capture_output=True, text=True)
        logger.info(f"Script {script_path} completed successfully.")
        logger.debug(f"Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running script {script_path}: {e}")
        logger.error(f"Error output: {e.stderr}")

def main() -> None:
    scripts: List[str] = [
        "python/simclr_IT/trainIT.py",
        "python/simclr_IT/testIT.py",
        "python/simCLR_resnet/trainRN.py",
        "python/simCLR_resnet/testRN.py",
        "python/LR/train_testLR.py",
        "python/graph.py"
    ]

    # Run each script
    for script in scripts:
        run_script(script)

    logger.info("All scripts have been executed.")

if __name__ == "__main__":
    main()