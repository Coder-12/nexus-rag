import logging
import sys
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        handlers=[
            # Console (Docker / K8s friendly)
            logging.StreamHandler(sys.stdout),

            # File (audit + debugging)
            logging.FileHandler(LOG_DIR / "nexus_rag.log"),
        ],
    )