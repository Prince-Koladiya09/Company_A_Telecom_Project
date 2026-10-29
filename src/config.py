from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "telecom"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

CLIENT_FILE = DATA_DIR / "Client.csv"
RECORD_FILE = DATA_DIR / "Record.csv"

TARGET = "churn"
ID_COLUMN = "Customer_ID"
RANDOM_STATE = 42
TEST_SIZE = 0.30

for folder in (OUTPUT_DIR, FIGURE_DIR, TABLE_DIR):
    folder.mkdir(parents=True, exist_ok=True)
