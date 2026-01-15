import logging
import os
from datetime import datetime

# Create timestamped log file
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# CORRECT: Create 'logs' directory path (NO filename)
logs_dir = os.path.join(os.getcwd(), "logs")  # Just the directory
os.makedirs(logs_dir, exist_ok=True)  # Create the directory

# CORRECT: Create full path to log file
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)  # Directory + filename

# Configure Logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

""" if __name__ == "__main__":
    logging.info("Logging has started") """