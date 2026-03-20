import logging
import os

LOG_DIR = "logs"
os.makedirs(LOG_DIR,exist_ok = True)
LOG_FILE = os.path.join(LOG_DIR,'app.log')

def get_logger(name:str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(LOG_FILE)
        ch = logging.StreamHandler()

        fmter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmter)
        ch.setFormatter(fmter)
        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger
    
