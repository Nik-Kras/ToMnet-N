"""
One function to create a logger for any flow / endpoint / utility
"""
import logging
import os

LOG_FORMAT="[%(levelname)s] - (%(asctime)s) - FUN:%(funcName)s():%(lineno)s >>> %(message)s"

def create_logger(logger_name: str, logger_filename: str = None):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("{}".format(LOG_FORMAT), "%Y-%m-%d %H:%M:%S")

    # create file handler which logs even debug messages
    if logger_filename:
        
        top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

        # Construct the absolute path to the output directory and create it if it doesn't exist
        output_dir = os.path.join(top_level_dir, 'logs')
        os.makedirs(output_dir, exist_ok=True)

        # Construct the absolute path to the output file
        output_path = os.path.join(output_dir, f'{logger_filename}.log')
        
        file_handler = logging.FileHandler(output_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger