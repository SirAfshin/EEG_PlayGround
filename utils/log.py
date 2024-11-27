import sys
import os

# Dynamically add the root directory to sys.path
# Assumes that 'models' and 'utils' are in the same project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import logging

formatter = logging.Formatter(
     fmt="{asctime} - {levelname} - {message}",
     style="{",
     datefmt="%Y-%m-%d %H:%M:%S",
     )

def get_logger(file_path:str): 
    """
    Creates and configures a logger that writes log messages to both the console
    and a specified file.

    Parameters:
    file_path (str): The path to the log file where log messages should be written.

    Returns:
    logging.Logger: A logger instance that writes log messages to the console
                    and the specified log file.
    
    The logger is set to the 'INFO' logging level, meaning messages at the
    'INFO' level and higher will be logged. If there are any existing handlers
    attached to the logger, they are cleared before new handlers are added to
    avoid duplicate log entries.
    """
    # create the logger
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")

    # Remove existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # create handlers
    # console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")

    # Add format to handlers
    # console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # logger.addHandler(console_handler)
    logger.addHandler(file_handler)
 
    return logger

def get_logger_with_console(file_path:str): 
    """
    Creates and configures a logger that writes log messages to both the console
    and a specified file.

    Parameters:
    file_path (str): The path to the log file where log messages should be written.

    Returns:
    logging.Logger: A logger instance that writes log messages to the console
                    and the specified log file.
    
    The logger is set to the 'INFO' logging level, meaning messages at the
    'INFO' level and higher will be logged. If there are any existing handlers
    attached to the logger, they are cleared before new handlers are added to
    avoid duplicate log entries.
    """
    # create the logger
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")

    # Remove existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")

    # Add format to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
 
    return logger


if __name__ == "__main__":
    # Testing the logger
    lger = get_logger('log_test.txt')
    lger.info("THIS IS THE FIRST TIME")
    lger.info("TT")
    lger.info("222")

    lger = get_logger('log2_test.txt')
    lger.info("log33")