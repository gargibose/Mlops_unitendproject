import sys
import logging
import src.logger
from src.logger import logging


def error_message_detail(error, error_detail: sys):
    """
    Creates a detailed error message containing:
    - File name where the exception occurred
    - Line number of the exception
    - Original error message
    """

    # exc_info() returns a tuple:
    # (exception type, exception value, traceback object)
    _, _, exc_tb = error_detail.exc_info()

    # Get the filename where the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Create a detailed error message
    error_message = (
        "Error occurred in python script name [{0}] "
        "line number [{1}] "
        "error message [{2}]"
    ).format(
        file_name,
        exc_tb.tb_lineno,
        str(error)
    )

    return error_message


class CustomException(Exception):

    def __init__(self, error_message, error_detail: sys):
        # Call the constructor of the parent Exception class
        super().__init__(error_message)

        # Generate the detailed error message
        self.error_message = error_message_detail(
            error_message,
            error_detail=error_detail
        )

    def __str__(self):
        # This method is automatically called when the exception is printed
        return self.error_message
if __name__ == "__main__":
    try:
        a = 1 / 0  # This will raise a ZeroDivisionError

    except Exception as e:
        logging.info("Division by zero")
        raise CustomException(e, sys)  # Raise our custom exception with details
''' Exception occurs
        ↓
except Exception as e
        ↓
raise CustomException(e, sys)
        ↓
error_message_detail()
        ↓
Gets:
  - File name
  - Line number
  - Actual error
        ↓
Creates formatted message
        ↓
__str__() returns message when printed '''