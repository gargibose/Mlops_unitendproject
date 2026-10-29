import logging  #we use logging module to log the information for the files in the project
import os #os module is used to interact with the operating system, such as creating directories and handling file paths
from datetime import datetime#datetime module is used to work with dates and times, such as getting the curre
import sys

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y__%H_%M_%S')}.log" #we create a log file with the current timestamp in the filename
log_path = os.path.join(os.getcwd(),"logs",LOG_FILE) #we create a path for the log file in the logs directory
os.makedirs(log_path,exist_ok=True) #we create the logs directory if it doesn't exist
LOG_FILE_PATH = os.path.join(log_path,LOG_FILE) #we create the full path for the log file that includes the logs directory and the log file name
logging.basicConfig(
    filename=LOG_FILE_PATH, #we specify the log file path where the logs will be written
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s", #we specify the format of the log messages
    level=logging.INFO #we set the logging level to INFO, which means that all messages with a severity level of INFO or higher will be logged
)
   
'''Error Occurs
     │
     ▼
Logging Records Event
     │
     ▼
CustomException Raised
     │
     ▼
sys.exc_info()
     │
     ▼
Traceback Object
     │
     ├── File Name
     ├── Line Number
     └── Original Error
     │
     ▼
Formatted Error Message
     │
     ▼
Displayed to User'''