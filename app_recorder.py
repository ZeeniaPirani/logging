import psutil
from datetime import datetime
import time

# path to logging file, can change depending on which file logging should be done in
log_file = "running_apps_log.txt"

# try-except allows user to exit logger without throwing error
try:
    with open(log_file, 'a') as f:
        timestamp = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        while True:
            # writes to logging file, records timestamp
            f.write(f"\n[{timestamp}] Currently Running: \n")
            # creates set to make sure no duplicates are printed out
            seen = set()

            # iterates through process names, adds to file and set if name not seen
            for proc in psutil.process_iter(["name"]):
                try:
                    name = proc.info['name']
                    if name not in seen:
                        f.write(name + ", ")
                        seen.add(name)
                # if permission to view process info is denied or process doesn't exist, skip name
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
    
            # logs every 2 minutes, can change based on frequency of logging
            time.sleep(120)

except KeyboardInterrupt:
    print("\nLogging stopped by user")
    # writes timestamp of user stopping logging for accurate log files
    with open(log_file, 'a') as f:
        f.write("Logging stopped at " + datetime.now().strftime("%d/%m/%Y, %H:%M:%S") + "\n")