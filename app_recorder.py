import psutil
from datetime import datetime
import time

log_file = "running_apps_log.txt"

try:
    with open(log_file, 'a') as f:
        timestamp = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        while True:
            f.write(f"\n[{timestamp}] Currently Running: \n")
            seen = set()

            for proc in psutil.process_iter(["name"]):
                try:
                    name = proc.info['name']
                    if name not in seen:
                        f.write(name + ", ")
                        seen.add(name)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
    
            time.sleep(120)

except KeyboardInterrupt:
    print("\nLogging stopped by user")
    with open(log_file, 'a') as f:
        f.write("Logging stopped at " + datetime.now().strftime("%d/%m/%Y, %H:%M:%S") + "\n")