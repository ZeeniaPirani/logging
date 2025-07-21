from PIL import ImageGrab
import time
import os

# Creates folder screenshots (if not already created) to store all screenshots
folder_name = "screenshots" 
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

# Constantly runs loop
while True:
    # Takes a screenshot and sets current time to some variable (for suitable logging name?)
    screenshot = ImageGrab.grab()
    currenttime = time.localtime()

    # Converts time into properly formatted string, creates file name for screenshot
    str_from_time = time.strftime("%Y-%m-%d_%H-%M-%S", currenttime)
    file_name = str_from_time + ".png"
    
    # Moves and saves screenshot in proper folder
    file_path = os.path.join(folder_name, file_name)
    screenshot.save(file_path)

    # Pauses for 10 seconds, runs loop again
    time.sleep(10)