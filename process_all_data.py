import os
import platform
import argparse
import json
import subprocess

parser = argparse.ArgumentParser(description="Processes all screenshots in directory")
parser.add_argument("file_path", help="Path to screenshot directory")
args = parser.parse_args()
screenshot_directory = args.file_path

if not os.path.exists(screenshot_directory):
    print("Directory does not exist")
    exit()

screenshots = []
for screenshot in os.listdir(screenshot_directory):
    if screenshot.lower().endswith(".png"):
        screenshots.append(screenshot)

screenshots.sort()

if platform.system() == "Windows":
    venv_path = os.path.join("newvenv", "Scripts", "python.exe")
else:
    venv_path = os.path.join("newvenv", "bin", "python3.10")

for screenshot in screenshots:
    image_path = os.path.join(screenshot_directory, screenshot)
    json_name = os.path.splitext(screenshot)[0] + ".json"
    json_path = os.path.join(screenshot_directory, json_name)

    subprocess.run([venv_path, "analyze_screenshots.py", image_path, image_path, json_path])

json_files = []
for file in os.listdir(screenshot_directory):
    if file.lower().endswith(".json"):
        json_files.append(file)

json_files.sort()

all_data = {}

for json_file in json_files:
    json_path = os.path.join(screenshot_directory, json_file)
    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
            key_name = os.path.splitext(json_file)[0]
            all_data[key_name] = data
            print(f"Loaded {json_file}")
        except json.JSONDecodeError:
            print(f"Failed to parse {json_file}")

combined_path = os.path.join(screenshot_directory, "results.json")
with open(combined_path, 'w') as f:
    json.dump(all_data, f, indent=4)