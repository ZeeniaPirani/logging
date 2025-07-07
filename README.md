# Analyzing Screenshots in Python

This project analyzes user-inputted screenshots using text, face, and object recognition in Python.

---

### Prerequisites
1. **Python 3.10**
  - For Windows - install from: https://www.python.org/downloads/windows/
  - If the newest version is already installed, create a virtual environment with Python 3.10
    - Creating a virtual environment: https://docs.python.org/3/library/venv.html

2. **CMake**
  - Download: https://cmake.org/download/
  - Make sure CMake is added to the system PATH

3. **Visual Studio (Windows only)**
  - Required for some packages, such as face_recognition
  - Download: https://visualstudio.microsoft.com/downloads/

---

### Set Up Instructions
1. **Download necessary files and create a project folder**
  - `MobileNetSSD_deploy.caffemodel`
  - `MobileNetSSD_deploy.prototxt`
  - `analyze_screenshots.py`
  - `requirements.txt`

2. **Set up and activate a virtual environment (if not on Python 3.10)**
  - Linux/MacOS
```
python3.10 -m venv <venv_name>
source <venv_name>/bin/activate
```
  - Windows (Command Prompt)
```
py -3.10 -m venv <venv_name>
<venv_name>\Scripts\activate.bat
```
  - Windows (Powershell)
```
py -3.10 -m venv <venv_name>
<venv_name>\Scripts\Activate.ps1
```

3. **Install dependencies from requirements.txt**
  - `pip install -r requirements.txt`

4. **Open the project directory in PyCharm**

5. **Place the image you want to analyze in the same folder as code files**

 6. **Run code from the Windows terminal**
  - Use command `python analyze_screenshots.py`
  - Make sure that the virtual environment is activated before running the script
