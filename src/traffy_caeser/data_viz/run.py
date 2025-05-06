import subprocess
import os


def main():
    base_dir = os.path.dirname(__file__)
    main_path = os.path.join(base_dir, "main.py")
    subprocess.run(["streamlit", "run", main_path])
