import os
import subprocess

root_dir = os.path.dirname(os.path.abspath(__file__))
clean_dir = os.path.join(root_dir, "GPS_clean.py")
subprocess.run(["python", clean_dir])
smooth_dir = os.path.join(root_dir, "GPS_smooth.py")
subprocess.run(["python", smooth_dir])
