"""
Convert h264 videos (in the current folder) to mp4 using MP4Box
"""
import os
import subprocess

files = [f for f in os.listdir(".") if f.endswith(".h264")]

for f in files:
	print("Converting {}".format(f))
	ok = subprocess.call(['MP4Box', '-add', f, f[:-4] + "mp4"])
