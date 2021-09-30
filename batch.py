import os
import numpy as np
import pickle

with open("batch.p", "rb") as f:
	batch = pickle.load(f)

command = ''


for row in batch:
	command = command + "./compiled " + str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + " &"

command = command + "& fg" # && fg keeps latest script in foreground so ctrl-C can close

os.system(command)


