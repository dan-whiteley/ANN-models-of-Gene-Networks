import os
import numpy as np
import pickle

# this script should take a parameter range and number of repeats, creating a large number of trials,
# split into batches equal to number of cores, then send to batch.py to run at once.

#number of nodes (arg 2)
param1 = np.arange(15,16) #min,max,stepsize 0.1 and 0.003

#not used in this example (arg 3)
param2 = np.arange(15,16)

repeats = 1

totaltrials = repeats*len(param1)*len(param2)

trials = np.zeros((totaltrials,3)) #replace p with number of parameters + 1

cores = 6

batches = int(np.floor(totaltrials/cores)+1)

#create a large array, each row is a trial, with cols ID,param1,param2.
offset = 0
ID = offset

for x in range(repeats):
	for p1 in param1:
		for p2 in param2:
			trials[ID - offset] = [ID,p1,p2]
			ID = ID + 1


print(trials)

#compile the script just the once

#replace -l morph.. with ${HOME}/usr/lib/libmorphologica.so on threadbeast

scriptname = "main"
#if morphologica not installed, command = "g++ --std=c++17 " +scriptname+ ".cpp -o compiled -I/$$$PATH_TO$$$/morphologica -lhdf5"
#print(command)
os.system("g++ --std=c++17 " +scriptname+ ".cpp -o compiled `pkg-config --libs libmorphologica`")
#os.system(command)

count = 0

for i in range(batches):
	# take a batch of trials and save them for batch.py to access
	with open("batch.p", "wb") as f:
		pickle.dump(trials[count:count+cores], f) #if this goes out of range numpy will deal fine

	# run the batch
	os.system("python batch.py")

	print(count)
	print("batch done")

	# python should wait for all cores to finish, although if some of them don't finish and there's an overlap it shouldn't break, just slow down.

	count = count + cores


