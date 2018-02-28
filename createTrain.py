import numpy as np
from scipy import misc
import os

letter = misc.imread("./Train/character_1_ka/1340.png")

print(letter.shape)

dir = os.fsencode("./Train")

label = np.empty((1,36))

final = np.empty((1,1023))
labels = np.empty((1,36))

for file in os.listdir(dir):
    filename = os.fsdecode(file)
    if(filename.endswith(".npy")):
        f = os.path.join(dir,file)
        data = np.load(os.fsdecode(f))[1:]

        cat=data[0][0]
        data = np.delete(data,0,axis=1)

        l = np.zeros((data.shape[0],35))
        if(cat<36):
            l = np.insert(l,int(cat),1,axis=1)
        else:
            l = np.append(l, np.ones((data.shape[0],1)), axis=1)

        final = np.concatenate((final, data),axis=0)
        labels = np.concatenate((labels, l), axis=0)

np.save("images",final[1:])
np.save("labels",labels[1:])