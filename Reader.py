import numpy as np
from scipy import misc
import os

letter = misc.imread("./Train/character_1_ka/1340.png")

print(letter.shape)

dir = os.fsencode("./Train")

for file in os.listdir(dir):
    filename = os.fsdecode(file)

    if(filename.split('_')[0] == 'character'):
        cat = filename.split('_')[1]
        imageDir = os.path.join(dir,file)
        print(imageDir)

        arr = np.empty((1,1024))

        for imagefiles in os.listdir(imageDir):
            img = os.path.join(imageDir,imagefiles)

            letter = misc.imread(img)
            letter = letter.flatten().reshape((1,-1))

            letter[0][0] = cat

            arr = np.concatenate((arr,letter),axis=0)

        np.save(str(cat),arr)
        print(arr.shape)







