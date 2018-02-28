import numpy as np

a = np.load("p.npy")

b = np.argmax(np.load("labels.npy"),axis=1)

X = (a==b)

print(np.sum(X)/np.size(a))