import numpy as np
from sklearn import decomposition
from sklearn.svm import LinearSVC

data = np.load('images.npy')
lab = np.load('labels.npy')

pca = decomposition.PCA(n_components=16)

cls = LinearSVC(random_state=0)

labels = np.argmax(lab,axis=1)

pca.fit(data)
f = pca.fit_transform(data)

# print(f.shape)
# print(labels.shape)

cls.fit(f,labels)

print(cls.score(f,labels))
# print(f.shape)

J = cls.predict(f)
np.save("abx",J)







