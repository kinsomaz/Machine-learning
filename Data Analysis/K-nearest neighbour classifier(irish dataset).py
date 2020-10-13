print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 5
# import some data to play with
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target
X_train = X[:-25]
y_train = y[:-25]

X_test = X[-25:]
y_test = y[-25:]

h = .02  # step size in the mesh

# we create an instance of Neighbours Classifier and fit the data.
model = neighbors.KNeighborsClassifier(n_neighbors)
model.fit(X_train,y_train)
nbrs =neighbors.NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X_train)

query_point = X_test[7]
true_class_of_query_point = y_test[7]
predicted_class_for_query_point = model.predict([query_point])
print("predicted_class_for_query_point:%i"%(predicted_class_for_query_point))
print("true_class_of_query_point:%i"%(true_class_of_query_point))

#finding the index and distance of the nearest neighbors
n = query_point.reshape(1, -1)
distances, indices = nbrs.kneighbors(n)
print("nearestNeighbor:")
print(X_train[indices])
print("distances_of_index:")
print(distances[0][0])
print(distances[0][1])
print(distances[0][2])
# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold,edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i)"% (n_neighbors))

# Plot also the query point
plt.scatter(query_point[0], query_point[1], c=true_class_of_query_point , cmap='Purples' ,edgecolor='k', s=20)

plt.show()








