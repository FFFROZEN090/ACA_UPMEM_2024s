import numpy as np
import matplotlib.pyplot as plt

# Reading CSV Files
data = np.genfromtxt('kmeans_results.csv', delimiter=',')

# Extracting data points and labels
X = data[:, :-1]  
labels = data[:, -1].astype(int)  

# Visualization
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.title('K-means Clustering Results')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
