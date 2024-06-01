"""
# Importing the libraries
"""

from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm, null_space, inv
from scipy.sparse.linalg import eigs, LinearOperator
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import networkx as nx
import time



"""# SC ALgorithm"""

def sc(W, D, k):
  start_time = time.time()

  # step 1: Find laplacian matrix L
  L = D - W
  # step 2: find sqrt of D matrix
  sqrtD = sqrtm(D)
  # step 3: find normalized Laplacian matrix
  Ln = np.linalg.inv(sqrtD) @ L @ np.linalg.inv(sqrtD)
  # step 4: compute eigenvalues of Ln
  Ln_eigenvalues, Ln_eigenvectors = np.linalg.eig(Ln)
  sorted_indices = np.argsort(Ln_eigenvalues)  # Sort in ascending order
  Ln_eigenvalues = Ln_eigenvalues[sorted_indices][:k]
  X = Ln_eigenvectors[:, sorted_indices][:, :k]
  X = np.real(X)
  # step 5: find H matrix
  H = np.linalg.inv(sqrtD) @ X
  H = np.real(H)
  # step 6: apply k-means clustering on H
  kmeans = KMeans(n_clusters=k, n_init=10, max_iter=500)
  clusterLabels = kmeans.fit_predict(H)

  end_time = time.time()
  execution_time = end_time - start_time

  return clusterLabels, execution_time



"""# Importing dataset"""

friendship = pd.read_csv('../data/friendship_clean.csv', header=None)
group = pd.read_csv('../data/group_clean.csv', header=None, dtype=int)

G = nx.Graph()

for m in range(friendship.shape[0]):
    i = group.index[group[0] == friendship.iloc[m, 0]][0]
    j = group.index[group[0] == friendship.iloc[m, 1]][0]
    if friendship.iloc[m, 2] == 1:
        G.add_edge(i, j)

W = nx.to_numpy_array(G)
n = W.shape[0]

D = np.diag(np.sum(W, axis=1))

print("details of weight matrix:")
print(W)
print(W.shape)

print("details of degree matrix:")
print(D.shape)
print(D)

components = nx.connected_components(G)
largest_component = max(components, key=len)
SG = G.subgraph(largest_component)

group = group.sort_values(by=0)

g = group.iloc[list(largest_component), :]
gmale = g[1].values
gfemale = np.double(~gmale)

F = (gfemale - np.sum(gfemale)/n).reshape(-1, 1)

print("F matrix")
print(F)
print(F.shape)



"""# Getting node positions and plotting"""

node_positions = nx.spring_layout(G)
X = np.array([pos for _, pos in node_positions.items()])
print("printing node positions:")
print(X)

x_coords = [pos[0] for pos in node_positions.values()]
y_coords = [pos[1] for pos in node_positions.values()]



plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(x_coords, y_coords)
# nx.draw(G, pos=node_positions, with_labels=False, edge_color='none')
plt.title('Original Graph')

k = 3
clusterLabels, execution_time = sc(W, D, k)
print(clusterLabels)
print("execution time: " + str(execution_time))


# plotting
unique_clusters = set(clusterLabels)
num_clusters = len(unique_clusters)
color_map = plt.cm.tab10  # Use tab10 colormap directly
cluster_colors = [color_map(i) for i in range(num_clusters)]  # Assign a unique color for each cluster

# Get x and y coordinates from node positions
# x_coords = [pos[0] for pos in node_positions.values()]
# y_coords = [pos[1] for pos in node_positions.values()]

# Plot nodes with different colors for each cluster
plt.figure(figsize=(10, 5))
plt.scatter(x_coords, y_coords, c=[cluster_colors[cluster] for cluster in clusterLabels])  # Scatter plot nodes with cluster-specific colors
plt.title('Clustered Graph - Spectral Clustering')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()



"""# Get group partitions"""

group_partitions = []

F_ = [item for sublist in F for item in sublist]
print(F_)

group_partitions1 = set()
group_partitions2 = set()

for i in range(len(F_)):
  if(F_[i] < 0):
    group_partitions1.add(i)
  else:
    group_partitions2.add(i)

group_partitions.append(group_partitions1)
group_partitions.append(group_partitions2)

print(group_partitions)



"""# Evaluating balance and average balance

"""

def balance(cluster, group_partitions):
  min_balance = float('inf')

  for group_partition in group_partitions:
    for other_group_partition in group_partitions:
      if group_partition != other_group_partition:
        intersection = len(cluster.intersection(group_partition))
        other_intersection = len(cluster.intersection(other_group_partition))
        balance_value = intersection / other_intersection
        min_balance = min(min_balance, balance_value)

  return min_balance

def average_balance(clusters, group_partitions):
  k = len(clusters)
  total_balance = sum(balance(cluster, group_partitions) for cluster in clusters)
  return total_balance / k



"""# Getting clusters"""

def getclusters(clusterLabels):
  clusterLabels_ = clusterLabels
  print(clusterLabels_)

  clusters = []
  unique_cluster_labels = set(clusterLabels_)
  for i in range(len(unique_cluster_labels)):
    clusters.append(set())

  for i in range(len(clusterLabels_)):
    clusters[clusterLabels_[i]].add(i)

  return clusters



"""# Checking for different k values"""

avg_balances = []
avg_balances.append(0)
avg_balances.append(0)

for i in range(2, 7):
  num_clusters = i
  color_map = plt.cm.tab10  # Use tab10 colormap directly
  cluster_colors = [color_map(i) for i in range(num_clusters)]

  cluster_labels, execution = sc(W, D, i)
  print("execution time for " + str(i) + " clusters is: " + str(execution))

  # get clusters
  clusters = getclusters(cluster_labels)
  print("printing clusters:\n")
  print(clusters)

  # getting avg balance
  avg_balance = average_balance(clusters, group_partitions)
  print("printing avg balance")
  print(avg_balance)
  avg_balances.append(avg_balance)

  plt.figure(figsize=(8, 6))
  plt.scatter(x_coords, y_coords, c=[cluster_colors[cluster] for cluster in cluster_labels], label="n_cluster-"+str(i))  # Scatter plot nodes with cluster-specific colors
  plt.title('Clustered Graph - Scalable SC with ' + str(i) + ' clusters')
  plt.xlabel('X-axis')
  plt.ylabel('Y-axis')
  plt.show()

print("printing avg balances:\n")
print(avg_balances)



"""# Plotting balances for different k's"""

k_values = list(range(len(avg_balances)))
k_values = k_values[2:]

avg_balances = avg_balances[2:]

print(avg_balances)
print(k_values)

plt.plot(k_values, avg_balances, marker='o', linestyle='-')
y_ticks = [i*0.25 for i in range(5)]
plt.yticks(y_ticks)
plt.xlabel("k")
plt.ylabel("Average Balance")
plt.title("Balance vs. k")

plt.grid(True)
plt.show()

