import numpy as np
import os
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import linalg

from utils.read_off import read_off

## 3D Point Cloud Load

# off_name = 'toilet_0107'
off_name = 'flower_pot_0157'
# off_name = 'tent_0165'
# off_name = 'guitar_0189'

v, f = read_off('resources/' + off_name + '.off')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(v[:,0], v[:,1], v[:,2], marker='o', s=1)
ax.set_title('Point Cloud of ' + off_name)
# plt.show()


## Add noise to Point Cloud

nv = v.shape[0]
dists = cdist(v, v)
s_ = 0.3*np.median(dists.T.reshape(1, nv**2).squeeze())
noise = np.random.normal(scale=0.1*s_, size=v.shape)
X = v + noise

## Plot Noised Point Cloud

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], marker='o', s=1)
ax.set_title('Noised Point Cloud of ' + off_name)
# plt.show()

## Build affinity matrix

r = s_
sigma = s_

dists_noise = cdist(X, X)
close_dists = (dists_noise <= r).astype('float')
W = sparse.lil_matrix(close_dists)
# W.setdiag(np.zeros(nv))
rows,cols = W.nonzero()
Distances_adj = np.exp(-(1 / (2*sigma**2))* dists_noise[rows,cols]**2)
W[rows, cols] = Distances_adj

## Normalized Graph Laplacian

D_sqrt = sparse.diags(np.squeeze(np.asarray(W.sum(axis=1))**(-0.5)))
N = sparse.identity(nv) - D_sqrt @ W @ D_sqrt

# Compute eigen decomposition

# N_eigVals, N_eigVecs = linalg.eigs(N, k=N.shape[0]-2)
N_eigVals, N_eigVecs = np.linalg.eig(N.A)
N_eigVals_original, N_eigVecs_original = N_eigVals, N_eigVecs
# Sort the eigen values with associated eigenvectors
idx_pn = N_eigVals.argsort()[::1] 
N_eigVals = N_eigVals[idx_pn] 
N_eigVecs = N_eigVecs[:, idx_pn]

# Plot the sorted eigenvalues

fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlabel('Statistical order')
plt.ylabel(r'$\lambda$')
ax.set_title('Graph Laplacian Sorted Eigenvalues')

plt.plot(np.arange(len(N_eigVals))+1, N_eigVals)

# Plot noised point clouds colored by eigenvectors
eig_vecs_idx = [1, 2, 4, 10]

fig = plt.figure()
fig.suptitle('Noisy PointClouds Colored By EigenVectors ')

for idx, vec_idx in enumerate(eig_vecs_idx):
    ax = fig.add_subplot(2, len(eig_vecs_idx)/2, idx+1, projection='3d')
    color = np.round(N_eigVecs[:, vec_idx-1], decimals=4)
    sc = ax.scatter(X[:,0], X[:,1], X[:,2], marker='o', s=2, c=color)
    fig.colorbar(sc, ax=ax)
    ax.set_title('k=' + str(vec_idx))

# plt.show()

## Create low-pass graph filter

## Plot low pass graph for various taus

max_eig = np.max(N_eigVals)
tau_list = [3, 5, 10]
fig = plt.figure(figsize=(15,5))
fig.suptitle('Low Pass Graph Filter for Various ' +r'$\tau$')

for idx, tau in enumerate(tau_list):
    h = np.exp(-(tau/max_eig)*N_eigVals)
    ax = fig.add_subplot(1, len(tau_list),idx+1)
    ax.plot(N_eigVals, h)
    plt.xlabel('N EigenValues')
    plt.ylabel(r'$h=exp({-\frac{\tau x}{\lambda_{max}}})$')
    ax.set_title(r'$\tau=$' + str(tau))

# plt.show()

## Denoise the 3D pointcloud for different taus

## Plot low pass graph for various taus

n = 10

max_eig = np.max(N_eigVals)
tau_list = [3, 5, 10]
fig = plt.figure(figsize=(15,5))
fig.suptitle('Apply Denoising on PointCloud for various ' +r'$\tau$')
eigVecs_mat = sparse.csr_matrix(N_eigVecs)
X_sparse = sparse.csr_matrix(X)
for idx, tau in enumerate(tau_list):
    h = np.exp(-(tau/max_eig)*N_eigVals)
    eigVals_mat = sparse.csr_matrix(np.diag(h))
    denoised_X = eigVecs_mat @ eigVals_mat @ eigVecs_mat.T @ X_sparse.A
    ax = fig.add_subplot(1, len(tau_list), idx+1, projection='3d')
    ax.scatter(denoised_X[:,0], denoised_X[:,1], denoised_X[:,2], marker='o', s=2)
    ax.set_title(r'$\tau=$' + str(tau))

plt.show()
print('tre')