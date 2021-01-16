import numpy as np
import os
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.sparse import coo_matrix, bmat, kron, eye
from scipy.sparse.linalg import cg
from scipy.linalg import toeplitz

## Section b

img_path = os.path.join(os.path.dirname(__file__), 'imgs/centrosaurus.png')

im = Image.open(img_path).convert('LA')
im_gray = ImageOps.grayscale(im) # Convert to grayscale
im_gray = np.array(im_gray) # Convert to ndarray
im_gray = im_gray / 255
plt.figure()
plt.imshow(im_gray, cmap='gray')
plt.title('Original Image')

nx = im_gray.shape[0]
ny = im_gray.shape[1]

z0 = im_gray.T.reshape(1, nx*ny).squeeze() # Column stack the image
noise = np.random.normal(0,np.sqrt(0.01),nx*ny)

y = z0 + noise

Y = y.reshape((ny, nx)).T
plt.figure()
plt.imshow(Y, cmap='gray')
plt.title('Noised Image')
# plt.show()

## Section C

## Build tooploitz matrix and L

M = coo_matrix((nx, nx))
M.setdiag(np.ones(nx-1))
M.setdiag(-np.ones(nx-1)/8, k=1)
M.setdiag(-np.ones(nx-1)/8, k=-1)

K = coo_matrix((nx, nx))
K.setdiag(-np.ones(nx-1)/8)
K.setdiag(-np.ones(nx-1)/8, k=1)
K.setdiag(-np.ones(nx-1)/8, k=-1)

M = toeplitz(M).item()

MM = coo_matrix(np.eye(ny))

KK = coo_matrix((ny, ny))
KK.setdiag(np.ones(ny-1), k=1)
KK.setdiag(np.ones(ny-1), k=-1)

L = kron(KK, K) + kron(MM, M)
L.setdiag(L.diagonal()[:,np.newaxis]-L.sum(axis=1).A)

# Solver equation

gamma_list = [0.5, 1, 1.5, 2.5, 5, 7,  10]
fig = plt.figure()
N = len(gamma_list)
rows = 2
cols = int(np.ceil(N / rows))
gs = gridspec.GridSpec(rows, cols)

for idx, gamma in enumerate(gamma_list):
    A = eye(L.shape[0]) + gamma*L
    b = y
    x, convergence_inf = cg(A, b, tol=1e-4)
    x = np.clip(x, 0, 1)
    X = x.reshape((ny, nx)).T
    ax = fig.add_subplot(gs[idx])
    ax.imshow(X, cmap='gray')
    ax.set_title('Reconstructed image for ' +r"$\gamma=}$" + str(gamma))

plt.show()
print('tre')