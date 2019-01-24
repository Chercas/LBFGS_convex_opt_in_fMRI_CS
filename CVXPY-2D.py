import numpy as np
#import matplotlib as mpl
import matplotlib.pyplot as plt
#import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import cvxpy as cvx
import imageio as im

def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0))

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0))

def rgb_to_gray(image):
    return np.dot(image[...,:3], [0.299, 0.587, 0.144])

Xorig = im.imread('escher_waterfall.jpeg')
print(rgb_to_gray(Xorig).shape)
X = spimg.zoom(rgb_to_gray(Xorig), 0.1)
ny,nx = X.shape

k = round(nx * ny * 0.5) # 50% sample
ri = np.random.choice(nx * ny, k, replace=False) # random sample of indices
b = X.T.flat[ri]
#b = np.expand_dims(b, axis=1)

# create dct matrix operator using kron (memory errors for large ny*nx)
A = np.kron(
    spfft.idct(np.identity(nx), norm='ortho', axis=0),
    spfft.idct(np.identity(ny), norm='ortho', axis=0)
    )
A = A[ri,:] # same as phi times kron

# do L1 optimization
vx = cvx.Variable(nx * ny)
objective = cvx.Minimize(cvx.norm(vx, 1))
constraints = [A*vx == b]
prob = cvx.Problem(objective, constraints)
result = prob.solve(verbose=True)
Xat2 = np.array(vx.value).squeeze()

Xat = Xat2.reshape(nx, ny).T # stack columns
Xa = idct2(Xat)

mask = np.zeros(X.shape)
mask.T.flat[ri] = 255
Xm = 255 * np.ones(X.shape)
Xm.T.flat[ri] = X.T.flat[ri]

plt.imshow(Xorig)
plt.show()

plt.imshow(X)
plt.show()

plt.imshow(Xa.T)
plt.show()

plt.imshow(Xm)
plt.show()

'''
n = 5000
t = np.linspace(0, 1/2, n)
y = np.sin(1394 * np.pi * t) + np.sin(3266 * np.pi * t)
yt = spfft.dct(y, norm='ortho')

m = 500# 2.5% sample
ri = np.random.choice(n, m, replace=False) # random sample of indices
ri.sort() # sorting not strictly necessary, but convenient for plotting
t2 = t[ri]
y2 = y[ri]

plt.plot(t, y, t2, y2, 'ro', markersize=3)
plt.axis([0.2, 0.5, -3.1, 3.1])
plt.show()

A = spfft.idct(np.identity(n), norm='ortho', axis=0)
A = A[ri]

# do L1 optimization
vx = cvx.Variable(n)
objective = cvx.Minimize(cvx.norm(vx, 1))
constraints = [A*vx == y2]
prob = cvx.Problem(objective, constraints)
result = prob.solve(verbose=True)

x = np.array(vx.value)
x = np.squeeze(x)
sig = spfft.idct(x, norm='ortho', axis=0)

plt.plot(t, sig, 'r', t, y, markersize=3)
plt.axis([1/5.2, 1/5, -2.2, 2.2])
plt.show()

plt.plot(t, yt, 'r', markersize=3)
plt.axis([0, 1/4, -35, 45])
plt.show()
plt.plot(t, x, 'b', markersize=3)
plt.axis([0, 1/4, -35, 45])
plt.show()
'''