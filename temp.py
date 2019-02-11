import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
#import cvxpy as cvx
#import imageio 
#import visvis as vv
#from pylbfgs import owlqn

import sys

sys.executable

def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def evaluate(x, g, step):
    """An in-memory evaluation callback."""

    # we want to return two things: 
    # (1) the norm squared of the residuals, sum((Ax-b).^2), and
    # (2) the gradient 2*A'(Ax-b)

    # expand x columns-first
    x2 = x.reshape((nx, ny)).T

    # Ax is just the inverse 2D dct of x2
    Ax2 = idct2(x2)

    # stack columns and extract samples
    Ax = Ax2.T.flat[ri].reshape(b.shape)

    # calculate the residual Ax-b and its 2-norm squared
    Axb = Ax - b
    fx = np.sum(np.power(Axb, 2))

    # project residual vector (k x 1) onto blank image (ny x nx)
    Axb2 = np.zeros(x2.shape)
    Axb2.T.flat[ri] = Axb # fill columns-first

    # A'(Ax-b) is just the 2D dct of Axb2
    AtAxb2 = 2 * dct2(Axb2)
    AtAxb = AtAxb2.T.reshape(x.shape) # stack columns

    # copy over the gradient vector
    np.copyto(g, AtAxb)

    return fx

# fractions of the scaled image to randomly sample at
sample_sizes = (0.1, 0.01)

# read original image
Xorig = spimg.imread('escher_waterfall.jpeg')
ny,nx,nchan = Xorig.shape

# for each sample size
Z = [np.zeros(Xorig.shape, dtype='uint8') for s in sample_sizes]
masks = [np.zeros(Xorig.shape, dtype='uint8') for s in sample_sizes]
for i,s in enumerate(sample_sizes):

    # create random sampling index vector
    k = round(nx * ny * s)
    ri = np.random.choice(nx * ny, k, replace=False) # random sample of indices

    # for each color channel
    for j in range(nchan):

        # extract channel
        X = Xorig[:,:,j].squeeze()

        # create images of mask (for visualization)
        Xm = 255 * np.ones(X.shape)
        Xm.T.flat[ri] = X.T.flat[ri]
        masks[i][:,:,j] = Xm

        # take random samples of image, store them in a vector b
        b = X.T.flat[ri].astype(float)

     # perform the L1 minimization in memory
        Xat2 = spopt.fmin_bfgs(nx*ny, evaluate, None, 5)

        # transform the output back into the spatial domain
        Xat = Xat2.reshape(nx, ny).T # stack columns
        Xa = idct2(Xat)
        Z[i][:,:,j] = Xa.astype('uint8')


'''
def dct2(x):
    return spfft.dct(spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

def idct2(x):
    return spfft.idct(spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)

# read original image and downsize for speed
Xorig = spimg.imread('escher_waterfall.jpeg', flatten=True, mode='L') # read in grayscale
X = spimg.zoom(Xorig, 0.04)
ny,nx = X.shape

#plt.imshow(Xorig_RGB_to_GS, cmap='gray')
#plt.imshow(X, cmap='gray')

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

print(A.shape)

vx = cvx.Variable(nx * ny)
objective = cvx.Minimize(cvx.norm(vx, 1))
constraints = [A*vx == b]
'''
'''
prob = cvx.Problem(objective, constraints)
result = prob.solve(verbose=True)
Xat2 = np.array(vx.value).squeeze()

Xat = Xat2.reshape(nx, ny).T
Xa = idct2(Xat)

if not np.allclose(X.T.flat[ri], Xa.T.flat[ri]):
    print('Warning! Achtung! Values at sample indices don\'t match original.')

mask = np.zeros(X.shape)
mask.T.flat[ri] = 255
Xm = 255 * np.ones(X.shape)
Xm.T.flat[ri] = X.T.flat[ri]

#plt.imshow(Xm.T.flat[ri], cmap='gray')
'''
'''
x = np.sort(np.random.uniform(0, 10, 15))
y = 3 + 0.2*x + 0.1*np.random.randn(len(x))

l1_fit = lambda x0, x, y: np.sum(np.abs(x0[0] * x + x0[1] -y))
xopt1 = spopt.fmin(func=l1_fit, x0=[1, 1], args=(x, y))
fit_1 = xopt1[0] * x + xopt1[1]
l2_fit = lambda x0, x, y: np.sum(np.power(x0[0] * x + x0[1] -y, 2))
xopt2 = spopt.fmin(func=l2_fit, x0=[1, 1], args=(x, y))
fit_2 = xopt2[0] * x + xopt2[1]

plt.plot(x, fit_2)
plt.plot(x, fit_1)
plt.plot(x, y, 'ro')
plt.show()

y_shift = y
y_shift[0] -=3
y_shift[14] += 3

l1_fit_shift = lambda x0, x, y_shift: np.sum(np.abs(x0[0] * x + x0[1] - y))
xopt1_shift = spopt.fmin(func=l1_fit_shift, x0 = [1, 1], args=(x, y_shift))
fit_1_shift = xopt1_shift[0] * x + xopt1_shift[1]
l2_fit_shift = lambda x0, x, y_shift: np.sum(np.power(x0[0] * x + x0[1] -y, 2))
xopt2_shift = spopt.fmin(func=l2_fit_shift, x0 = [1, 1], args=(x, y_shift))
fit_2_shift = xopt2_shift[0] * x + xopt2_shift[1]

plt.plot(x, y_shift, 'ro', x, fit_1_shift, x, fit_2_shift)
#plt.show()

n = 5000
t = np.linspace(0, 1/8, n)
y = np.sin(1394 * np.pi * t) + np.sin(3266 * np.pi * t)
yt = spfft.dct(y ,norm='ortho')


plt.plot(t, y)
plt.show()
plt.plot(t*50000, yt)
plt.axis([0, 1000, -40, 25])
plt.show()
plt.plot(t, y)
plt.axis([0, 0.02, -2.5, 2.5])
plt.show()

m = 500
ri = np.random.choice(n, m, replace=False)
ri.sort()
t2 = t[ri]
y2 = y[ri]


plt.plot(t, y)
plt.axis([0, 0.02, -2.5, 2.5])
plt.plot(t2, y2, 'ro')
plt.show()
plt.plot(t2, y2)
plt.axis([0, 0.02, -2.5, 2.5])
plt.show()

# create idct matrix operator
A = spfft.idct(np.identity(n), norm='ortho', axis=0)
A = A[ri]

# do L1 optimization
vx = cvx.Variable(n)
objective = cvx.Minimize(cvx.norm(vx, 1))
constraints = [A*vx == y2]
prob = cvx.Problem(objective, constraints)
result = prob.solve(verbose=True)

# Singnal recon
x = np.array(vx.value)
x = np.squeeze(x)
rec_sig = spfft.idct(x, norm='ortho', axis=0)

plt.plot(t, y)
plt.axis([0.02, 0.04, -2.5, 2.5])
plt.plot(t, rec_sig, 'ro', markersize = 3)
plt.show()
plt.plot(t, y)
plt.axis([0.02, 0.04, -2.5, 2.5])
plt.plot(t2, y2, 'ro', markersize=3)
plt.show()
plt.plot(t*50000, x, 'r')
plt.axis([0, 900, -40, 25])
plt.show()
plt.plot(t*50000, yt)
plt.axis([0, 900, -40, 25])
plt.show()
'''
