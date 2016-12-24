import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from sklearn.preprocessing import scale

def SquaredDistance(A, B):
  sum_dist = 0.
  lenA = len(A)
  for i in xrange(0, lenA):
    sum_dist += abs(A[i] - B[i]) * abs(A[i] - B[i])
  return sum_dist

def Normalize(vector_arr):
  return scale(vector_arr)

def GenerateGaussian(vector, num_vectors):
  res = []
  var = multivariate_normal(mean=vector, cov=np.diag(np.ones(len(vector))))
  vector_sz = len(vector)
  for i in xrange(0, num_vectors):
    if vector_sz == 1:
      res.append([var.rvs()])
    else:
      res.append(var.rvs().tolist())
  return res

def ColumnWiseLimits(vector_arr, d):
  min_val = vector_arr[0][d]
  max_val = vector_arr[0][d]
  for v in vector_arr:
    max_val = max(max_val, v[d])
    min_val = min(min_val, v[d])
  return (min_val, max_val)

def Draw2D(vector_arr):
  n = len(vector_arr)
  X,Y,U,V = zip(*zeroPadding(vector_arr))
  plt.figure()
  ax = plt.gca()
  ax.quiver(X,Y,U,V,angles='xy',scale_units='xy',scale=1)
  (x_min, x_max) = ColumnWiseLimits(vector_arr, 0)
  x_min = min(x_min, 1) - 1
  (y_min, y_max) = ColumnWiseLimits(vector_arr, 1)
  y_min = min(y_min, 1) - 1
  ax.set_xlim([x_min, x_max])
  ax.set_ylim([y_min, y_max])
  plt.draw()
  plt.show()

def zeroPadding(vector_arr):
  res = []
  for v in vector_arr:
    res.append([0, 0] + v)
  return res
