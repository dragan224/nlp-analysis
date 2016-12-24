import database
import vectors
import random

from random import randint
from hmm import Hmm
from datetime import datetime
random.seed(datetime.now())

def FillDB(num_states, vector_size, label, base_vector_arr, gauss_sizes):
  h_old = database.LoadSpecificHmm(label)
  generated_vectors = []
  for i in xrange(0, len(base_vector_arr)):
      generated_vectors.append(base_vector_arr[i])
      genvec = vectors.GenerateGaussian(base_vector_arr[i], gauss_sizes[i])
      generated_vectors.extend(genvec)

  if h_old is None:
    h_new = Hmm(num_states, vector_size, label)
    h_new.AppendMfccVectors(generated_vectors)
    h_new.KMeans()
    database.SaveHmm(h_new)
  else:
    h_old.AppendMfccVectors(generated_vectors)
    h_old.KMeans()
    database.SaveHmm(h_old)


for i in xrange(0, 10):
  FillDB(3, 2, 'jabuka', [[1, 5], [1, 10], [1, 30]], [randint(10, 20), randint(20, 30), randint(30, 40)])
  FillDB(4, 2, 'kruska', [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8]], [randint(10, 20), randint(20, 30), randint(30, 40), randint(10, 20), randint(20, 30), randint(30, 40), randint(10, 20), randint(20, 30)])
  FillDB(2, 2, 'test', [[1, 5], [2, 4]], [randint(10, 20), randint(20, 30), randint(30, 40)])
  FillDB(10, 2, 'raf', [[25, 20]], [randint(10, 25)])
  FillDB(15, 2, 'blabla', [[1, 2], [1, 2], [5, 8], [2, 2], [6, 7]], [randint(10, 20), randint(20, 30), randint(30, 40), randint(20, 30), randint(30, 40)])

