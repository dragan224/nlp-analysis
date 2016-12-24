import database
import vectors
import random

from random import randint
from hmm import Hmm
from datetime import datetime
random.seed(datetime.now())

hmms = database.LoadAllHmms()

def TestSingleInstance(label, base_vector_arr, gauss_sizes):
  best_label = 'OVO NE POSTOJI U BAZI 10010101011000%!!!!!'
  best_score = float('Inf')
  generated_vectors = []
  for i in xrange(0, len(base_vector_arr)):
      generated_vectors.append(base_vector_arr[i])
      genvec = vectors.GenerateGaussian(base_vector_arr[i], gauss_sizes[i])
      generated_vectors.extend(genvec)

  # print "Trazim %s" % label
  for h in hmms:
    score = h.CalculateScore(generated_vectors)
    # print "%lf %s" % (score, h.label)
    if score < best_score:
      best_score = score
      best_label = h.label

  # print ''
  return int(best_label == label)

def CalcTotal():
  total = 0
  for i in xrange(0, 5):
    total += TestSingleInstance('jabuka', [[1, 5], [1, 10], [1, 30]], [randint(10, 20), randint(20, 30), randint(30, 40)])
    total += TestSingleInstance('kruska', [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8]], [randint(10, 20), randint(20, 30), randint(30, 40), randint(10, 20), randint(20, 30), randint(30, 40), randint(10, 20), randint(20, 30)])
    total += TestSingleInstance('test', [[1, 5], [2, 4]], [randint(10, 20), randint(20, 30), randint(30, 40)])
    total += TestSingleInstance('raf', [[25, 20]], [randint(10, 20)])
    total += TestSingleInstance('blabla', [[1, 2], [1, 2], [5, 8], [2, 2], [6, 7]], [randint(10, 20), randint(20, 30), randint(30, 40), randint(20, 30), randint(30, 40)])
  return total

tot_sum = 0.
for i in xrange(0, 10):
  tot_sum += CalcTotal() / 25.
print "Prosecna preciznost %.5lf procenta" % ((tot_sum / 10.) * 100.)
