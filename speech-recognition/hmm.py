# _________
# |         |
# |         0
# |        /|\
# |        / \
# |
# |

import numpy as np
import pickle
import vectors

from scipy.stats import multivariate_normal
from sklearn import mixture
from random import randint

class Hmm(object):
  def __init__(self, num_states, mfcc_size, label):
    self.num_states = num_states
    self.label = label
    self.next_prob = np.ones(num_states - 1)
    self.self_prob = np.zeros(num_states - 1)
    self.features = []
    self.mfcc_size = mfcc_size
    self.nodes = [] 
    for i in xrange(0, num_states):
      var = multivariate_normal(mean=np.zeros(mfcc_size), cov=np.diag(np.ones(mfcc_size)))
      self.nodes.append(var)

  def AppendMfccVectors(self, vector_arr):
    self.features.append(vectors.Normalize(vector_arr))

  def initTrain(self):
    num_features = len(self.features)
    curr_clusters = []
    for i in xrange(0, num_features):
      cl = []
      for j in xrange(0, self.num_states):
        cl.append(j)
      n = len(self.features[i])
      cluster_size = n // self.num_states 
      for j in xrange(0, n - self.num_states):
        cl.append(randint(0, self.num_states-1))
      cl.sort()
      curr_clusters.append(cl)

    cluster_cnt = [0] * self.num_states
    vector_cluster_map = {}
    for idx in xrange(0, self.num_states):
      vector_cluster_map[idx] = []
    
    for i in xrange(0, num_features):
      cluster_len = len(curr_clusters[i])
      for j in xrange(0, cluster_len):
        cluster_id = curr_clusters[i][j]
        cluster_cnt[cluster_id] += 1
        vector_cluster_map[cluster_id].append(self.features[i][j])

    new_next_prob = []
    for i in xrange(0, self.num_states - 1):
      new_next_prob.append(1. * num_features / cluster_cnt[i])

    self.next_prob = -np.log(new_next_prob)
    self.self_prob = np.concatenate([-np.log(1.0 - np.array(new_next_prob)), [0.]])

    for i in xrange(0, self.num_states):
      self.nodes[i] = multivariate_normal(mean=np.mean(vector_cluster_map[i], axis=0), cov=np.diag(np.ones(self.mfcc_size)))

    return curr_clusters

  def alignClusters(self, vector_arr, prev_path):
    centroids = []
    path = []
    path.extend(prev_path)

    for var in self.nodes:
      centroids.append(var.mean)

    vector_arr_len = len(vector_arr)
    for i in xrange(1, vector_arr_len - 1):
      c1 = centroids[path[i - 1]]
      c2 = centroids[path[i]]
      d1 = vectors.SquaredDistance(vector_arr[i], c1)
      d2 = vectors.SquaredDistance(vector_arr[i], c2)
      if path[i] == path[i + 1] and d1 < d2:
        path[i] = path[i - 1]

    for i in xrange(vector_arr_len - 2, 0, -1):
      c1 = centroids[path[i + 1]]
      c2 = centroids[path[i]]
      d1 = vectors.SquaredDistance(vector_arr[i], c1)
      d2 = vectors.SquaredDistance(vector_arr[i], c2)
      if path[i] == path[i - 1] and d1 < d2:
        path[i] = path[i + 1]

    return path

  def KMeans(self, max_iter = 2048):
    itr = 0
    prev_clusters = self.initTrain()
    while itr < max_iter:

      itr += 1

      num_features = len(self.features)
      curr_clusters = []

      for i in xrange(0, num_features):
        path = self.alignClusters(self.features[i], prev_clusters[i])
        curr_clusters.append(path)
      
      if (curr_clusters == prev_clusters):
        break
      prev_clusters = curr_clusters

      # racunanje novih verovatnoca
      cluster_cnt = [0] * self.num_states
      vector_cluster_map = {}
      for idx in xrange(0, self.num_states):
        vector_cluster_map[idx] = []

      for i in xrange(0, num_features):
        cluster_len = len(curr_clusters[i])
        for j in xrange(0, cluster_len):
          cluster_id = curr_clusters[i][j]
          cluster_cnt[cluster_id] += 1
          vector_cluster_map[cluster_id].append(self.features[i][j])

      new_next_prob = []
      for i in xrange(0, self.num_states - 1):
        new_next_prob.append(1. * num_features / cluster_cnt[i])

      self.next_prob = -np.log(new_next_prob)
      self.self_prob = np.concatenate([-np.log(1.0 - np.array(new_next_prob)), [0.]])

      for i in xrange(0, self.num_states):
        self.nodes[i] = multivariate_normal(mean=np.mean(vector_cluster_map[i], axis=0), cov=np.diag(np.ones(self.mfcc_size)))
    
    return prev_clusters

  def Viterby(self, vector_arr):
    if len(vector_arr[0]) != self.mfcc_size:
      return float('Inf')

    prev_prob = {}
    path = {}

    prev_prob[0] = 0
    for y in xrange(1, self.num_states):
      prev_prob[y] = float('Inf')
    
    vector_arr_len = len(vector_arr)
    for t in xrange(1, vector_arr_len):

      curr_prob = {}
      
      curr_prob[0] = prev_prob[0] + self.self_prob[0] - np.log(self.nodes[0].pdf(vector_arr[t])) 

      for y in xrange(1, self.num_states):
        p1 = prev_prob[y-1] + self.next_prob[y-1] - np.log(self.nodes[y].pdf(vector_arr[t]))
        p2 = prev_prob[y] + self.self_prob[y] - np.log(self.nodes[y].pdf(vector_arr[t]))
        curr_prob[y] = min(p1, p2)
      
      prev_prob = curr_prob

    return prev_prob[self.num_states - 1]
  
  def CalculateScore(self, vector_arr):
    return self.Viterby(vectors.Normalize(vector_arr))

  def WriteToFile(self, file_name):
    f = open(file_name, 'w')
    pickle.dump(self, f)
    f.close()

def FromFile(file_name):
  f = open(file_name, 'r')
  res = pickle.load(f)
  f.close()
  return res

# X = hmm(3, 1)
# X.AppendMfccVectors([[1],[2],[3]])
# X.KMeans()
# print X.CalculateScore([[2], [3]])
