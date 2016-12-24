import nltk
import re
import random

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

DATA_FILE = '/users/dragan/Documents/mms_domaci_1/newsCorpora.csv' #422k 
DATA_POINTS_NB = [1000, 5000, 10000, 25000, 50000, 100000]
DATA_POINTS_MAXENT = [1000, 5000, 10000, 25000, 50000, 100000]
DATA_POINTS_SVM = 100

def ReadFile(fname):
  with open(fname) as f:
    return f.readlines()

def ExtractFeatures(row, use_timestamp = True, use_publisher = True, use_publisher_url = True, use_url = True):
  row_parsed = row.split('\t')
  features = {}
  features['tittle'] = row_parsed[1]
  if use_timestamp:
    features['timestamp'] = row_parsed[7]
  if use_publisher:
    features['publisher'] = row_parsed[3]
  if use_publisher_url:
    features['publisher_url'] = row_parsed[6]
  if use_url:
    features['url'] = row_parsed[2]
  return (features, row_parsed[4])

def ExtractFeaturesBagOfWords(row, use_timestamp = True, use_publisher = True, use_publisher_url = True, use_url = True):
  row_parsed = row.split('\t')
  features = {}
  tittle = row_parsed[1].split()
  for word in tittle:
    features[word] = True
  if use_timestamp:
    features['timestamp'] = row_parsed[7]
  if use_publisher:
    features['publisher'] = row_parsed[3]
  if use_publisher_url:
    features['publisher_url'] = row_parsed[6]
  if use_url:
    features['url'] = row_parsed[2]
  return (features, row_parsed[4])

def GenerateData(n, return_label, use_bag_of_words = False, use_timestamp = True, use_publisher = True, use_publisher_url = True, use_url = True):
  data = ReadFile(DATA_FILE)
  data_size = len(data)
  feature_list = []
  for i in xrange(0, n):
    row = data[i]
    features = {}
    if use_bag_of_words:
       features = ExtractFeaturesBagOfWords(row, use_timestamp, use_publisher, use_publisher_url, use_url)
    else:
      features = ExtractFeatures(row, use_timestamp, use_publisher, use_publisher_url, use_url)
    if return_label:
      feature_list.append(features)
    else:
      feature_list.append(features[0])
  return feature_list

def NaiveBayesAccuracy(use_bag_of_words = True, use_timestamp = True, use_publisher = True, use_publisher_url = True, use_url = True):
  accuracy = []
  for size in DATA_POINTS_NB:
    data = GenerateData(size, True, use_bag_of_words, use_timestamp, use_publisher, use_publisher_url, use_url)
    train_size = int(len(data) * 0.9)
    classifier = nltk.NaiveBayesClassifier.train(data[0:train_size])
    accuracy.append(nltk.classify.util.accuracy(classifier, data[train_size:]))
  return accuracy

def TestAllNaiveBayesClassifiers(use_bag_of_words):
  fset_1 = NaiveBayesAccuracy(use_bag_of_words, 1, 1, 1, 1)
  fset_2 = NaiveBayesAccuracy(use_bag_of_words, 0, 1, 1, 1)
  fset_3 = NaiveBayesAccuracy(use_bag_of_words, 0, 1, 0, 1)
  fset_4 = NaiveBayesAccuracy(use_bag_of_words, 0, 1, 0, 0)
  fset_5 = NaiveBayesAccuracy(use_bag_of_words, 0, 0, 0, 0)
  sz = len(fset_1)
  for i in xrange(0, sz):
    print "%.2lf %.2lf %.2lf %.2lf %.2lf %d" % (fset_1[i], fset_2[i], fset_3[i], fset_4[i], fset_5[i], DATA_POINTS_NB[i])

def MaxEntAccuracy(use_bag_of_words, use_timestamp = True, use_publisher = True, use_publisher_url = True, use_url = True):
  accuracy = []
  for size in DATA_POINTS_MAXENT:
    data = GenerateData(size, True, use_bag_of_words, use_timestamp, use_publisher, use_publisher_url, use_url)
    train_size = int(len(data) * 0.9)
    classifier = nltk.MaxentClassifier.train(data[0:train_size], trace=0)
    accuracy.append(nltk.classify.util.accuracy(classifier, data[train_size:]))
  return accuracy

def TestAllMaxEntClassifiers(use_bag_of_words):
  fset_1 = MaxEntAccuracy(use_bag_of_words, 1, 1, 1, 1)
  fset_2 = MaxEntAccuracy(use_bag_of_words, 0, 1, 1, 1)
  fset_3 = MaxEntAccuracy(use_bag_of_words, 0, 1, 0, 1)
  fset_4 = MaxEntAccuracy(use_bag_of_words, 0, 1, 0, 0)
  fset_5 = MaxEntAccuracy(use_bag_of_words, 0, 0, 0, 0)
  sz = len(fset_1)
  for i in xrange(0, sz):
    print "%.2lf %.2lf %.2lf %.2lf %.2lf %d" % (fset_1[i], fset_2[i], fset_3[i], fset_4[i], fset_5[i], DATA_POINTS_MAXENT[i])

def ShowMostInformativeFeatures(n): #fset_4 - 100k
  data = GenerateData(100000, True, True, 0, 1, 0, 0)
  train_size = int(len(data) * 0.9)
  classifier = nltk.NaiveBayesClassifier.train(data[0:train_size])
  classifier.show_most_informative_features(n)

def ParseMostInformativeFeatures(fname):
  with open(fname) as f:
    contents = f.readlines()
    publishers = []
    words = []
    for line in contents:
      line_parsed = line.strip().split()
      if line_parsed[0] == 'publisher':
        publisher = re.search("\'.*\'", line.strip())
        if not publisher:
           publisher = re.search("\".*\"", line.strip())
        publisher = publisher.group(0)
        rest = line.strip()[line.strip().find(publisher) + len(publisher):]
        rest_split = rest.strip().split()
        publishers.append("%s\t%s\t%s\t%s\t%s" % (publisher[1:-1], rest_split[0], rest_split[2], rest_split[4], rest_split[6]))
      else:
        words.append("%s %s %s %s %s" % (line_parsed[0], line_parsed[3], line_parsed[5], line_parsed[7], line_parsed[9]))
    
    for w in words:
      print w
    print "----------------SEPERATOR-------"
    for p in publishers:
      print p

# Zbog malog broja
def GenerateDataSVM(n, use_bag_of_words = False, use_timestamp = True, use_publisher = True, use_publisher_url = True, use_url = True):
  data = ReadFile(DATA_FILE)
  data_size = len(data)
  feature_list = []
  for i in xrange(0, data_size):
    row = data[i]
    features = {}
    if use_bag_of_words:
       features = ExtractFeaturesBagOfWords(row, use_timestamp, use_publisher, use_publisher_url, use_url)
    else:
      features = ExtractFeatures(row, use_timestamp, use_publisher, use_publisher_url, use_url)
    feature_list.append(features)

  def __extractClassFeatures(feature_list, count, label):
    class_features_train = []
    class_features_test = []
    ten_percent_count = int(count * 0.2) + 1
    for f in feature_list:
      if f[1] == label:
        count -= 1
        if count < ten_percent_count:
          class_features_test.append(f)
        else:
          class_features_train.append(f)
        if count == 0:
          break
    return (class_features_train, class_features_test)

  train_data = []
  train_data.extend(__extractClassFeatures(feature_list, n // 4, 'b')[0])
  train_data.extend(__extractClassFeatures(feature_list, n // 4, 'e')[0])
  train_data.extend(__extractClassFeatures(feature_list, n // 4, 'm')[0])
  train_data.extend(__extractClassFeatures(feature_list, n // 4, 't')[0])
  
  test_data = []
  test_data.extend(__extractClassFeatures(feature_list, n // 4, 'b')[1])
  test_data.extend(__extractClassFeatures(feature_list, n // 4, 'e')[1])
  test_data.extend(__extractClassFeatures(feature_list, n // 4, 'm')[1])
  test_data.extend(__extractClassFeatures(feature_list, n // 4, 't')[1])
  
  return (train_data, test_data)

def SVMAccuracy(svm_type, use_bag_of_words, use_timestamp = True, use_publisher = True, use_publisher_url = True, use_url = True):
  train_data, test_data = GenerateDataSVM(DATA_POINTS_SVM, use_bag_of_words, use_timestamp, use_publisher, use_publisher_url, use_url)
  classifier = SklearnClassifier(svm_type)
  classifier.train(train_data)
  return nltk.classify.accuracy(classifier, test_data)

def TestAllSVMAccuracy(svm_type, use_bag_of_words):
  fset_1 = SVMAccuracy(svm_type, use_bag_of_words, 1, 1, 1, 1)
  fset_2 = SVMAccuracy(svm_type, use_bag_of_words, 0, 1, 1, 1)
  fset_3 = SVMAccuracy(svm_type, use_bag_of_words, 0, 1, 0, 1)
  fset_4 = SVMAccuracy(svm_type, use_bag_of_words, 0, 1, 0, 0)
  fset_5 = SVMAccuracy(svm_type, use_bag_of_words, 0, 0, 0, 0)
  print "%.2lf %.2lf %.2lf %.2lf %.2lf" % (fset_1, fset_2, fset_3, fset_4, fset_5)

def Main():
  # TestAllSVMAccuracy(SVC(), True)
  # TestAllSVMAccuracy(SVC(), False)
  # TestAllSVMAccuracy(LinearSVC(), True)
  # TestAllSVMAccuracy(LinearSVC(), False)
  # TestAllSVMAccuracy(NuSVC(), True)
  # TestAllSVMAccuracy(NuSVC(), False)
  # TestAllNaiveBayesClassifiers(True)
  # TestAllMaxEntClassifiers(True)
  # TestAllMaxEntClassifiers(False)
  # ShowMostInformativeFeatures(2100)
  # ParseMostInformativeFeatures("/users/dragan/Downloads/import.txt")

Main()
