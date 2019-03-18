from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# This uses python3

import numpy as np
import csv
import copy

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from sklearn.model_selection import train_test_split
from parameters2D import start_f, end_f, label_column
#from imblearn.over_sampling import RandomOverSampler

class DataSet(object):
  """ Dataset structure"""
  def __init__(self,
               features,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """

    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert features.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (features.shape, labels.shape))
      self._num_examples = features.shape[0]

    self._features = features
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

    labels_probas = []
    for label in self._labels:
        if label == 0:
            labels_probas.append([1,0])
        if label == 1:
            labels_probas.append([0,1])

    self._labels_probas = labels_probas


def load_triangle_data(filepath,filepath2, normalised=False):
  dtype = dtypes.float32 # this doesn't matter
  reshape=True
  my_data = np.genfromtxt(filepath)
  print("did i enter this function?")
  print("triangle data shape:")
  print(my_data.shape)
  my_data = normalise_data_triangle(my_data)
  print(my_data.shape)

  my_data2 = np.genfromtxt(filepath2,delimiter=';')
  print("old data shape:")
  my_data2 = normalise_data(my_data2)
  print(my_data2.shape)


  nTot,_ = my_data.shape

  np.random.shuffle(my_data)
  np.random.shuffle(my_data2)
  my_data2 = my_data2[0:nTot,:]

  #proportion of old dataset to take
  lamb = 0.0

  #nTarget,_ = my_data.shape
  #nSource,_ = my_data2.shape
  nTarget = lamb*nTot
  nSource = (1-lamb)*nTot

  t1 = my_data[0:nTarget,0:23]
  t2 = my_data2[0:nSource,0:23]

  X = np.vstack((t1,t2))
  print('Stacked:')
  print(X.shape)
  Y = np.hstack((my_data[0:nTarget,23],my_data2[0:nSource,23]))

  train_data, test_data, train_labels, test_labels = train_test_split(X,Y, test_size=0.5, random_state=42,stratify=Y)

  # From test data, split again to get validation training set
  test_data, validation_data, test_labels, validation_labels = train_test_split(test_data, test_labels, test_size=0.50, random_state=42,stratify=test_labels)

  # generate the Dataset objects
  train = DataSet(train_data, train_labels, dtype=dtype, reshape=reshape)
  validation = DataSet(validation_data, validation_labels, dtype=dtype,reshape=reshape)
  test = DataSet(test_data, test_labels, dtype=dtype, reshape=reshape)

  return base.Datasets(train=train, validation=validation, test=test)



def load_rd_quads(target_filepath,source_filepath, alpha = 0.0, normalised=False):
  dtype = dtypes.float32 # this doesn't matter
  reshape=True
  my_data = np.genfromtxt(target_filepath)
  my_data = augment_data_normalised(my_data)
  print("rd quad data shape:")
  print(my_data.shape)

  my_data2 = np.genfromtxt(source_filepath,delimiter=';')
  print("old data shape:")
  my_data2 = normalise_data(my_data2)
  print(my_data2.shape)

  nTot,_ = my_data.shape

  np.random.shuffle(my_data)
  np.random.shuffle(my_data2)
  my_data2 = my_data2[0:nTot,:]

  #proportion of old dataset to take
  lamb = alpha

  nTarget = lamb*nTot
  nSource = (1-lamb)*nTot

  t1 = my_data[0:nTarget,0:23]
  t2 = my_data2[0:nSource,0:23]

  X = np.vstack((t1,t2))
  print('Stacked:')
  print(X.shape)
  Y = np.hstack((my_data[0:nTarget,23],my_data2[0:nSource,23]))

  train_data, test_data, train_labels, test_labels = train_test_split(X,Y, test_size=0.5, random_state=42,stratify=Y)

  # From test data, split again to get validation training set
  test_data, validation_data, test_labels, validation_labels = train_test_split(test_data, test_labels, test_size=0.50, random_state=42,stratify=test_labels)

  # generate the Dataset objects
  train = DataSet(train_data, train_labels, dtype=dtype, reshape=reshape)
  validation = DataSet(validation_data, validation_labels, dtype=dtype,reshape=reshape)
  test = DataSet(test_data, test_labels, dtype=dtype, reshape=reshape)

  return base.Datasets(train=train, validation=validation, test=test)


def normalise_data_triangle(text_file):
  #dataset.append([xv,h, u_c, u_m,u_p, du, du_m, du_p, u_f_p, u_f_m, u_m_f_p, u_p_f_m, u_max, u_min, label_hio])
  #np.random.RandomState(seed=1)
  normalised = []
  for entry in text_file:
    u_max = entry[15] #np.max([entry[2],entry[3],entry[4],entry[7],entry[8]])
    u_min = entry[16] #np.min([entry[2],entry[3],entry[4],entry[7],entry[8]])

    if (u_max == u_min):
        thrs = np.random.uniform(0,1)
        if thrs > 0.9:
            normalised.append(entry)
        else:
            continue
    else:
        normalised.append(entry)

  return np.array(normalised)


def generate_data_sets(filepath,normalised=False):
  """ returns train, test and validation data"""
  dtype = dtypes.float32 # this doesn't matter
  reshape=True

  print('Data is coming from: %s' %(filepath))

  # Load data
  my_data = np.genfromtxt(filepath, delimiter=';')
  # first column gives the x coordinates

  # HEADERS: xv,h,u_m,u_c, u_p, du_m, du_p, u_f_p, u_f_m, label

  # sample from data in a "balanced manner"
  # test_split( features, label (0/1), how to split(her 40% test, 60% train), random state
  #  data  trash 1,2,3,4,5,6,7 useful, label
  #              [1:8] 9

  #if normalised == True:
  # pre_norm = my_data[1,:]
  # my_data = normalise_data(my_data)
  # pos_norm = my_data[1,:]
  # print('Normalised data.')
  # print(pre_norm, pos_norm)
  #else:
  #pass

  # expand dataset
  #my_data = normalise_data(my_data)
  print('exited loading data')
  print( my_data.shape )
  my_data = augment_data(my_data)
  print( my_data.shape )
  # oversample the positive label
  #X_resampled, y_resampled = boost_dataset(my_data[:,start_f:end_f], my_data[:,label_column]) #ros.fit_sample(my_data[:,start_f:end_f], my_data[:,label_column])
  X_resampled = my_data[:,start_f:end_f]
  #print(X_resampled.shape)
  y_resampled =  my_data[:,label_column]
  print(np.sum(my_data[:,label_column])/len(my_data[:,label_column]))

  # Get training and test data
  train_data, test_data, train_labels, test_labels = train_test_split(X_resampled,y_resampled, test_size=0.5, random_state=42,stratify=y_resampled)

  # From test data, split again to get validation training set
  test_data, validation_data, test_labels, validation_labels = train_test_split(test_data, test_labels, test_size=0.50, random_state=42,stratify=test_labels)

  # generate the Dataset objects
  train = DataSet(train_data, train_labels, dtype=dtype, reshape=reshape)
  validation = DataSet(validation_data, validation_labels, dtype=dtype,reshape=reshape)
  test = DataSet(test_data, test_labels, dtype=dtype, reshape=reshape)

  return base.Datasets(train=train, validation=validation, test=test)

def normalise_data(text_file):
  #dataset.append([xv,h, u_c, u_m,u_p, du, du_m, du_p, u_f_p, u_f_m, u_m_f_p, u_p_f_m, u_max, u_min, label_hio])
  #np.random.RandomState(seed=1)
  normalised = []
  for entry in text_file:
    u_max = entry[15] #np.max([entry[2],entry[3],entry[4],entry[7],entry[8]])
    u_min = entry[16] #np.min([entry[2],entry[3],entry[4],entry[7],entry[8]])

    dx = entry[0]
    dy = entry[1]

    u_c = map2(entry[2],u_max,u_min)
    u_l = map2(entry[3],u_max,u_min)
    u_r = map2(entry[4],u_max,u_min)
    u_t = map2(entry[5],u_max,u_min)
    u_b = map2(entry[6],u_max,u_min)

    du_dx = (u_r - u_l)/(2.)#.*h) #map2(entry[5],maxval,minval)
    du_dy = (u_t-u_b)/(2.)

    du_dx_r = (u_r - u_c)
    du_dx_l = (u_c - u_l)
    du_dy_t = (u_t - u_c)
    du_dy_b = (u_c - u_b)

    uc_f_r =  map2(entry[7],u_max,u_min)
    uc_f_l = map2(entry[8],u_max,u_min)
    uc_f_t = map2(entry[9],u_max,u_min)
    uc_f_b = map2(entry[10],u_max,u_min)
    ul_f_r = map2(entry[11],u_max,u_min)
    ur_f_l = map2(entry[12],u_max,u_min)
    ut_f_b = map2(entry[13],u_max,u_min)
    ub_f_t = map2(entry[14],u_max,u_min)

    label_hio = entry[17]
    #normal = [xv,h, u_c, u_m,u_p, du, du_m, du_p, u_f_p, u_f_m, u_m_f_p, u_p_f_m, u_max, u_min, label_hio]
    normal = [dx, dy, u_c, u_l, u_r, u_t, u_b, uc_f_r, uc_f_l, uc_f_t, uc_f_b, ul_f_r, ur_f_l, ut_f_b, ub_f_t, u_max, u_min, du_dx, du_dy, du_dx_r, du_dx_l, du_dy_b, du_dy_t, label_hio]

    if (u_max == u_min):
        thrs = np.random.uniform(0,1)
        if thrs > 0.9:
            normalised.append(normal)
        else:
            continue
    else:
        normalised.append(normal)

  return np.array(normalised)


def augment_data_normalised(text_file):
   #print('entered augmented data')
   #dataset.append([xv,h, u_c, u_m,u_p, du, du_m, du_p, u_f_p, u_f_m, u_m_f_p, u_p_f_m, u_max, u_min, label_hio])
   #np.random.RandomState(seed=1)
   normalised = []

   permutation  = [[1,2,3,0],[2,3,0,1],[3,0,1,2]]


   #  ! define a
   #  permut(:,1) = (/ 1,2,3,4 /)
   #permut(:,2) = (/ 2,3,4,1 /)
   #permut(:,3) = (/ 3,4,1,2 /)
   #permut(:,4) = (/ 4,1,2,3 /)

   for entry in text_file:
         u_max = entry[15] #np.max([entry[2],entry[3],entry[4],entry[7],entry[8]])
         u_min = entry[16] #np.min([entry[2],entry[3],entry[4],entry[7],entry[8]])

         dx = entry[0]
         dy = entry[1]
         fudge_label = 0
         if ( abs(u_max - u_min) < 0.5*sqrt(dx*dy)*(abs(u_max)+abs(u_min)) ):
             print(abs(u_max - u_min))
             print(0.5*sqrt(dx*dy)*(abs(u_max)+abs(u_min)))
             # f(abs(u_max-u_min).le.(0.5*sqrt(e2%volume)*(abs(u_max)+abs(u_min)))) then
             thrs = np.random.uniform(0,1)
             if thrs < 0.8:
                 continue
             else:
                 fudge_label = 1




         u_c = entry[2]
         u_l = entry[3]
         u_r = entry[4]
         u_t = entry[5]
         u_b = entry[6]

         du_dx = (u_r - u_l)/(2.)#.*h) #map2(entry[5],maxval,minval)
         du_dy = (u_t-u_b)/(2.)

         du_dx_r = (u_r - u_c)
         du_dx_l = (u_c - u_l)
         du_dy_t = (u_t - u_c)
         du_dy_b = (u_c - u_b)

         uc_f_r =  entry[7]
         uc_f_l = entry[8]
         uc_f_t = entry[9]
         uc_f_b = entry[10]
         ul_f_r = entry[11]
         ur_f_l = entry[12]
         ut_f_b = entry[13]
         ub_f_t = entry[14]

         means = [u_b,u_r,u_t,u_l]
         mysides = [uc_f_b, uc_f_r, uc_f_t, uc_f_l]
         outsides = [ub_f_t,ur_f_l,ut_f_b,ul_f_r]

         if (fudge_label):
             label_hio = 0
         else:
             label_hio = entry[23]

         #normal = [xv,h, u_c, u_m,u_p, du, du_m, du_p, u_f_p, u_f_m, u_m_f_p, u_p_f_m, u_max, u_min, label_hio]
         normal = [dx, dy, u_c, u_l, u_r, u_t, u_b, uc_f_r, uc_f_l, uc_f_t, uc_f_b, ul_f_r, ur_f_l, ut_f_b, ub_f_t, u_max, u_min, du_dx, du_dy, du_dx_r, du_dx_l, du_dy_b, du_dy_t, label_hio]
         normalised.append(normal)

         if label_hio == 1:
             for perm in permutation:

                 u_c = u_c
                 u_l = means[perm[3]] #map2(entry[3],u_max,u_min)
                 u_r = means[perm[1]] #map2(entry[4],u_max,u_min)
                 u_t = means[perm[2]]#map2(entry[5],u_max,u_min)
                 u_b = means[perm[0]]#map2(entry[6],u_max,u_min)

                 uc_f_r = mysides[perm[1]]
                 uc_f_l = mysides[perm[3]]
                 uc_f_t = mysides[perm[2]]
                 uc_f_b = mysides[perm[0]]
                 ul_f_r = outsides[perm[3]]#map2(entry[11],u_max,u_min)
                 ur_f_l = outsides[perm[1]]#map2(entry[12],u_max,u_min)
                 ut_f_b = outsides[perm[2]]#map2(entry[13],u_max,u_min)
                 ub_f_t = outsides[perm[0]]#map2(entry[14],u_max,u_min)

                 du_dx = (u_r - u_l)/(2.)#.*h) #map2(entry[5],maxval,minval)
                 du_dy = (u_t-u_b)/(2.)

                 du_dx_r = (u_r - u_c)
                 du_dx_l = (u_c - u_l)
                 du_dy_t = (u_t - u_c)
                 du_dy_b = (u_c - u_b)

                 normal = [dx, dy, u_c, u_l, u_r, u_t, u_b, uc_f_r, uc_f_l, uc_f_t, uc_f_b, ul_f_r, ur_f_l, ut_f_b, ub_f_t, u_max, u_min, du_dx, du_dy, du_dx_r, du_dx_l, du_dy_b, du_dy_t, label_hio]
                 normalised.append(normal)

   return np.array(normalised)

def augment_data(text_file):
   #dataset.append([xv,h, u_c, u_m,u_p, du, du_m, du_p, u_f_p, u_f_m, u_m_f_p, u_p_f_m, u_max, u_min, label_hio])
   #np.random.RandomState(seed=1)
   print('entered augmented data')

   normalised = []

   permutation  = [[1,2,3,0],[2,3,0,1],[3,0,1,2]]

   for entry in text_file:
         u_max = entry[15] #np.max([entry[2],entry[3],entry[4],entry[7],entry[8]])
         u_min = entry[16] #np.min([entry[2],entry[3],entry[4],entry[7],entry[8]])

         """if (u_max == u_min):
             thrs = np.random.uniform(0,1)
             if thrs < 0.9:
                 continue

         dx = entry[0]
         dy = entry[1]"""

         dx = entry[0]
         dy = entry[1]
         fudge_label = 0
         if ( abs(u_max - u_min) < 0.5*np.sqrt(dx*dy)*(abs(u_max)+abs(u_min)) ):
              #print(abs(u_max - u_min))
              #print(0.5*np.sqrt(dx*dy)*(abs(u_max)+abs(u_min)))
              # f(abs(u_max-u_min).le.(0.5*sqrt(e2%volume)*(abs(u_max)+abs(u_min)))) then
              thrs = np.random.uniform(0,1)
              if thrs < 0.6:
                  continue
              else:
                  fudge_label = 1

         u_c = map2(entry[2],u_max,u_min)
         u_l = map2(entry[3],u_max,u_min)
         u_r = map2(entry[4],u_max,u_min)
         u_t = map2(entry[5],u_max,u_min)
         u_b = map2(entry[6],u_max,u_min)

         du_dx = (u_r - u_l)/(2.)#.*h) #map2(entry[5],maxval,minval)
         du_dy = (u_t-u_b)/(2.)

         du_dx_r = (u_r - u_c)
         du_dx_l = (u_c - u_l)
         du_dy_t = (u_t - u_c)
         du_dy_b = (u_c - u_b)

         uc_f_r =  map2(entry[7],u_max,u_min)
         uc_f_l = map2(entry[8],u_max,u_min)
         uc_f_t = map2(entry[9],u_max,u_min)
         uc_f_b = map2(entry[10],u_max,u_min)
         ul_f_r = map2(entry[11],u_max,u_min)
         ur_f_l = map2(entry[12],u_max,u_min)
         ut_f_b = map2(entry[13],u_max,u_min)
         ub_f_t = map2(entry[14],u_max,u_min)

         means = [u_b,u_r,u_t,u_l]
         mysides = [uc_f_b, uc_f_r, uc_f_t, uc_f_l]
         outsides = [ub_f_t,ur_f_l,ut_f_b,ul_f_r]

         #label_hio = entry[17]

         if (fudge_label==1):
              label_hio = 0
         else:
              label_hio = entry[17]

         #normal = [xv,h, u_c, u_m,u_p, du, du_m, du_p, u_f_p, u_f_m, u_m_f_p, u_p_f_m, u_max, u_min, label_hio]
         normal = [dx, dy, u_c, u_l, u_r, u_t, u_b, uc_f_r, uc_f_l, uc_f_t, uc_f_b, ul_f_r, ur_f_l, ut_f_b, ub_f_t, u_max, u_min, du_dx, du_dy, du_dx_r, du_dx_l, du_dy_b, du_dy_t, label_hio]
         normalised.append(normal)

         thrs = np.random.uniform(0,1)
         if label_hio == 1:
             if thrs < 0.5:
                       continue
         elif label_hio == 0:
             if thrs < 0.85:
                       continue

         for perm in permutation:

                 u_c = u_c
                 u_l = means[perm[3]] #map2(entry[3],u_max,u_min)
                 u_r = means[perm[1]] #map2(entry[4],u_max,u_min)
                 u_t = means[perm[2]]#map2(entry[5],u_max,u_min)
                 u_b = means[perm[0]]#map2(entry[6],u_max,u_min)

                 uc_f_r = mysides[perm[1]]
                 uc_f_l = mysides[perm[3]]
                 uc_f_t = mysides[perm[2]]
                 uc_f_b = mysides[perm[0]]
                 ul_f_r = outsides[perm[3]]#map2(entry[11],u_max,u_min)
                 ur_f_l = outsides[perm[1]]#map2(entry[12],u_max,u_min)
                 ut_f_b = outsides[perm[2]]#map2(entry[13],u_max,u_min)
                 ub_f_t = outsides[perm[0]]#map2(entry[14],u_max,u_min)

                 du_dx = (u_r - u_l)/(2.)#.*h) #map2(entry[5],maxval,minval)
                 du_dy = (u_t-u_b)/(2.)

                 du_dx_r = (u_r - u_c)
                 du_dx_l = (u_c - u_l)
                 du_dy_t = (u_t - u_c)
                 du_dy_b = (u_c - u_b)

                 normal = [dx, dy, u_c, u_l, u_r, u_t, u_b, uc_f_r, uc_f_l, uc_f_t, uc_f_b, ul_f_r, ur_f_l, ut_f_b, ub_f_t, u_max, u_min, du_dx, du_dy, du_dx_r, du_dx_l, du_dy_b, du_dy_t, label_hio]
                 normalised.append(normal)

   return np.array(normalised)

def map2(value,maxvalue,minvalue):
  """ Map function values to [-1,1] interval,
  the derivatives are rescaled but not mapped to [-1,1] """
  if (maxvalue == minvalue) and (maxvalue != 0):
    norm = value/float(maxvalue)
  elif (maxvalue == 0) and (maxvalue == minvalue):
    norm = value
  else:
    norm = (value-minvalue)/(maxvalue-minvalue) +  -1.0*(maxvalue-value)/(maxvalue-minvalue)

  return norm

def load_data(filepath, normalised=False):
  return generate_data_sets(filepath, normalised=normalised)

def boost_dataset(features, labels, ratio=0.05):
  """ Resample datapoints to get a more balanced dataset """
  positive_entries = []
  for indice, label in enumerate(labels):
    if label == 1:
      positive_entries.append(indice)

  number_of_samples = int((ratio*len(labels) - len(positive_entries))/float(1-ratio))
  #sample from positive entries
  print('current size of data: %i' %len(labels))
  print('number of samples: %i' %number_of_samples)
  print('number of positive samples: %i' %len(positive_entries))

  indices_to_repeat = np.random.choice(positive_entries,number_of_samples)

  print(features.shape)
  print(labels.shape)
  repeated_data = np.zeros((features.shape[0]+number_of_samples,features.shape[1]))
  repeated_labels = np.zeros((labels.shape[0] + number_of_samples))

  repeated_data[0:features.shape[0],:] = copy.deepcopy(features[:,:])
  repeated_labels[0:features.shape[0]] = copy.deepcopy(labels[:])
  for idx, indice in enumerate(indices_to_repeat):
    repeated_data[features.shape[0]+idx,:] = copy.deepcopy(features[indice,:])
    repeated_labels[features.shape[0]+idx] = copy.deepcopy(labels[indice])

  print(features.shape)
  print(repeated_data.shape)
  return repeated_data, repeated_labels

if __name__=='__main__':
    generate_data_sets('../dataset/dataset_new_10.4.18.csv', normalised = True)
