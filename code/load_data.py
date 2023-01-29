import pickle
import sys
import time 

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # need for python 3
  return d

dataset="1"
cfile = "./../../data/cam/cam" + dataset + ".p"
ifile = "./../../data/imu/imuRaw" + dataset + ".p"
vfile = "./../../data/vicon/viconRot" + dataset + ".p"

ts = tic()
camd = read_data(cfile)
imud = read_data(ifile)
vicd = read_data(vfile)
toc(ts,"Data import")

#============================================================================#
#My Code
#============================================================================#

import matplotlib.pyplot as plt
import numpy as np
import transforms3d

vicon = vicd['rots']

orientation = []

for i in range(len(vicon[0][0])):
  rotation_matrix = [[vicon[0][0][i], vicon[0][1][i], vicon[0][2][i]],[vicon[1][0][i], vicon[1][1][i], vicon[1][2][i]], [vicon[2][0][i], vicon[2][1][i], vicon[2][2][i]]]
  R = np.asmatrix(rotation_matrix)
  quaternions = transforms3d.quaternions.mat2quat(R)
  euler = transforms3d.euler.quat2euler(quaternions)
  orientation.append(list(euler))

orientation = np.array(orientation)
plt.plot(orientation[:,0])
plt.show()