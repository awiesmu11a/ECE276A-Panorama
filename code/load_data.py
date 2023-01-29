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
imu = imud['vals']

bias_acc = 0
scale_factor_acc = (3300 / (1023 * 300)) * (3.14159265 / 180)
acc = []

bias_omega = 0
scale_factor_gyro = (3300 / (1023 * 190.8)) * (3.14159265 / 180)
omega = []


for i in range(len(imu[0])):
  acc.append([(imu[0][i] - bias_acc), (imu[1][i] - bias_acc), (imu[2][i] - bias_acc)])
  acc[-1][0] = acc[-1][0] * scale_factor_acc
  acc[-1][1] = acc[-1][1] * scale_factor_acc
  acc[-1][2] = acc[-1][2] * scale_factor_acc  
  omega.append([(imu[3][i] - bias_acc), (imu[4][i] - bias_acc), (imu[5][i] - bias_acc)])
  omega[-1][0] = omega[-1][0] * scale_factor_gyro
  omega[-1][1] = omega[-1][1] * scale_factor_gyro
  omega[-1][2] = omega[-1][2] * scale_factor_gyro

orientation = []

for i in range(len(vicon[0][0])):
  rotation_matrix = [[vicon[0][0][i], vicon[0][1][i], vicon[0][2][i]],[vicon[1][0][i], vicon[1][1][i], vicon[1][2][i]], [vicon[2][0][i], vicon[2][1][i], vicon[2][2][i]]]
  R = np.asmatrix(rotation_matrix)
  quaternions = transforms3d.quaternions.mat2quat(R)
  euler = transforms3d.euler.quat2euler(quaternions)
  orientation.append(list(euler))

orientation = np.array(orientation)
acc = np.array(acc)
omega = np.array(omega)
plt.plot(acc[:,2])
plt.show()