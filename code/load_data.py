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
import math

def quat_prod(q1, q2):
  q1s = q1[0]
  q2s = q2[0]
  q1v = q1[1:]
  q2v = q2[1:]
  q_cross = list(np.cross(q1v, q2v))
  q_temp1 = [q1s * q for q in q2v]
  q_temp2 = [q2s * q for q in q1v]
  qv = [sum(q) for q in zip(*[q_cross, q_temp1, q_temp2])]
  qs = (q1s * q2s) - (np.dot(q1v, q2v))
  
  return [qs, qv[0], qv[1], qv[2]]

def exp_quat(q):
  qs = q[0]
  qv = q[1:]
  qv_norm = np.linalg.norm(np.array(qv))
  if qv_norm <= 1e-3: return [1, 0, 0, 0]
  qv = [q * (np.sin(qv_norm) / qv_norm) for q in qv]
  quat = [np.cos(qv_norm), qv[0], qv[1], qv[2]]
  quat = [math.exp(qs) * q for q in quat]
  return quat


vicon = vicd['rots']
imu = imud['vals']

bias_acc = [sum(imu[0, :750]) / 750, sum(imu[1, :750]) / 750, sum(imu[2, :750]) / 750]
scale_factor_acc = (3300 / (1023 * 300)) * (3.14159265 / 180)
acc = []

bias_omega = [sum(imu[3, :750]) / 750, sum(imu[4, :750]) / 750, sum(imu[5, :750]) / 750]
scale_factor_gyro = (3300 / (1023 * 190.8)) * (3.14159265 / 180)
omega = []

#Build acceleration and angular velocity array
for i in range(len(imu[0])):
  acc.append([(imu[0][i] - bias_acc[0]), (imu[1][i] - bias_acc[1]), (imu[2][i] - bias_acc[2])])
  acc[-1][0] = acc[-1][0] * scale_factor_acc
  acc[-1][1] = acc[-1][1] * scale_factor_acc
  acc[-1][2] = acc[-1][2] * scale_factor_acc  
  omega.append([(imu[3][i] - bias_omega[0]), (imu[4][i] - bias_omega[1]), (imu[5][i] - bias_omega[2])])
  omega[-1][0] = omega[-1][0] * scale_factor_gyro
  omega[-1][1] = omega[-1][1] * scale_factor_gyro
  omega[-1][2] = omega[-1][2] * scale_factor_gyro

orientation = []

#Build Euler angles orientation vector array
for i in range(len(vicon[0][0])):
  rotation_matrix = [[vicon[0][0][i], vicon[0][1][i], vicon[0][2][i]],[vicon[1][0][i], vicon[1][1][i], vicon[1][2][i]], [vicon[2][0][i], vicon[2][1][i], vicon[2][2][i]]]
  R = np.asmatrix(rotation_matrix)
  quaternions = transforms3d.quaternions.mat2quat(R)
  euler = transforms3d.euler.quat2euler(quaternions)
  orientation.append(list(euler))

orientation = np.array(orientation)
acc = np.array(acc)
omega = np.array(omega)

quat_motion_model = []
quat_motion_model.append([1,0,0,0])

for i in range(omega.shape[0]):
  if i == 0:
    continue
  timestep = imud['ts'][0][i] - imud['ts'][0][i - 1]
  theta_vector = [timestep * omega[i - 1][1], timestep * omega[i - 1][2], timestep * omega[i - 1][0]]
  if i >= 1600 and i<=2200: print([temp / timestep for temp in theta_vector])
  #eta = [theta_comp / np.linalg.norm(np.array(theta_vector)) for theta_comp in theta_vector]
  s = (np.sin((np.linalg.norm(np.array(theta_vector))) / 2)) / np.linalg.norm(np.array(theta_vector))
  c = np.cos((np.linalg.norm(np.array(theta_vector))) / 2)
  delta_quat = ([c, s * theta_vector[0], s * theta_vector[1], s * theta_vector[2]])
  #delta_quat = exp_quat([0, (timestep * omega[i - 1][1]) / 2, (timestep * omega[i - 1][2]) / 2, (timestep * omega[i - 1][0]) / 2])
  quat_motion_model.append(quat_prod(quat_motion_model[-1], delta_quat))

orientation_motion_model = []

for i in range(len(quat_motion_model)):
  orientation_motion_model.append(list(transforms3d.euler.quat2euler(np.array(quat_motion_model[i]))))

orientation_motion_model = np.array(orientation_motion_model)

#plt.plot(acc[:,1])
#plt.plot(acc[:,2])
#plt.plot(acc[:,0])
plt.plot(orientation_motion_model[:,1])
plt.plot(orientation[:,1], color="red")
plt.show()