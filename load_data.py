import numpy as np
import pandas as pd

def loadData(measurment_path, validate_path):
    """
    Loads the data from the given path.
    """
    with open(measurment_path) as file_name:
        data = np.loadtxt(file_name, delimiter=",")

    Cm = data[:,0]
    alpha_m = data[:,1]
    beta_m = data[:,2]
    Vtot = data[:,3]
    Z = np.array([data[:,1], data[:,2], data[:,3]])
    U = np.array([data[:,4], data[:,5], data[:,6]])
    Au = data[:,4]
    Av = data[:,5]
    Aw = data[:,6]

    with open(validate_path) as file_name:
        data = np.loadtxt(file_name, delimiter=",")

    Cm_v = data[:,[0]]
    alpha_v = data[:,[1]]
    beta_v = data[:,[2]]

    return Cm, U, Z, Cm_v, alpha_v, beta_v


