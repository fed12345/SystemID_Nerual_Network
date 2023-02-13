import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import control.matlab
from load_data import loadData
from state_estimation.state_dynamics.kf_calc_f import kf_calc_f
from state_estimation.state_dynamics.kf_calc_h import kf_calc_h
from state_estimation.state_dynamics.rk4 import rk4
from state_estimation.state_dynamics.kf_calc_Fx import kf_calc_Fx
from state_estimation.state_dynamics.kf_calc_Hx import kf_calc_Hx

def iterativeEKL(U_k, Z_k, plotting = False):
    ########################################################################
    ## Set simulation parameters
    ########################################################################

    n               = 4                 # state dimension
    nm              = 3                 # number of measurements
    m               = 3                # number of inputs
    dt              = 0.01              # time step (s)
    N               = len(U_k[0])      # sample dimension
    epsilon         = 10**(-10)         # IEKF threshold
    doIEKF          = True              # If false, EKF without iterations is used
    maxIterations   = 100               # maximum amount of iterations per sample

    ########################################################################
    ## Set initial values for states and statistics
    ########################################################################

    E_x_0       = np.matrix('160; 0; 0; 0')    # initial estimate of optimal value of x_k1_k1
    x_0         = np.matrix('2; -3')    # initial true state

    B           = np.matrix(np.eye(m))
    B           = np.vstack([B,[0,0,0]])  # input matrix
    G           = np.matrix(np.eye(n))
    


    # Initial estimate for covariance matrix
    std_x_0     = np.matrix('0.02, 0.02, 0.02, 0.02')                   # initial standard deviation of state prediction error
    P_0         = np.diagflat(np.power(std_x_0, 2))     # initial covariance of state prediction error

    # Process noise statistics
    E_w         = np.matrix('0, 0, 0, 0')
    std_w_u     = 1e-3
    std_w_v     = 1e-3
    std_w_w     = 1e-3
    std_w_c     = 0
    std_w       = np.matrix([std_w_u, std_w_v,std_w_w,std_w_c])         # standard deviation of system noise
    Q           = np.diagflat(np.power(std_w, 2))  # variance of system noise

    # Measurement noise statistics
    E_v         = np.matrix('0, 0, 0')
    std_v_a     = 0.035
    std_v_B     = 0.013
    std_v_V     = 0.110           # bias of measurement noise
    std_v       = np.matrix([std_v_a,std_v_B,std_v_V])             # standard deviation of measurement noise
    R           = np.diagflat(np.power(std_v, 2))         # variance of measurement noise

    ########################################################################
    ## Initialize Extended Kalman filter
    ########################################################################

    t_k         = 0
    t_k1        = dt

    # allocate space to store traces
    XX_k1_k1    = np.zeros([n, N])
    PP_k1_k1    = np.zeros([n, n, N])
    STD_x_cor   = np.zeros([n, N])
    STD_z       = np.zeros([nm, N])
    ZZ_pred     = np.zeros([nm, N])
    alpha_t     = np.zeros([N, 1])
    beta_t      = np.zeros([N, 1])
    V_t         = np.zeros([N, 1])
    IEKFitcount = np.zeros([N, 1])

    # initialize state estimation and error covariance matrix
    x_k1_k1     = E_x_0     # x(0|0) = E(x_0)
    P_k1_k1     = P_0       # P(0|0) = P(0)

    t0          = time.time()

    # Run the filter through all N samples
    for k in range(0, N):

        # x(k+1|k) (prediction)
        t, x_k1_k   = rk4(kf_calc_f, x_k1_k1, U_k[:,k], [t_k, t_k1])

        # Calc Jacobians Phi(k+1, k) and Gamma(k+1, k)
        Fx          = kf_calc_Fx(0, x_k1_k, U_k[:,k])
        # Continuous to discrete time transformation of Df(x,u,t)
        ss_B        = control.matlab.ss(Fx, B, np.zeros([4,4]), 0)
        ss_G        = control.matlab.ss(Fx, G, np.zeros([4,4]), 0)
        Psi         = control.matlab.c2d(ss_B, dt).B
        Phi         = control.matlab.c2d(ss_G, dt).A
        Gamma       = control.matlab.c2d(ss_G, dt).B

        # P(k+1|k) (prediction covariance matrix)
        P_k1_k      = Phi*P_k1_k1*Phi.transpose() + Gamma*Q*Gamma.transpose()
        eta2    = x_k1_k
        err     = 2*epsilon
        itts    = 0
        while (err > epsilon):
            if (itts >= maxIterations):
                print("Terminating IEKF: exceeded max iterations (%d)\n" %(maxIterations))
                break

            itts    = itts + 1
            eta1    = eta2

            # Construct the Jacobian H = d/dx(h(x))) with h(x) the observation model transition matrix
            Hx           = kf_calc_Hx(0, eta1, U_k[:,k])

            # Observation and observation error predictions
            z_k1_k      = kf_calc_h(0, eta1, U_k[:,k])                         # prediction of observation (for validation)
            P_zz        = Hx*P_k1_k*Hx.transpose() + R      # covariance matrix of observation error (for validation)
            std_z       = np.sqrt(np.diag(P_zz))            # standard deviation of observation error (for validation)

            # K(k+1) (gain)
            K           = (P_k1_k*Hx.transpose())*np.linalg.inv(P_zz)  

            # new observation
            eta2        = x_k1_k + K*(np.matrix(Z_k[:,k]).transpose() - z_k1_k - Hx*(x_k1_k - eta1))
            err         = np.linalg.norm(eta2-eta1)/np.linalg.norm(eta1)

        IEKFitcount[k]  = itts
        x_k1_k1         = eta2

        # P(k|k) (correction) using the numerically stable form of P_k_1k_1 = (eye(n) - K*Hx) * P_kk_1
        P_k1_k1     = (np.eye(n) - K*Hx)*P_k1_k*(np.eye(n) - K*Hx).transpose() + K*R*K.transpose()
        std_x_cor   = np.sqrt(np.diag(P_k1_k1))         # standard deviation of state estimation error (for validation)

        # Next step
        t_k         = t_k1
        t_k1        = t_k1 + dt

        # store results
        ZZ_pred[:,k]        = z_k1_k.reshape(3,)
        XX_k1_k1[:, k]      = np.array(x_k1_k1).transpose()
        PP_k1_k1[:,:,k]     = P_k1_k1
        STD_x_cor[:, k]     = std_x_cor
        STD_z[:, k]         = std_z
        alpha_t[k] = np.arctan(x_k1_k1[2]/x_k1_k1[0])
        beta_t[k] = np.arctan(x_k1_k1[1]/np.sqrt(x_k1_k1[0]**2+x_k1_k1[2]**2))
        V_t[k] = np.sqrt(x_k1_k1[0]**2+x_k1_k1[1]**2+x_k1_k1[2]**2)
        t1 = time.time()

        # calculate measurement estimation error (possible in real life)
        EstErr_z    = ZZ_pred-Z_k

    if plotting == True:
        fig0, ax0 = plt.subplots(nm,figsize=(12,6))
        ax0[0].plot(Z_k[0].transpose(), 'b', label='alpha')
        ax0[0].plot(ZZ_pred[0].transpose(), 'r', label='alpha pred')
        ax0[0].plot(alpha_t, 'g', label='alpha true')
        plt.grid(True)
        ax0[1].plot(Z_k[1].transpose(), 'b', label='beta')
        ax0[1].plot(ZZ_pred[1].transpose(), 'r', label='beta pred')
        ax0[1].plot(beta_t, 'g', label='beta true')
        plt.grid(True)
        ax0[2].plot(Z_k[2].transpose(), 'b', label='Velocity')
        ax0[2].plot(ZZ_pred[2].transpose(), 'r', label='Velocity pred')
        ax0[2].plot(V_t, 'g', label='Velocity true')
        plt.xlim(0,N)
        plt.grid(True)
        fig0.legend()
        fig0.suptitle('True States and measurement')

        plt.figure(figsize=(12,6))
        plt.plot(Z_k[0].transpose(), color = '#00A6D6', label='Measured Angle of Attack')
        plt.plot(ZZ_pred[0].transpose(), 'r', label='Predicted Angle of Attack')
        plt.plot(alpha_t, 'g', label='Corrected Angle of Attack')
        plt.xlim(0,N)
        plt.grid(True)
        plt.ylabel('Angle [rad]')
        plt.xlabel('Time [cs]')
        plt.title('Angle of Attack')
        plt.legend()
        plt.savefig('plots/AOA', format='eps')

        plt.figure(figsize=(12,6))
        plt.plot(XX_k1_k1[3].transpose(), color = '#00A6D6', label='Upwash Coefficient')
        plt.xlim(0,N)
        plt.grid(True)
        plt.ylabel('Upwash Coefficient[-]')
        plt.xlabel('Time [cs]')
        plt.title('Estimated Upwash Coefficient')
        plt.savefig('plots/Upwash_Coefficient', format='eps')


        fig1, ax1 = plt.subplots(nm, figsize=(12,6))
        plt.suptitle('Measurement estimation error with STD')
        for state in range(0,nm):
            if (state == 0):
                ax1[state].plot(STD_z[state].transpose(), 'r')
                ax1[state].plot(EstErr_z[state].transpose(), 'b')
                ax1[state].plot(STD_z[state].transpose()*-1, 'g')
            if (state == 1):
                ax1[state].plot(STD_z[state].transpose(), 'r--')
                ax1[state].plot(EstErr_z[state].transpose(), 'b--')
                ax1[state].plot(STD_z[state].transpose()*-1, 'g--')
            if (state == 2):
                ax1[state].plot(STD_z[state].transpose(), 'r--')
                ax1[state].plot(EstErr_z[state].transpose(), 'b--')
                ax1[state].plot(STD_z[state].transpose()*-1, 'g--')
            ax1[state].set_xlim(0,N)
            ax1[state].grid(True)
            ax1[state].set_title('Measurement '+str(state+1))
            ax1[state].legend([ 'upper error STD', 'estimation error','lower error STD'], loc='upper right')


        plt.figure(figsize=(12,6))
        plt.plot(IEKFitcount, 'b')
        plt.xlim(0, N)
        if (np.max(IEKFitcount) > 0):
            plt.ylim(0, np.max(IEKFitcount))
        plt.grid(True)
        plt.title('IEKF iterations at each sample')
        plt.show()
    return alpha_t, beta_t, V_t


# TODO: Plot C a upwash and calucalte alpha true

# if __name__ == '__main__':

#     U_k, Z_k, Cm_v, alpha_v, beta_v = load_data('F16traindata_CMabV_2022.csv', 'F16validationdata_CMab_2022.csv')
#     print("Finished loading data...")
#     Iterative_EKL(U_k, Z_k, plotting=True)