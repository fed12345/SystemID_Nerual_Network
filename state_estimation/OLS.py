import numpy as np


def Ols(alpha, alpha_val, beta, beta_val, V, V_val, C_m, C_m_v, m, n):
    """Ordinary Least Squares Estimaiton of C_m
    Args:
        alpha (array): angle of attack
        beta (array): angle of sideslip
        V (array): Velocity
        C_m (array): Pitching moment coefficient
        m (int): Order of polynomial
        n (int): number of states
    """    
    n_entry_eval = np.size(alpha)
    n_entry = np.size(alpha_val)

    #Create regression Matrix
    A = np.ones((n_entry_eval, n*m + 1), dtype=float)
    A_val = np.ones((n_entry, n*m + 1), dtype=float)

    #Fill in regression matrix
    for i in range(0,m):
        shift = i*n
        A[:,shift+1] = alpha.ravel()**(i+1)
        A[:,shift+2] = beta.ravel()**(i+1)
        A[:,shift+3] = V.ravel()**(i+1)
        A_val[:,shift+1] = alpha_val.ravel()**(i+1)
        A_val[:,shift+2] = beta_val.ravel()**(i+1)
        A_val[:,shift+3] = V_val.ravel()**(i+1)

    #Find Coefficients
    COV =np.linalg.inv(A.transpose().dot(A))
    theta = COV.dot(A.transpose()).dot(C_m)

    #Evaluate Coefficients
    C_m_eval = A.dot(theta)
    C_m_val = A_val.dot(theta)

    #Calculate mean square error for C_m(on validation data)
    error_eval = C_m - C_m_eval
    error = C_m_v - C_m_val
    MSE = np.sum((error_eval)**2)/n_entry


    #Calculate parameter Covariance Matirx
    var = np.mean(error.transpose().dot(error))/(n_entry - n*m)
    COV_Theta  = var*COV

    #Chek if estimator is BLUE
    mean_error = np.mean(error)
    
    return C_m_eval, MSE, COV_Theta,mean_error


def OlsSpecialValidation(alpha, beta, C_m, m, n):
    """Ordinary Least Squares Estimaiton of C_m
    Args:
        alpha (array): angle of attack
        beta (array): angle of sideslip
        C_m (array): Pitching moment coefficient
        m (int): Order of polynomial
        n (int): number of states
    """    
    n_entry = np.size(alpha)
    #Create regression Matrix
    A = np.ones((n_entry, n*m + 1 + 1), dtype=float)

    #Fill in regression matrix
    for i in range(0,m):    
        shift = i*n
        A[:,[shift+1]] = alpha**(i+1)
        A[:,[shift+2]] = beta**(i+1)
    #Add cross terms
    A[:,[-1]] = alpha*beta
    #Find Coefficients
    COV =np.linalg.inv(A.transpose().dot(A))
    theta = COV.dot(A.transpose()).dot(C_m)
    #Evaluate Coefficients
    C_m_eval = A.dot(theta)
    #Calculate mean square error for C_m(on validation data)
    error = C_m - C_m_eval
    MSE = np.sum((error)**2)/n_entry

    return C_m_eval, MSE