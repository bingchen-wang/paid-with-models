""" Generalization Bounds Setup

This script defines the accuracy function using the standard generalization bounds (Mohri, Rostamizadeh, and Talwalkar 2018; Karimireddy, Guo, and Jordan, 2022) 
and the valuation function as a linear function.

Import this to set up the environment for the replication of experiment results presented in the paper.

"""

import numpy as np

a_opt = 1
k = 1

def make_SGB(a_opt, k):
    def a(x):
        x = np.array(x, dtype = float)
        def b(x):
            b = a_opt - (np.sqrt(2*k*(2+np.log(x/k)))+4)/np.sqrt(x)
            return np.where(b>0,b,0)

        # Create a mask for the condition
        mask = x <= k * np.exp(-2)

        # Initialize the result array with zeros
        result = np.zeros_like(x)

        # Apply the b function only where the mask is False
        result[~mask] = b(x[~mask])
        return result

    def a_der(x):
        x = np.array(x, dtype = float)
        def b_der(x):
            b_der = (k*np.log(x/k)+ 2*np.sqrt(2*k*(2+np.log(x/k)))+k)/(x**(3/2)*np.sqrt(2*k*(np.log(x/k)+2)))
            return np.where(a(x)>0, b_der,0)

        # Create a mask for the condition
        mask = x <= k * np.exp(-2)

        # Initialize the result array with zeros
        result = np.zeros_like(x)

        # Apply the b function only where the mask is False
        result[~mask] = b_der(x[~mask])

        return result
    
    def a_hess(x):
        def b_hess(x):
            numerator = - 12 - 16* k - 3*k**2 - 4 * k * (3 + 2 *k) * np.log(x/k) - 3*k**2*(np.log(x/k))**2
            demoninator = 2*np.sqrt(2)*x**(5/2)*(2+2*k+k*np.log(x/k))**(3/2)
            return numerator/demoninator

        # Create a mask for the condition
        mask = x <= k * np.exp(-2)

        # Initialize the result array with zeros
        result = np.zeros_like(x)

        # Apply the b function only where the mask is False
        result[~mask] = b_hess(x[~mask])
        return result


    return a, a_der, a_hess

a, a_der, a_hess = make_SGB(a_opt, k)

def v(x):
    return 100 * x

def v_der(x):
    return np.ones(len(x)) * 100

def v_hess(x):
    return 0

def u(x,c):
    return x[len(c):] - c * x[:len(c)]