""" Optimal Contract Solver (The Standard Generalization Bound Version)

This script allows user to compute the optimal contract for the monopolistic screening problem in collaborative machine learning.

It contains two functions:
    - ContractSol_CompleteInfo: computes the optimal contract under complete information given N and p
    - ContractSol_SGB: computes the optimal contract under incomplete information given N and p

"""

# import the packages
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint, BFGS, fsolve


def ContractSol_CompleteInfo(N, p, c, k= None, a = None, a_der = None, v = None, v_der = None, verbose = False, bounds = False):
    """
    Solves the designed monopolistic screening problem with complete information in collaborative ML and returns the result.

    Args:
        N (int): Total number of participants.
        p (ndarray): Probabilities for different agent types.
        c (ndarray): Private costs for different agent types.
        a (func): Accuracy function
        a_der (func): Derivative of the accuracy function
        v (func): Valuation function
        v_der (func): Derivative of the valuation function
        verbose (bool): Whether or not to print out detailed optimization result
        bounds (bool): Whether or not to include bounds for the policy variables

    Returns:
        tuple: (result, solutions)
            result (dict): dictionary containing the averaged results
                'n_outcomes' (ndarray): all possible permutations of Multinomial(N,p)
                'n_probs' (ndarray): probabilities corresponding to the permutations
                'mr' (ndarray): averaged contributions and model rewards (in accuracy), weighted by the relative probabilities
                'success' (bool): whether the alogrithm converges
                'reserv_u' (ndarray): reservation utilties of the agents
                'utility' (ndarray): averaged utilities of the agents, weighted by the relative probabilities
                'mt' (ndarray): averaged contributions and model rewards (in monetary units), weighted by the relative probabilities
                't_cap' (float): averaged value of collectively trained model,  weighted by the relative probabilities
            solutions (list): list containing solutions for each permutation
    """
    #     K: number of types     
    K = len(c)

    # [Specified by user] ----- Define the accuracy and valuation functions -----
    # Demo mode will be triggered if the user does not supply accuracy and valuation functions.
    if a == None:
        def a(x):
            return 2/np.pi * np.arctan(0.05 * x)
        def a_der(x):
            return 1/(10*np.pi) * 1/(1+(0.05 * x)**2)
    if v == None:
        def v(x):
            return 100 * x

        def v_der(x):
            return np.ones(len(x)) * 100

    def u(x, c):
        return v(x[len(c):]) - c * x[:len(c)]
    # [Main Algorithm] 

    # <Step 1> ----- Calculate the fixed costs/reservation utility -----
    # Unconstrained minimisation problem
    def make_function(c):
        def function(x):
            return -(v(a(x)) - c*x)
        return function
    unconstrained_obj = [make_function(c[i]) for i in range(K)]
    bounds = [(0, None) for _ in range(K)]

    if k == None:
        k = 1
    m_reserv = np.array([minimize(unconstrained_obj[i], k * np.exp(-2)*v(1)/c[i],  bounds=[bounds[i]]).x.item() for i in range(K)])
    # make sure that m_reserv is non-negative.
    m_reserv = np.where(m_reserv>= 0, m_reserv, 0)

    # make sure that f is non-negative
    f = v(a(m_reserv)) - c * m_reserv
    f = np.where(f> 0, f, 0)
    m_reserv = np.where(f> 0, m_reserv, 0)


    # <Step 2> ----- Compute the expected numbers of participants of each type -----
    
    def multinorm_pmf(x):
        return stats.multinomial.pmf(x, N, p)
        # x: list of length K [n_1, ..., n_K]
        # N: total numebr of trials
        # p: list of length K [p_1, ..., p_K]

    def generate_multinomial_outcomes(n, k, prefix=[]):
        if k == 1:
            # Only one category left, all remaining trials must go into this category
            yield prefix + [n]
        else:
            for i in range(n + 1):
                # Allocate i trials to the current category, and recursively allocate the rest
                yield from generate_multinomial_outcomes(n - i, k - 1, prefix + [i])

    # Precompute all the possible outcomes
    all_outcomes = np.array([outcome for outcome in generate_multinomial_outcomes(N, K)])
    prob_outcomes = multinorm_pmf(all_outcomes)

    
    # <Step 3> ----- Define the objective function and its Jacobian -----
    #  N.b. the conversion of the maximisation problem to a minimisation problem
    def ContractSol_OneRound(n, c, k= None, a = None, a_der = None, v = None, v_der = None, verbose = False, bounds = False):
        def obj_f(x):
            return - a(n @ x[:int(len(x)/2)])

        def Jacobian_f(x):
            J_m = (n * a_der(n @ x[:int(len(x)/2)]))
            J = np.hstack([J_m,np.zeros(len(J_m))])
            return - J
        
        def obj_v(x):
            return v(-obj_f(x))

        # <Step 4> ----- Define the nonlinear constraints -----

        def NC_f(z):
            m, r = z[:int(len(z)/2)], z[int(len(z)/2):]
            output = np.zeros(len(z))
            output[:K] = - obj_f(z) - r
            output[K:] = v(r) - c * m - f
            return output

        def Jacobian_NC(z):
            m, r = z[:int(len(z)/2)], z[int(len(z)/2):]
            sec_1 = - np.repeat(Jacobian_f(z).reshape(1,-1), K, axis = 0) - np.hstack([np.zeros((K,K)), np.eye(K)])
            sec_2 = np.hstack([-np.diag(c), np.diag(v_der(r))])
            return np.vstack([sec_1, sec_2])

        def nonlinearConstraint(K):
            lb = 0
            ub = np.hstack([np.zeros(K), np.zeros(K)])
            return NonlinearConstraint(NC_f, lb, ub, jac = Jacobian_NC, hess= BFGS())

        nonlinear_constraint = nonlinearConstraint(K)

        # <Step 6> ----- Define the bounds -----
        bounds = Bounds(0, np.hstack([np.ones(K) * np.infty, np.ones(K)]))

        # <Step 7> ----- Find a good starting point (compute the pooling contract) -----
        def pool_f(m):
            return v(a(N*m)) - c[0] * m - f[0]

        root = fsolve(pool_f, (v(1)-f[0])/c[0])
        x0 = np.hstack([np.repeat(root, K), np.repeat(a(root), K)])

        # <Step 8> ----- Solve the nonlinear constrained optimisation problem and fetch the results-----
        if bounds:
            res = minimize(obj_f, x0, method = 'trust-constr', jac = Jacobian_f, hess = BFGS(), constraints = [nonlinear_constraint], options={'verbose': int(verbose)}, bounds = bounds)
        else:
            res = minimize(obj_f, x0, method = 'trust-constr', jac = Jacobian_f, hess = BFGS(), constraints = [nonlinear_constraint], options={'verbose': int(verbose)})

        res_mr = res['x']
        res_mt = np.hstack([res_mr[:len(c)], v(res_mr[len(c):])])

        return res_mr, res['success'], f, u(res_mr, c), res_mt, obj_v(res_mr)
    
    
    Solutions = [ContractSol_OneRound(outcome, c, k, a, a_der, v, v_der, verbose, bounds)  for outcome in all_outcomes]
    RES = {'n_outcomes': None, 'n_probs': None, 'mr': None, 'success': None, 'reserv_u': None, 'utility': None, 'mt': None, 't_cap': None}
    RES['n_outcomes'] = all_outcomes
    RES['n_probs'] = prob_outcomes
    RES['mr'] = np.average(np.array([solution[0] for solution in Solutions]), axis = 0, weights = prob_outcomes)
    RES['success'] = not np.any(np.array([solution[1] for solution in Solutions], dtype = object) != True)
    RES['reserv_u'] = np.average(np.array([solution[2] for solution in Solutions]), axis = 0, weights = prob_outcomes)
    RES['utility'] = np.average(np.array([solution[3] for solution in Solutions]), axis = 0, weights = prob_outcomes)
    RES['mt'] = np.average(np.array([solution[4] for solution in Solutions]), axis = 0, weights = prob_outcomes)
    RES['t_cap'] = np.average(np.array([solution[5] for solution in Solutions]), weights = prob_outcomes)
    
    return RES, Solutions
    
def ContractSol_SGB(N, p, c, k= None, a_funcs = None, v_funcs = None, exact_Hessian = True, maxiter = 1000, verbose = False, bounds = False):
    """
    Solves the designed monopolistic screening problem in collaborative ML and returns the result (tailored for using the generalization bound as the accuracy function.)
 
    Args:
        N (int): Total number of participants.
        p (ndarray): Probabilities for different agent types.
        c (ndarray): Private costs for different agent types.
        a_funcs (tuples): (a, a_der, a_hess) Accuracy function, its first and second derivatives.
        v_funcs (tuples):(v, v_der, v_hess) Valuation function, its first and second derivatives.
        exact_Hessian (bool): if True, compute exact Hessians; otherwise, use BFGS().
        maxiter (int): Maximum iternation of the trust-region algorithm.
        verbose (bool): Whether or not to print out detailed optimization result
        bounds (bool): Whether or not to include bounds for the policy variables
 
    Returns:
        Solution (dict): dictionary containing the solution to the monopolistic screening problem
            'n_outcomes': all possible permutations of Multinomial(N,p)
            'n_probs': probabilities corresponding to the permutations
            'mt': contributions and model rewards (in monetary units)
            'success': whether the alogrithm converges
            'reserv_u': reservation utilties of the agents
            'reserv_m': reservation contributions of the agents
            'utility': utilities of the agents by participating in the collaborative ML scheme
            't_bar': maximal rewardable model values expected by different agent types—this is used in propostional assignment
            'E[v(a_max)]': expected value of the collectively trained model
    """
    #     K: number of types     
    K = len(c)

    # [Specified by user] ----- Define the accuracy and valuation functions -----
    # Demo mode will be triggered if the user does not supply accuracy and valuation functions.
    def make_SGB(a_opt, k):
        def a(x):
            x = np.array(x, dtype = float)
            def b(x):
                return a_opt - (np.sqrt(2*k*(2+np.log(x/k)))+4)/np.sqrt(x)

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
                return (k*np.log(x/k)+ 2*np.sqrt(2*k*(2+np.log(x/k)))+k)/(x**(3/2)*np.sqrt(2*k*(np.log(x/k)+2)))

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

    if a_funcs == None:
        a, a_der, a_hess = make_SGB(1, 1)
    else:
        a, a_der, a_hess = a_funcs
    if v_funcs == None:
        def v(x):
            return 100 * x

        def v_der(x):
            return np.ones(len(x)) * 100
        
        def v_hess(x):
            return 0
    else:
        v, v_der, v_hess = v_funcs
        
    def u(x, c):
        return x[len(c):] - c * x[:len(c)]
    
    
    # [Main Algorithm] 
    # <Step 1> ----- Calculate the fixed costs/reservation utility -----
    # Unconstrained minimisation problem
    def make_function(c):
        def function(x):
            return -(v(a(x)) - c*x)
        return function
    unconstrained_obj = [make_function(c[i]) for i in range(K)]
    bounds = [(0, None) for _ in range(K)]

    if k == None:
        k = 1
    m_reserv = np.array([minimize(unconstrained_obj[i], k * np.exp(-2)*v(1)/c[i],  bounds=[bounds[i]]).x.item() for i in range(K)])
    # make sure that m_reserv is non-negative.
    m_reserv = np.where(m_reserv >= 0, m_reserv, 0)

    # make sure that f is non-negative
    f = v(a(m_reserv)) - c * m_reserv
    f = np.where(f > 0, f, 0)
    m_reserv = np.where(f > 0, m_reserv, 0)
    
    
    # <Step 2> ----- Pre-compute all possible outcomes and associated probabilities -----
    # Key bottleneck of the exact solver
    # Challenge: Combinatorial difficulty

    def multinorm_pmf(x):
        return stats.multinomial.pmf(x, N, p)
        # x: list of length K [n_1, ..., n_K]
        # N: total numebr of trials
        # p: list of length K [p_1, ..., p_K]

    def generate_multinomial_outcomes(n, k, prefix=[]):
        if k == 1:
            # Only one category left, all remaining trials must go into this category
            yield prefix + [n]
        else:
            for i in range(n + 1):
                # Allocate i trials to the current category, and recursively allocate the rest
                yield from generate_multinomial_outcomes(n - i, k - 1, prefix + [i])

    # Precompute all the possible outcomes
    all_outcomes = np.array([outcome for outcome in generate_multinomial_outcomes(N, K)])
    prob_outcomes = multinorm_pmf(all_outcomes)


    # <Step 3> ----- Define the objective function and its Jacobian -----
    #  N.b. the conversion of the maximisation problem to a minimisation problem
    def obj_f(x):
        return - sum(prob_outcomes *  a(all_outcomes @ x[:int(len(x)/2)]))

    def Jacobian_f(x):
        J_m = sum((all_outcomes * a_der(all_outcomes @ x[:int(len(x)/2)]).reshape(-1,1)) * prob_outcomes.reshape(-1,1))
        J = np.hstack([J_m,np.zeros(len(J_m))])
        return - J
    
    def Hessian_f(x):
        K = int(len(x)/2)
        H_tt = np.zeros((K,K))
        H_tm = np.zeros((K,K))
        H_mm = - sum((all_outcomes[:,np.newaxis,:] * all_outcomes[:,:, np.newaxis] * a_hess(all_outcomes @ x[:int(len(x)/2)]).reshape(-1,1,1))* prob_outcomes.reshape(-1,1,1))
        H = np.vstack([np.hstack([H_mm, H_tm]), np.hstack([H_tm, H_tt])])
        return H
    
    # Define the expected valuation
    def customize_outcomes(i):
        # i is the private type of the agent.
        mask = all_outcomes[:,i] >= 1
        outcomes = all_outcomes[mask] 
        probs = prob_outcomes[mask]/sum(prob_outcomes[mask])
        return outcomes, probs
    def expected_value(x, i = None):
        if type(i) != type(None):
            outcomes, probs = customize_outcomes(i)
        else:
            outcomes, probs = all_outcomes, prob_outcomes

        return sum(probs *  v(a(outcomes @ x[:int(len(x)/2)])))

    def Jacobian_ev(x,i):
        outcomes, probs = customize_outcomes(i)
        J_m = sum((outcomes * v_der(a(outcomes @ x[:int(len(x)/2)])).reshape(-1,1) * a_der(outcomes @ x[:int(len(x)/2)]).reshape(-1,1)) * probs.reshape(-1,1))
        J = np.hstack([J_m,np.zeros(len(J_m))])
        return J
    
    def Hessian_ev(x,i):
        outcomes, probs = customize_outcomes(i)
        K = int(len(x)/2)
        H_tt = np.zeros((K,K))
        H_tm = np.zeros((K,K))
        total_m = outcomes @ x[:int(len(x)/2)]
        a_val = a(total_m)
        inner = v_hess(a_val) * (a_der(total_m))**2 + v_der(a_val) * a_hess(total_m)
        H_mm = sum((outcomes[:,np.newaxis,:] * outcomes[:,:, np.newaxis] * inner.reshape(-1,1,1))* probs.reshape(-1,1,1))
        H = np.vstack([np.hstack([H_mm, H_tm]), np.hstack([H_tm, H_tt])])
        return H
    # <Step 4> ----- Define the linear constraints -----

    def linearConstraint(K):
        A_m = np.eye(K-1, K) - np.eye(K-1, K, k = 1)
        A_t = np.zeros((K-1, K))
        A = np.hstack([A_m, A_t])
        return LinearConstraint(A, -np.inf, 0) # LinearConstraint(A, lb_arry, ub_arry)

    linear_constraint = linearConstraint(K)

    # <Step 5> ----- Define the nonlinear constraints -----

    def NC_f(z):
        m, t = z[:int(len(z)/2)], z[int(len(z)/2):]
        output = np.zeros(3*K-1)
        output[:K] = np.array([expected_value(z,i) - t[i] for i in range(K)])
        output[K:2*K-1] = t[:-1] - t[1:] - c[1:] *(m[:-1] - m[1:])
        output[2*K-1:] = t - c * m - f
        return output

    def Jacobian_NC(z):
        m, t = z[:int(len(z)/2)], z[int(len(z)/2):]
        sec_1 = np.vstack([Jacobian_ev(z, i) for i in range(K)]) - np.hstack([np.zeros((K,K)), np.eye(K, K)])
        sec_2l = np.hstack([-np.diag(c[1:]), np.zeros(K-1).reshape(-1,1)]) + np.hstack([np.zeros(K-1).reshape(-1,1), np.diag(c[1:])])
        sec_2r = np.eye(K-1, K) - np.eye(K-1, K, k = 1)
        sec_2 = np.hstack([sec_2l, sec_2r])
        sec_3 = np.hstack([-np.diag(c), np.eye(K, K)])
        return np.vstack([sec_1, sec_2, sec_3])

    def Hessian_NC(z,v):
        H_sum = sum([v[i] * Hessian_ev(z,i) for i in range(K)])
        return H_sum
    
    def nonlinearConstraint(K):
        lb = 0
        ub = np.hstack([np.repeat(np.infty,K), np.zeros(K), np.repeat(np.infty,K-1)])
        if exact_Hessian:
            output = NonlinearConstraint(NC_f, lb, ub, jac = Jacobian_NC, hess= Hessian_NC)
        else:
            output = NonlinearConstraint(NC_f, lb, ub, jac = Jacobian_NC, hess= BFGS())
        return output

    nonlinear_constraint = nonlinearConstraint(K)
    
    # <Step 6> ----- Define the bounds -----
    bounds = Bounds(0, np.hstack([np.repeat(np.infty, K), np.repeat(np.infty, K)]))
    
    # <Step 7> ----- Find a good starting point (compute the pooling contract) -----
    def pool_f(m):
        return v(a(N*m)) - c[0] * m - f[0]

    root = fsolve(pool_f, (v(1)-f[0])/c[0])
    if pool_f(root) < -1e-8:
        root = 0
        
    x0 = np.hstack([np.repeat(root, K), np.repeat(a(root), K)])

    # <Step 8> ----- Solve the nonlinear constrained optimisation problem and fetch the results-----
    if exact_Hessian:
        hess_opt = Hessian_f
    else:
        hess_opt = BFGS()
        
    if bounds:
        res = minimize(obj_f, x0, method = 'trust-constr', jac = Jacobian_f, hess = hess_opt, constraints = [linear_constraint, nonlinear_constraint], options={'verbose': int(verbose), 'maxiter': maxiter}, bounds = bounds)
    else:
        res = minimize(obj_f, x0, method = 'trust-constr', jac = Jacobian_f, hess = hess_opt, constraints = [linear_constraint, nonlinear_constraint], options={'verbose': int(verbose), 'maxiter': maxiter})

    # If constraint violation occurs, flag it to the user.
    if res['constr_violation'] > 1e-5:
        res['success'] = 'constraint violation'

    # Store the results into a dictionary
    Solution = {'n_outcomes': None, 'n_probs': None, 'mt': None, 'success': None, 'reserv_u': None, 'reserv_m': None, 
                'utility': None, 't_bar': None, 'E[v(a_max)]': None}
    Solution['n_outcomes'] = all_outcomes
    Solution['n_probs'] = prob_outcomes
    Solution['mt'] = res['x']
    Solution['success'] = res['success']
    Solution['reserv_u'] = f
    Solution['reserv_m'] = m_reserv
    Solution['utility'] = u(res['x'], c)
    Solution['t_bar'] = np.array([expected_value(res['x'], i) for i in range(K)])
    Solution['E[v(a_max)]'] = expected_value(res['x'])

    # Print out necessary information if verbose:
    if verbose:
        # Return results
        print("Result:", res['x'])
        print("Success:", res['success'])
        print("Reservation Utility:", f)
        print("Utility:", u(res['x'],c))
        print("Expected Value:", np.array([expected_value(res['x'], i) for i in range(K)]))

    return Solution
