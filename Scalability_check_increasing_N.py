from ContractSol_SGB import *
from MultitypeVisualizer import *
from Generalization_Bounds_Setup import *
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import scipy.stats as stats

NRange = 2**np.arange(1,11)
K = 5
k = 1

contract_sol = []
ComputeTime = []


for i, N in enumerate(NRange):
    start_time = time.time()
    p = np.repeat(1/K,K)
    c = np.linspace(1, 0.4, K)
    print('-'*43 + f'{{ Scenario {i+1} }}'+ '-'*43 )
    print(f'cost structure: {c}')
    contract_sol.append(ContractSol_SGB(N, p, c, k, (a, a_der, a_hess), (v, v_der, v_hess), True, 2000, True))
    print('-'*100)
    print('')
    end_time = time.time()
    elapsed_time = end_time - start_time
    ComputeTime.append(elapsed_time)
    print(f"Elapsed time: {elapsed_time} seconds")

df = pd.DataFrame({
    'N': NRange,
    'ComputeTime': ComputeTime
})

# Save DataFrame to a CSV file
df.to_csv('compute_time_results_N.csv', index=False)
print("DataFrame saved as 'compute_time_results_N.csv'")