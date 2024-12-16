# Paid with Models: Optimal Contract Design for Collaborative Machine Learning
Codebase for "Paid with Models: Optimal Contract Design for Collaborative Machine Learning". Implementation and experiments of optimal contracting design for incentivizing machine learning collaboration.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/status-active-brightgreen)](https://github.com/bingchen-wang/paid-with-models)

![Optimal Contract Design for Collaborative Machine Learning: The Timeline.](assets/figures/Timeline.png)

## Author & Contact Information
- **Name:** [Bingchen Wang](https://github.com/bingchen-wang)  
- **Email:** [bw2506@columbia.edu](mailto:bw2506@columbia.edu)  
- **Institution/Organization:** National University of Singapore (NUS)
- **Website/Portfolio:** [Bingchen's Google Site](https://sites.google.com/view/bingchenwang/)

Feel free to contact me if you have any code-related question.
## Requirements
Ensure Python dependencies are met by installing the following:

```bash
pip install -r requirements.txt
```

## Replication of Paper Results

### Section 6.1 Two-type Case
<div style="text-align: center; width: 50%; margin: 0 auto;">
  <img src="assets/figures/Two-type_case_onecolumn.png" alt="Two-type Case" width="50%">
  <p style="text-align: left; line-height: 1.6;">
    <strong>Top:</strong> Optimal contracts under incomplete information for varied probability of high-cost type 
    $p_1 \in (0,1)$ and total number of participants 
    $N \in [2, 100]$, with $c = \{0.02, 0.01\}$. <br>
    <strong>Bottom:</strong> Information costs for the coordinator and information rents for the parties under incomplete information vis-à-vis complete information.
  </p>
</div>

> You can replicate Section 6.1 by running the script [📜 Experiment_Twotype_Case.ipynb](Experiment_Twotype_Case.ipynb).

### Section 6.2 Multi-type Case
<div style="text-align: center; width: 50%; margin: 0 auto;">
  <img src="assets/figures/Multitype_case_onecolumn.png" alt="Multi-type Case" width="50%">
  <p style="text-align: left; line-height: 1.6;">
    Optimal contract designs for multi-type scenarios. <br> 
    <strong>Scenario 1:</strong> All types would train a model on their own. <br> 
    <strong>Scenario 2:</strong> All types would not train a model on their own due to prohibitive costs. <br> 
    <strong>Scenario 3:</strong> Some types would train the model on their own and others would not.
  </p>
</div>

> You can replicate Section 6.2 by running the script [📜 Experiment_Multitype_Case.ipynb](Experiment_Multitype_Case.ipynb).

### Appendix A.1 Related Work
<div style="text-align: center; width: 50%; margin: 0 auto;">
  <img src="assets/figures/max accu not the same as max data volume.png" alt="Related Work" width="50%">
  <p style="text-align: left; line-height: 1.6;">
    Functions $f$ and $g$ and the $m_1$'s that give the maximum values.
  </p>
</div>

> You can replicate Appendix A.1 by running the script [📜 Appendix_A_Related_Work.ipynb](Appendix_A_Related_Work.ipynb).

## Additional Experiments
In addition to the experiments presented in the paper, we provide two additional experiments in the repository for interested readers.


### Constraint Simplication: Speed-up effect?
The contribution of constraint analysis in the paper comes in two fronts. First, it helps establish clean properties of the optimal contract. Second, it may help speed up computation (~15% for Sec. 6.2.).  However, the effect depends significantly on the choice of optimization algorithm—it is expected to be more pronounced for active-set methods [1] and likely to matter less so for others.

> The scripts [📜 Experiment_Twotype_Case.ipynb](Experiment_Twotype_Case.ipynb) and [📜 Experiment_Multitype_Case_noSimp.ipynb](Experiment_Multitype_Case_noSimp.ipynb) run through the experiments in Sec. 6.1 and Sec. 6.2 without the constraint simplication.

### Scalability: Running-time increase with $N$ and $I$.
Our preliminary experiments show that the algorithm’s running time increases at roughly a factor of 2 per additional type added, while increments in $N$ has less effect. This corroborates our discussion on the combinatorial challenge in Appendix B.9. Our current model is best suited for cases where $N$ and $I$ are reasonably finite, as is the case of cross-silos collaboration.  For reference, it took 8.03 seconds (wall-clock) to run the experiments in Sec 6.2 on a Macbook Pro with M2 chip. We posit that scalability can be improved with relaxation on the distribution assumption, which is an active research area [2]. It can also be improved with better approximation algorithms, though with a trade-off between welfare and speed of computation.

> Interested reader can implement [📜 Scalability_check_increasing_N.py](Scalability_check_increasing_N.py) and [📜 Scalability_check_increasing_types.py](Scalability_check_increasing_types.py) to see the increase in running time.



## References

1. **Nocedal, J., & Wright, S. J. (2006).** *Numerical Optimization*. Springer, 2006, p.424. [Available through Springer](https://link.springer.com/book/10.1007/978-0-387-40065-5).  

2. **Dütting, P., et al. (2019).** *Simple versus Optimal Contracts*. Presented at EC'19. [Available through ACM Digital Library](https://dl.acm.org/doi/10.1145/3328526.3329591).
