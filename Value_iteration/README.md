# value iteration algorithm on a basic problem 

This project implements a **Value Iteration** algorithm on a basic problem.


## Bellman's Equation

$$
V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a) \Big[ R(s,a,s') + \gamma V_k(s') \Big]
$$

## Optimal Policy Extraction

$$
\pi^*(s) = \arg\max_a \sum_{s'} P(s'|s,a) \Big[ R(s,a,s') + \gamma V(s') \Big]
$$