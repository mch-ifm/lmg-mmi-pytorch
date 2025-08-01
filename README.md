# lmg-mmi-pytorch
A driven-dissipative version of the permutation-invariant Lipkin-Meshkov-Glick model. Finite system size case requires finding the null vector of a large sparse matrix, which we achieve efficiently with pytorch. The thermodynamic limit of $N\to\infty$ is easy to solve with `scipy.integrate.solve_ivp`.
