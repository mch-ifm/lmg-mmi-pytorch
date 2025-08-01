import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

# Simulation parameters
ts = 1000.0  # start time for averaging
tf = 2000.0  # final time
t_eval = np.linspace(ts, tf, 20000)
J = 3.0
g = 1.0
h = 0.5
Gamma_np = np.linspace(0, 5, 21)


def ent_np(x0):
    clamp_epsilon = 1e-16
    x_min = -1.0 + clamp_epsilon
    x_max =  1.0 - clamp_epsilon
    assert x_min > -1.0 and x_max < 1.0
    x = np.clip(x0, x_min, x_max)
    return (-((1 - x) / 2) * np.log((1 - x) / 2) - ((1 + x) / 2) * np.log((1 + x) / 2)) * (x0 > -1.0) * (x0 < 1.0)


def ode_func_np(t, y, J, g, Gamma, h):
    x, y_val, z = y
    dxdt = J * y_val * z - (g * x) / 2 + (Gamma * x * z) / 2
    dydt = -J * x * z - (g * y_val) / 2 + (Gamma * y_val * z) / 2 - h * z
    dzdt = g * (1 - z) - (Gamma * (x ** 2 + y_val ** 2)) / 2 + h * y_val
    return [dxdt, dydt, dzdt]


mutinf_list = []
mag_avg_list = []
ent_avg_list = []
ent_rloc_list = []

for Gamma in tqdm(Gamma_np):
    sol = solve_ivp(ode_func_np, [0, tf], [0.1, 0.1, 0.0], args=(J, g, Gamma, h), dense_output=True)
    y_vals = sol.sol(t_eval)
    x_vals = y_vals[0]
    y_vals_comp = y_vals[1]
    z_vals = y_vals[2]

    x_avg = np.trapezoid(x_vals, t_eval) / (tf - ts)
    y_avg = np.trapezoid(y_vals_comp, t_eval) / (tf - ts)
    z_avg = np.trapezoid(z_vals, t_eval) / (tf - ts)
    mag_avg_list.append([x_avg.item(), y_avg.item(), z_avg.item()])

    r_vals = np.sqrt(x_vals ** 2 + y_vals_comp ** 2 + z_vals ** 2)
    ent_vals = ent_np(r_vals)
    ent_avg = np.trapezoid(ent_vals, t_eval) / (tf - ts)
    ent_avg_list.append(ent_avg.item())

    r_avg = np.sqrt(x_avg ** 2 + y_avg ** 2 + z_avg ** 2)
    ent_rloc = ent_np(r_avg)
    ent_rloc_list.append(ent_rloc.item())
    mutinf = ent_rloc - ent_avg
    mutinf_list.append(mutinf.item())

print(f"""
gamma = np.array({Gamma_np.tolist()})
mutinf = np.array({mutinf_list})
mag_avg = np.array({mag_avg_list})
ent_avg = np.array({ent_avg_list})
ent_rloc = np.array({ent_rloc_list})
""")