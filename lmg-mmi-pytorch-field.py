import torch
import math
import numpy as np
import time
from matplotlib import pyplot as plt
from tqdm import tqdm

n = 10  # system size
h_field = 0.5  # field strength
max_iter = 100_000  # max number of iterations per data point
tol = 1e-12  # intended accuracy
mutual_information_datafile = 'my_field_data.json'  # to store mutual information data
save_mutual_information = True
tabGamma = np.arange(0, 5.01, 0.1).tolist()  # Gamma/gamma values
pbar_update_step = 200  # update progress bar every pbar_update_step

output_file_name = f"h={h_field:.2f}_data.py"
half_n = n // 2
clamp_min = tol**2
tol_digits = 3 + math.ceil(-math.log10(tol))

pbar = tqdm(total=len(tabGamma) * max_iter, smoothing=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
cdtype = torch.complex128

n_items = (n+1)*(n+2)*(n+3) // 6

DEFAULT_PLOT_CONFIG = {  # for the live progress plot
    'xlabel': 'Iteration',
    'ylabel': 'Loss',
    'grid': True,
    'max_plot_loss': 2,  # Only update plot if loss < this value.
    'prediction_fmt': 'k--',  # Format for the predicted loss curve.
    'grad_fmt': 'g--',  # Format for the gradient norm curve.
    'loss_line_kwargs': {'alpha': 1, 'lw': 2},  # Style for the main loss curve.
    'tight_layout': True,
}


class LivePlotter:
    # plots "loss" (error estimate) during the computation
    def __init__(self, ax=None, **config):
        self.cfg = DEFAULT_PLOT_CONFIG.copy()
        self.cfg.update(config)
        self.max_plot_loss = self.cfg['max_plot_loss']

        if ax is None:
            self.fig = plt.figure(dpi=300)
            self.ax = self.fig.subplots()
        else:
            self.ax = ax
            self.fig = ax.figure

        self.ax.set_xlabel(self.cfg['xlabel'])
        self.ax.set_ylabel(self.cfg['ylabel'])
        self.ax.set_xlim(0, max_iter)
        self.ax.set_ylim(tol, 1e2)

        if self.cfg['grid']:
            self.ax.grid(True)
        if self.cfg['tight_layout']:
            self.fig.tight_layout()

        self.prediction_line, = self.ax.semilogy([], [], self.cfg['prediction_fmt'], zorder=-1, alpha=0.5)
        self.grad_line, = self.ax.semilogy([], [], self.cfg['grad_fmt'], zorder=-1, alpha=0.5)
        self.loss_line = None
        self.reset(True)
        self.fig.show()

    def reset(self, initial=False):
        """Secondary (re-)initialization: clear data lists and reset curves."""
        self.i_list = []
        self.loss_list = []
        self.grad_list = []
        self.time_list = []
        self.prediction_line.set_data([], [])
        self.grad_line.set_data([], [])
        if not initial:
            self.loss_line, = self.ax.semilogy([], [], **self.cfg['loss_line_kwargs'])

    def update(self, i, loss, grad_norm, num_iter=None, tol=None):
        """
        Regular update: append new data and update all curves.

        Parameters:
          i        : current iteration number.
          loss     : current loss value.
          grad_norm: current gradient norm.
          num_iter : total number of iterations (used for prediction curve).
          tol      : tolerance (used for prediction curve).
        """
        # if loss < self.max_plot_loss:
        self.i_list.append(i+1)
        self.loss_list.append(loss)
        self.grad_list.append(grad_norm)
        self.time_list.append(time.time())
        self.ax.set_xlim(0, num_iter)
        self.ax.set_ylim(tol, 1.3*np.max(self.loss_list))

        # Update the predicted loss curve when enough data are available.
        should_plot = len(self.i_list) >= 10 and num_iter is not None and tol is not None
        if should_plot:
            index_0 = (len(self.i_list) * 4) // 5
            i0, i1 = self.i_list[index_0], self.i_list[-1]
            l0, l1 = self.loss_list[index_0], self.loss_list[-1]
            t0, t1 = self.time_list[index_0], self.time_list[-1]
            if l0 > l1:
                a = (np.log(l1) - np.log(l0)) / (i1 - i0)
                b = (i1*np.log(l0) - i0*np.log(l1)) / (i1 - i0)
                it = min(num_iter, (np.log(tol) - b)/a)
                xx = np.linspace(i0, it, 100)
                yy = np.exp(a*xx + b)
                self.prediction_line.set_data(xx, yy)

        self.loss_line.set_data(self.i_list, self.loss_list)
        self.grad_line.set_data(self.i_list, self.grad_list)

        self.ax.relim()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
        return np.log10(l0 / l1) * 3600 / (t1 - t0) if should_plot else 0

    def save(self):
        self.fig.savefig("last_loss_plot.pdf")


# ent: the entropy function
def ent(x0):
    x = max(clamp_min, min(1-clamp_min, x0))
    return -((1 - x) / 2) * math.log((1 - x) / 2) - ((1 + x) / 2) * math.log((1 + x) / 2)


# Helper: build a sparse identity matrix
def sparse_eye(N, dtype, device):
    indices = torch.stack([torch.arange(N, device=device), torch.arange(N, device=device)])
    values = torch.ones(N, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices, values, (N, N), dtype=dtype, device=device)


def sparse_block_diag(*sparse_matrices):
    """
    Construct a block-diagonal sparse COO matrix from the given sparse COO matrices.

    Args:
        *sparse_matrices: A variable number of torch.sparse_coo_tensor objects.

    Returns:
        A new torch.sparse_coo_tensor representing the block-diagonal matrix.
    """
    if not sparse_matrices:
        raise ValueError("At least one sparse matrix is required.")

    device = sparse_matrices[0].device
    dtype = sparse_matrices[0].dtype
    total_rows = 0
    total_cols = 0
    block_indices = []
    block_values = []

    for sp in sparse_matrices:
        if not sp.is_sparse:
            raise ValueError("All input matrices must be sparse COO tensors.")
        rows, cols = sp.size()
        indices = sp._indices()
        values = sp._values()
        offset = torch.tensor([[total_rows], [total_cols]], device=device)
        shifted_indices = indices + offset
        block_indices.append(shifted_indices)
        block_values.append(values)
        total_rows += rows
        total_cols += cols

    all_indices = torch.cat(block_indices, dim=1)
    all_values = torch.cat(block_values)
    new_size = (total_rows, total_cols)
    return torch.sparse_coo_tensor(all_indices, all_values, size=new_size, device=device, dtype=dtype)


def sparse_kron(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute the Kronecker product of two sparse COO matrices A and B.

    Args:
        A (torch.Tensor): A sparse COO tensor of size (m, n).
        B (torch.Tensor): A sparse COO tensor of size (p, q).

    Returns:
        torch.Tensor: A sparse COO tensor representing the Kronecker product,
                      of size (m*p, n*q).
    """
    if not (A.is_sparse and B.is_sparse):
        raise ValueError("Both A and B must be sparse COO tensors.")

    A_idx = A._indices()  # shape: (2, nnz_A)
    A_vals = A._values()  # shape: (nnz_A,)
    B_idx = B._indices()  # shape: (2, nnz_B)
    B_vals = B._values()  # shape: (nnz_B,)
    m, n = A.size()
    p, q = B.size()
    kron_row = A_idx[0].unsqueeze(1) * p + B_idx[0].unsqueeze(0)  # shape: (nnz_A, nnz_B)
    kron_col = A_idx[1].unsqueeze(1) * q + B_idx[1].unsqueeze(0)  # shape: (nnz_A, nnz_B)
    new_row = kron_row.reshape(-1)
    new_col = kron_col.reshape(-1)
    new_indices = torch.stack([new_row, new_col], dim=0)  # shape: (2, nnz_A * nnz_B)
    new_vals = (A_vals.unsqueeze(1) * B_vals.unsqueeze(0)).reshape(-1)
    new_size = (m * p, n * q)

    return torch.sparse_coo_tensor(new_indices, new_vals, size=new_size, device=A.device, dtype=A.dtype)


def cg_solve(M, A_func, b, x0=None, tol_cg=tol, max_cg=1000, pbar=None, plotter=None, k0=0, g=0, pbar_update_step=pbar_update_step):
    """
    A simple conjugate gradient solver for Ax = b,
    where A_func(x) returns A*x.
    """
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0
    r = b - A_func(x)
    p = r.clone()
    rsold = torch.dot(r.conj(), r).real

    for i in range(max_cg):
        Ap = A_func(p)
        alpha = rsold / (torch.dot(p.conj(), Ap).real + 1e-20)
        x = x + alpha * p
        if (i+1) % 50 == 0:
            r = b - A_func(x)
        else:
            r = r - alpha * Ap
        rsnew = torch.dot(r.conj(), r).real
        p = r + (rsnew / rsold) * p
        rsold = rsnew

        del Ap

        if pbar is not None and (i+1) % pbar_update_step == 0:
            res = ((M @ (x/x.norm())).norm() ** 2).item()
            if plotter is not None:
                hourly_loss = plotter.update(k0 + i + 1, res, None, max_iter, tol)
                if pbar is not None:
                    pbar.set_description(f"G/g: {g.real:.3f}, Loss: {res:.{tol_digits}f}, L/h: {hourly_loss:.3f}")
                    pbar.update(pbar_update_step)
            if res < tol_cg:
                pbar.update(max_cg - i)
                pbar.write(f"Inverse iteration converged (residual {res:.2e}) at iter {i+1:_}/{max_cg:_}")
                break

    return x


def null_vector_inverse_iteration_simple(
        M, tol=tol, max_iter=max_iter, v0=None, pbar=pbar, plotter=None, g=0):
    device = M.device
    dtype = M.dtype
    n = M.shape[1]

    # Build M* as a sparse tensor.
    indices_MH = M.indices()[[1, 0]]
    values_MH = M.values().conj()
    M_H_sparse = torch.sparse_coo_tensor(
        indices_MH, values_MH, size=(n, n), dtype=dtype, device=device
    ).coalesce()

    # Define A(v) = M_H*(M*v)
    def A_mv(v):
        return M_H_sparse @ (M @ v)

    # Initialize starting vector.
    if v0 is None:
        v = torch.randn(n, dtype=dtype, device=device)
    else:
        v = v0.clone()
    v = v / v.norm()
    v_zeros = torch.zeros_like(v)

    if plotter is not None:
        plotter.reset()

    v = cg_solve(M, A_mv, v_zeros, v, tol_cg=tol, max_cg=max_iter, pbar=pbar, plotter=plotter, k0=k, g=g)
    del M_H_sparse
    v = v / v.norm()
    res = ((M @ v).norm()**2).item()

    if res > tol:
        if pbar is not None:
            pbar.write(f"Inverse iteration finished without full convergence (residual {res:.2e}).")

    return v


live_fig = plt.figure(dpi=300)
live_ax = live_fig.subplots()
plotter = LivePlotter(live_ax)

tabdim = []
for j in range(half_n + 1):
    num = math.factorial(n)
    den = math.factorial(half_n + j + 1) * math.factorial(half_n - j)
    tabdim.append((2 * j + 1) * num / den)


def Ap(j, m):
    return ((j - m) * (j + m + 1))**0.5

def Am(j, m):
    return ((j + m) * (j - m + 1))**0.5

def Bp(j, m):
    return ((j - m) * (j - m - 1))**0.5

def Bm(j, m):
    return -((j + m) * (j + m + 1))**0.5

def Dp(j, m):
    return -((j + m + 1) * (j + m + 2))**0.5

def Dm(j, m):
    return ((j - m + 1) * (j - m + 2))**0.5


m1_vals_list = []
for k in range(0, half_n + 1):
    arr = np.arange(-k, k + 1)
    J, I = np.meshgrid(arr, arr, indexing='ij')
    m1_vals_list.append(n - I - J)

m1_vals = -np.concatenate([block.ravel() for block in m1_vals_list])
m1_size = m1_vals.size
indices_m1 = torch.from_numpy(np.stack([np.arange(m1_size), np.arange(m1_size)])).to(torch.int64)
values_m1 = torch.tensor(m1_vals, dtype=dtype)
m1 = torch.sparse_coo_tensor(indices_m1, values_m1, size=(m1_size, m1_size)).to(device)


def accumulate_entries(k_iter, i_bounds_func, j_bounds_func, row_func, col_func, val_func, desc):
    rows_list = []
    cols_list = []
    vals_list = []

    for k in k_iter:
        i_min, i_max = i_bounds_func(k)
        j_min, j_max = j_bounds_func(k)
        i_vals = np.arange(i_min, i_max + 1)
        j_vals = np.arange(j_min, j_max + 1)
        J, I = np.meshgrid(j_vals, i_vals, indexing='ij')
        rows = row_func(k, I, J)
        cols = col_func(k, I, J)
        vals = val_func(k, I, J)
        rows_list.append(np.rint(rows).astype(np.int64).ravel())
        cols_list.append(np.rint(cols).astype(np.int64).ravel())
        vals_list.append(vals.ravel())

    return np.concatenate(rows_list), np.concatenate(cols_list), np.concatenate(vals_list)


size_m = int((n + 1) * (n + 2) * (n + 3) / 6)
rows2, cols2, vals2 = accumulate_entries(
    k_iter=range(0, half_n),                      # k = 0,...,half_n-1
    i_bounds_func=lambda k: (-k, k),              # i in -k,...,k
    j_bounds_func=lambda k: (-k, k),              # j in -k,...,k
    row_func=lambda k, I, J: ((4 * k**2 - 1) * k / 3) + ((k + J) * (2 * k + 1)) + I + k,
    col_func=lambda k, I, J: ((4 * (k + 1)**2 - 1) * (k + 1) / 3) + ((k + J) * (2 * k + 3)) + I + k,
    val_func=lambda k, I, J: Bp(k + 1, I - 1) * Bp(k + 1, J - 1) * (half_n + k + 2) / ((k + 1) * (2 * k + 3)),
    desc='m2',
)
indices_m2 = torch.from_numpy(np.stack([rows2, cols2])).to(torch.int64)
values_m2 = torch.tensor(vals2, dtype=dtype)
m2 = torch.sparse_coo_tensor(indices_m2, values_m2, size=(size_m, size_m)).to(device)

rows3, cols3, vals3 = accumulate_entries(
    k_iter=range(1, half_n + 1),                  # k = 1,...,half_n
    i_bounds_func=lambda k: (-k + 1, k),          # i in -k+1,...,k
    j_bounds_func=lambda k: (-k + 1, k),          # j in -k+1,...,k
    row_func=lambda k, I, J: ((4 * k**2 - 1) * k / 3) + ((k + J) * (2 * k + 1)) + I + k,
    col_func=lambda k, I, J: ((4 * k**2 - 1) * k / 3) + ((k + J - 1) * (2 * k + 1)) + I + k - 1,
    val_func=lambda k, I, J: Ap(k, I - 1) * Ap(k, J - 1) * (half_n + 1) / ((k + 1) * k),
    desc='m3'
)
indices_m3 = torch.from_numpy(np.stack([rows3, cols3])).to(torch.int64)
values_m3 = torch.tensor(vals3, dtype=dtype)
m3 = torch.sparse_coo_tensor(indices_m3, values_m3, size=(size_m, size_m)).to(device)

rows4, cols4, vals4 = accumulate_entries(
    k_iter=range(1, half_n + 1),                  # k = 1,...,half_n
    i_bounds_func=lambda k: (-k + 2, k),          # i in -k+2,...,k
    j_bounds_func=lambda k: (-k + 2, k),          # j in -k+2,...,k
    row_func=lambda k, I, J: ((4 * k**2 - 1) * k / 3) + ((k + J) * (2 * k + 1)) + I + k,
    col_func=lambda k, I, J: ((4 * (k - 1)**2 - 1) * (k - 1) / 3) + ((k + J - 2) * (2 * k - 1)) + I + k - 2,
    val_func=lambda k, I, J: Dp(k - 1, I - 1) * Dp(k - 1, J - 1) * (half_n - k + 1) / ((2 * k - 1) * k),
    desc='m4'
)
indices_m4 = torch.from_numpy(np.stack([rows4, cols4])).to(torch.int64)
values_m4 = torch.tensor(vals4, dtype=dtype)
m4 = torch.sparse_coo_tensor(indices_m4, values_m4, size=(size_m, size_m)).to(device)

tab = []
for j in range(half_n + 1):
    current = []
    # Note: range(1,2*j+1) is empty if j==0.
    for i in range(1, 2 * j + 1):
        current.append(math.sqrt(2 * i * (j + 1) - i * (i + 1)))
    tab.append(current)

sx = [torch.tensor([[0.0]], device=device)]
for idx in range(1, len(tab)):
    size_mat = 2 * (idx + 1) - 1
    m = torch.zeros((size_mat, size_mat), device=device, dtype=dtype)
    off_diag = torch.tensor(tab[idx], device=device, dtype=dtype) / 2.0
    for i in range(size_mat - 1):
        m[i, i + 1] = off_diag[i]
        m[i + 1, i] = off_diag[i]
    sx.append(m)

sy = []
sy.append(torch.tensor([[0.0]], device=device, dtype=cdtype))

for idx in range(1, len(tab)):
    size_mat = 2 * (idx + 1) - 1
    m = torch.zeros((size_mat, size_mat), device=device, dtype=cdtype)
    off_diag = torch.tensor(tab[idx], device=device, dtype=cdtype) / 2.0
    for i in range(size_mat - 1):
        m[i, i + 1] = -1j * off_diag[i]
        m[i + 1, i] = 1j * off_diag[i]
    sy.append(m)

tab2 = []
for j in range(half_n + 1):
    tab2.append([float(i) for i in range(-j, j + 1)])

sz = []
for arr in tab2:
    diag_tensor = torch.tensor(arr, device=device, dtype=dtype)
    sz.append(torch.diag(diag_tensor))

opm = []
for sxi, syi in zip(sx, sy):
    opm.append((sxi + 1j * syi).real)

tabhamint = [((sx_j @ sx_j) + (sy_j @ sy_j)).real / n for sx_j, sy_j in zip(sx, sy)]

tabliovhamint = []
tabliovhamx = []
tabliovglobdis = []

for j in range(len(sx)):
    eye = sparse_eye(2 * (j + 1) - 1, device=device, dtype=dtype)

    t0 = tabhamint[j].to_sparse_coo()
    tabliovhamint.append(-1j * (sparse_kron(eye, t0) - sparse_kron(t0.mT, eye)))

    t1 = sx[j].to_sparse_coo()
    tabliovhamx.append(-1j * (sparse_kron(eye, t1) - sparse_kron(t1.mT, eye)))

    t2 = (opm[j].T.conj() @ opm[j]).to_sparse_coo()
    k2 = 2 * sparse_kron(torch.conj(opm[j]).to_sparse_coo(), opm[j].to_sparse_coo())
    tabliovglobdis.append((k2 - sparse_kron(eye, t2) - sparse_kron(t2.mT, eye)) / (2 * n))

liovhamint = sparse_block_diag(*tabliovhamint)
liovhamx = sparse_block_diag(*tabliovhamx)
liovglobdis = sparse_block_diag(*tabliovglobdis)

eigvecs = []
v = None

for gamma in tabGamma[::-1]:
    v_cached = False
    liov_gamma =  (
            3.0 * liovhamint + h_field * liovhamx + gamma * liovglobdis + 0.5 * (m1 + m2 + m3 + m4)
    ).detach().coalesce()
    if gamma == tabGamma[0]:
        pbar.write(f"\nRank:     {liov_gamma.shape[0]:.1e} ({liov_gamma.shape[0]:d})")
        pbar.write(f"Num. el.: {liov_gamma.numel():.1e} ({liov_gamma.numel():d})")
        pbar.write(f"Non-zero: {liov_gamma._nnz():.1e} ({liov_gamma._nnz():d})")
        pbar.write(f"Density:  {liov_gamma._nnz()/liov_gamma.numel():.1e}")

    if v is None:
        v = torch.randn(liov_gamma.shape[1], dtype=liov_gamma.dtype, device=liov_gamma.device)
        v = v / v.norm()

    v = null_vector_inverse_iteration_simple(liov_gamma, v0=v.detach().clone(), plotter=plotter, g=gamma)
    eigvecs.append(v.detach().cpu())
    del liov_gamma
    torch.cuda.empty_cache()

pbar.close()
plotter.save()
eigvecs = eigvecs[::-1]

indices_list = []
for k in range(half_n + 1):
    size = 2 * k + 1
    base = int((4 * k * k - 1) * k / 3) if k > 0 else 0
    i = torch.arange(size, device=device)
    j = torch.arange(size, device=device)
    grid_i, grid_j = torch.meshgrid(i, j, indexing='ij')
    indices = (base + grid_i * size + grid_j).to(device=device)
    indices_list.append(indices)

denmatun = []
for v in eigvecs:
    v = v.to(indices_list[0].device)
    blocks = [v[idx] for idx in indices_list]
    denmatun.append(blocks)

denmat = []
for denmatun_m in denmatun:
    total_trace = sum(torch.trace(b) for b in denmatun_m)
    denmat.append([b / total_trace for b in denmatun_m])

taboc = [
    [torch.clamp(torch.trace(denmat_m_j).real, min=clamp_min) for denmat_m_j in denmat_m]
    for denmat_m in denmat
]

tabdenmat = [
    [denmat_m_j / taboc_m_j for denmat_m_j, taboc_m_j in zip(denmat_m, taboc_m)]
    for denmat_m, taboc_m in zip(denmat, taboc)
]

entr = []
for m, taboc_m in enumerate(taboc):
    s = 0.0
    for j, taboc_m_j in enumerate(taboc_m):
        eigvals = torch.linalg.eigvals(tabdenmat[m][j]).real
        s += taboc_m_j * (
            - torch.log(taboc_m_j)
            - torch.sum(eigvals * torch.log(eigvals.abs() + 1e-16))
            + math.log(tabdim[j])
        )
    entr.append(s.real)

av = [[], [], []]
for denmat_m in denmat:
    sums = [0.0, 0.0, 0.0]
    for dm, sx_, sy_, sz_ in zip(denmat_m, sx, sy, sz):
        for j, s in enumerate((sx_, sy_, sz_)):
            sums[j] += torch.trace(dm @ s.to(dtype=cdtype)).real
    for j in range(3):
        av[j].append(2.0 / n * sums[j])

mutinf = [
    (tabGamma[m], (ent(math.sqrt(sum(av[j][m] ** 2 for j in range(3)))) - entr[m] / n).item())
    for m in tqdm(range(len(tabGamma)), desc='Building mutinf')
]
s_tot_per_site = [(s / n).item() for s in entr]
mx_list, my_list, mz_list = torch.tensor(av).tolist()
s_locals = [ent(math.sqrt(mx_list[i]**2 + my_list[i]**2 + mz_list[i]**2)) for i in range(len(tabGamma))]
magnetizations = list(zip(mx_list, my_list, mz_list))

output_blurb = f"""\
    {n}: {{
        'g_inf_list': np.array({mutinf}),
        's_tot_per_site': np.array({s_tot_per_site}),
        's_locals': np.array({s_locals}),
        'magnetizations': np.array({magnetizations}),
    }},
"""

print(output_blurb)

with open(output_file_name, "a") as f:
    f.write(output_blurb)
