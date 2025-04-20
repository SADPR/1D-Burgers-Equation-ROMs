import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import RegularGridInterpolator

# ==== Configuration Toggle ====
use_physics = True  # Set to True to include PDE, IC, BC losses

# ==== PINN Model ====
class PINN_Burgers1D(nn.Module):
    def __init__(self, hidden_layers=4, hidden_neurons=50):
        super(PINN_Burgers1D, self).__init__()
        layers = [nn.Linear(2, hidden_neurons), nn.Tanh()]
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_neurons, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

# ==== Residual Computation ====
def compute_residual(model, x, t, mu2):
    x.requires_grad = True
    t.requires_grad = True
    u = model(x, t)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    f = 0.02 * torch.exp(mu2 * x)
    residual = u_t + u * u_x - f
    return residual

# ==== Data Generation ====
def generate_training_points(L, T, N_ic, N_bc, N_f):
    x_ic = torch.linspace(0, L, N_ic).view(-1, 1)
    t_ic = torch.zeros_like(x_ic)
    u_ic = torch.ones_like(x_ic)

    t_bc = torch.linspace(0, T, N_bc).view(-1, 1)
    x_bc = torch.zeros_like(t_bc)

    x_f = torch.rand(N_f, 1) * L
    t_f = torch.rand(N_f, 1) * T

    return x_ic, t_ic, u_ic, x_bc, t_bc, x_f, t_f

# ==== FD Interpolator ====
def load_fd_interpolator(fd_path, L, T, N, n_steps):
    U_FD = np.load(fd_path)
    x_fd = np.linspace(0, L, N)
    t_fd = np.linspace(0, T, n_steps + 1)
    return RegularGridInterpolator((x_fd, t_fd), U_FD, bounds_error=False, fill_value=None)

# ==== Sample FD Points ====
def sample_fd_supervision_points(L, T, N_fd_samples):
    x = torch.rand(N_fd_samples, 1) * L
    t = torch.rand(N_fd_samples, 1) * T
    return x, t

# ==== Training with Adam ====
def train_adam(model, optimizer, epochs, data, mu1, mu2, L, T, fd_interp, N_fd, n_steps, lambda_fd):
    x_ic, t_ic, u_ic, x_bc, t_bc, x_f, t_f = data
    start = time.time()

    # === NEW: Generate full FD grid points ===
    x_lin = np.linspace(0, L, N_fd)
    t_lin = np.linspace(0, T, n_steps + 1)
    x_grid, t_grid = np.meshgrid(x_lin, t_lin, indexing='ij')
    xt_fd_np = np.stack([x_grid.flatten(), t_grid.flatten()], axis=-1)
    u_fd_true = torch.tensor(fd_interp(xt_fd_np), dtype=torch.float32).view(-1, 1)

    # Convert to tensors for training
    x_fd = torch.tensor(xt_fd_np[:, 0:1], dtype=torch.float32)
    t_fd = torch.tensor(xt_fd_np[:, 1:2], dtype=torch.float32)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        loss_ic = loss_bc = loss_pde = torch.tensor(0.0)

        if use_physics:
            u_pred_ic = model(x_ic, t_ic)
            loss_ic = torch.mean((u_pred_ic - u_ic)**2)

            u_pred_bc = model(x_bc, t_bc)
            loss_bc = torch.mean((u_pred_bc - mu1)**2)

            res_f = compute_residual(model, x_f, t_f, mu2)
            loss_pde = torch.mean(res_f**2)

        # === NEW: Use all FD points ===
        u_fd_pred = model(x_fd, t_fd)
        loss_fd = torch.mean((u_fd_pred - u_fd_true)**2)

        loss = lambda_fd * loss_fd
        if use_physics:
            loss += loss_pde + 10 * loss_ic + 10 * loss_bc

        loss.backward()
        optimizer.step()

        if epoch % 1 == 0:
            print(f"[Adam] Epoch {epoch:4d}: Total={loss.item():.6f} | PDE={loss_pde.item():.6f} | IC={loss_ic.item():.6f} | BC={loss_bc.item():.6f} | FD={loss_fd.item():.6f}")

    print(f"[Adam] Done. Time: {time.time() - start:.2f}s")


# ==== Training with LBFGS ====
def train_lbfgs(model, data, mu1, mu2, L, T, fd_interp, N_fd, n_steps, lambda_fd):
    x_ic, t_ic, u_ic, x_bc, t_bc, x_f, t_f = data

    # === NEW: Precompute full FD mesh ===
    x_lin = np.linspace(0, L, N_fd)
    t_lin = np.linspace(0, T, n_steps + 1)
    x_grid, t_grid = np.meshgrid(x_lin, t_lin, indexing='ij')
    xt_fd_np = np.stack([x_grid.flatten(), t_grid.flatten()], axis=-1)
    u_fd_true = torch.tensor(fd_interp(xt_fd_np), dtype=torch.float32).view(-1, 1)

    x_fd = torch.tensor(xt_fd_np[:, 0:1], dtype=torch.float32)
    t_fd = torch.tensor(xt_fd_np[:, 1:2], dtype=torch.float32)

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=1000,
        max_eval=1000,
        history_size=50,
        line_search_fn="strong_wolfe"
    )

    print("[LBFGS] Starting...")
    start = time.time()
    iteration = [0]

    def closure():
        optimizer_lbfgs.zero_grad()

        loss_ic = loss_bc = loss_pde = torch.tensor(0.0)

        if use_physics:
            u_pred_ic = model(x_ic, t_ic)
            loss_ic = torch.mean((u_pred_ic - u_ic)**2)

            u_pred_bc = model(x_bc, t_bc)
            loss_bc = torch.mean((u_pred_bc - mu1)**2)

            res_f = compute_residual(model, x_f, t_f, mu2)
            loss_pde = torch.mean(res_f**2)

        # === NEW: Use all FD grid points ===
        u_fd_pred = model(x_fd, t_fd)
        loss_fd = torch.mean((u_fd_pred - u_fd_true)**2)

        total_loss = lambda_fd * loss_fd
        if use_physics:
            total_loss += loss_pde + 10 * loss_ic + 10 * loss_bc

        total_loss.backward()

        if iteration[0] % 10 == 0:
            print(f"[LBFGS] Iter {iteration[0]:04d}: Total={total_loss.item():.6f} | PDE={loss_pde.item():.6f} | IC={loss_ic.item():.6f} | BC={loss_bc.item():.6f} | FD={loss_fd.item():.6f}")
        iteration[0] += 1
        return total_loss

    optimizer_lbfgs.step(closure)
    print(f"[LBFGS] Done in {iteration[0]} iterations. Time: {time.time() - start:.2f}s")


# ==== Plotting ====
def plot_prediction(model, L, T):
    x = torch.linspace(0, L, 200).view(-1, 1)
    t = torch.full_like(x, T)
    with torch.no_grad():
        u_pred = model(x, t).cpu().numpy()

    plt.plot(x.cpu().numpy(), u_pred, label=f"t={T}")
    plt.xlabel("x")
    plt.ylabel("u(x,T)")
    plt.title("PINN Prediction at Final Time")
    plt.grid()
    plt.legend()
    plt.show()

# ==== Main ====
if __name__ == "__main__":
    L = 100.0
    T = 25.0
    mu1 = torch.tensor(4.25)
    mu2 = torch.tensor(0.015)

    N_ic = 100
    N_bc = 100
    N_f = 10000
    epochs = 2000
    lr = 1e-3

    N_fd = 512
    n_steps = 500
    N_fd_samples = 2000
    lambda_fd = 5.0

    fd_path = "../../FD/fd_training_data/fd_simulation_mu1_4.250_mu2_0.0150.npy"
    fd_interp = load_fd_interpolator(fd_path, L, T, N_fd, n_steps)

    data = generate_training_points(L, T, N_ic, N_bc, N_f)

    model = PINN_Burgers1D()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_adam(model, optimizer, epochs, data, mu1, mu2, L, T, fd_interp, N_fd, n_steps, lambda_fd)
    train_lbfgs(model, data, mu1, mu2, L, T, fd_interp, N_fd, n_steps, lambda_fd)

    plot_prediction(model, L, T)

