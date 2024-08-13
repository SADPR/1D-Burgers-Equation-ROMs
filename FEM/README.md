## Strong Form of the 1D Burgers' Equation

The 1D Burgers' equation with viscosity and a source term is given by:

$$
\frac{\partial u(x,t)}{\partial t} + u(x,t) \frac{\partial u(x,t)}{\partial x} - \nu \frac{\partial^2 u(x,t)}{\partial x^2} = f(x,t), \ \ \ \forall x \in [0, L], \ \forall t \in [0, T]
$$

where:
- $u(x,t)$ is the velocity field.
- $\nu$ is the kinematic viscosity.
- $f(x,t)$ is the source term.
- $L$ is the length of the spatial domain.
- $T$ is the final time.

### Boundary and Initial Conditions

The equation is subject to the following boundary and initial conditions:

$$
\begin{aligned}
&u(0, t) = u_{\text{left}}(t)=\mu_1, &&\forall t \in [0, T], \\
&u(x, 0) = u_{\text{initial}}(x)=1, &&\forall x \in [0, L].
\end{aligned}
$$

### Source Term

In the specific case we are considering, the source term $f(x,t)$ is given by:

$$
f(x,t) = 0.02 e^{\mu_2 x}
$$

where $\mu_2$ is a parameter that defines the spatial variation of the source term.

## Weak Form of the 1D Burgers' Equation

To derive the weak form of the 1D Burgers' equation, we start with the strong form:

$$
\frac{\partial u(x,t)}{\partial t} + u(x,t) \frac{\partial u(x,t)}{\partial x} = \nu \frac{\partial^2 u(x,t)}{\partial x^2} + f(x,t)
$$

We multiply by a test function $v(x)$ and integrate over the domain $[0, L]$:

$$
\int_0^L \left( \frac{\partial u(x,t)}{\partial t} + u(x,t) \frac{\partial u(x,t)}{\partial x} - \nu \frac{\partial^2 u(x,t)}{\partial x^2} - f(x,t) \right) v(x)  dx = 0
$$

We handle the second-order derivative term $- \nu \frac{\partial^2 u}{\partial x^2}$ using integration by parts:

$$
-\nu \int_0^L \frac{\partial^2 u}{\partial x^2} v  dx = \left[ - \nu \frac{\partial u}{\partial x} v \right]_0^L + \int_0^L \nu \frac{\partial u}{\partial x} \frac{\partial v}{\partial x}  dx
$$

Assuming homogeneous Dirichlet boundary conditions (i.e., $u = 0$ on $\partial \Omega$), the boundary term vanishes:

$$
\left. \nu \frac{\partial u}{\partial x} v \right|_0^L = 0
$$

Thus, we obtain:

$$
\int_0^L \left( \frac{\partial u}{\partial t} v + u \frac{\partial u}{\partial x} v + \nu \frac{\partial u}{\partial x} \frac{\partial v}{\partial x} - f(x,t) v \right) dx = 0
$$

The weak form of the 1D Burgers' equation with viscosity and a source term is:

$$
\int_0^L \left( \frac{\partial u}{\partial t} v + u \frac{\partial u}{\partial x} v + \nu \frac{\partial u}{\partial x} \frac{\partial v}{\partial x} \right) dx = \int_0^L f(x,t) v  dx
$$

This equation must be satisfied for all test functions $v(x)$ in the appropriate function space.

## Finite Element Discretization

### 1. Discretizing the Domain

Discretize the domain $[0, L]$ into $E$ finite elements with nodes $x_i$, where $i = 1, 2, \ldots, E$.

### 2. Approximating the Solution and Test Functions

Approximate the solution $u(x,t)$ and the test function $v(x)$ using finite element basis functions $N_i(x)$:

$$
u(x,t) \approx \sum_{j=1}^{E} U_j(t) N_j(x), \quad v(x) = N_i(x)
$$

Here, $U_j(t)$ are the time-dependent coefficients, and $N_j(x)$ are the shape functions.

### 3. Substituting into the Weak Form

Substitute these approximations into the weak form and simplify:

$$
\sum_{j=1}^{E} \frac{dU_j(t)}{dt} \int_0^L N_j(x) N_i(x) \, dx + \sum_{j=1}^{E} U_j(t) \int_0^L \left( \sum_{k=1}^{E} U_k(t) N_k(x) \right) \frac{\partial N_j(x)}{\partial x} N_i(x) \, dx + \nu \sum_{j=1}^{E} U_j(t) \int_0^L \frac{\partial N_j(x)}{\partial x} \frac{\partial N_i(x)}{\partial x} \, dx = \int_0^L f(x,t) N_i(x) \, dx
$$

### 4. Defining the Matrices

This can be rewritten in matrix form:

- **Mass Matrix $\mathbf{M}$:**

$$
\mathbf{M}_{ij} = \int_0^L N_i(x) N_j(x) \, dx
$$

- **Convection Term $\mathbf{C}(\mathbf{U})$:**

$$
\mathbf{C}_{ij}(\mathbf{U}) = \sum_{k=1}^{E} U_k(t) \int_0^L N_k(x) \frac{\partial N_j(x)}{\partial x} N_i(x) \, dx
$$

- **Diffusion Matrix $\mathbf{K}$:**

$$
\mathbf{K}_{ij} = \nu \int_0^L \frac{\partial N_i(x)}{\partial x} \frac{\partial N_j(x)}{\partial x} \, dx
$$

- **Load Vector $\mathbf{F}$:**

$$
\mathbf{F}_i(t) = \int_0^L f(x,t) N_i(x) \, dx
$$

### 5. Discretized System of Equations

The overall system of equations is:

$$
\mathbf{M} \frac{d\mathbf{U}(t)}{dt} + \mathbf{C}(\mathbf{U})\mathbf{U} + \mathbf{K} \mathbf{U} = \mathbf{F}(t)
$$

where:
- $\mathbf{U}(t)$ is the vector of unknowns at the nodes.

### 6. Time Discretization

To solve this system over time, you apply a time discretization method such as the implicit Euler scheme:

$$
\mathbf{M} \frac{\mathbf{U}^{n+1} - \mathbf{U}^n}{\Delta t} + \mathbf{C}(\mathbf{U}^{n+1}) \mathbf{U}^{n+1} + \mathbf{K} \mathbf{U}^{n+1} = \mathbf{F}^{n+1}
$$

This forms a nonlinear system at each time step, which can be solved iteratively.
