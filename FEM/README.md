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
\frac{\partial u(x,t)}{\partial t} + u(x,t) \frac{\partial u(x,t)}{\partial x} - \nu \frac{\partial^2 u(x,t)}{\partial x^2} = f(x,t)
$$

We multiply by a test function $v(x)$ and integrate over the domain $[0, L]$:

$$
\int_0^L \left( \frac{\partial u(x,t)}{\partial t} + u(x,t) \frac{\partial u(x,t)}{\partial x} - \nu \frac{\partial^2 u(x,t)}{\partial x^2} - f(x,t) \right) v(x) \, dx = 0
$$

We handle the second-order derivative term $- \nu \frac{\partial^2 u}{\partial x^2}$ using integration by parts:

$$
\int_0^L - \nu \frac{\partial^2 u}{\partial x^2} v \, dx = \left[ - \nu \frac{\partial u}{\partial x} v \right]_0^L + \int_0^L \nu \frac{\partial u}{\partial x} \frac{\partial v}{\partial x} \, dx
$$

Assuming natural boundary conditions, the boundary term vanishes:

$$
\left. \nu \frac{\partial u}{\partial x} v \right|_0^L = 0
$$

Thus, we obtain:

$$
\int_0^L \left( \frac{\partial u}{\partial t} v + u \frac{\partial u}{\partial x} v + \nu \frac{\partial u}{\partial x} \frac{\partial v}{\partial x} - f(x,t) v \right) dx = 0
$$

The weak form of the 1D Burgers' equation with viscosity and a source term is:

$$
\int_0^L \left( \frac{\partial u}{\partial t} v + u \frac{\partial u}{\partial x} v + \nu \frac{\partial u}{\partial x} \frac{\partial v}{\partial x} \right) dx = \int_0^L f(x,t) v \, dx
$$

This equation must be satisfied for all test functions $v(x)$ in the appropriate function space.

## Finite Element Method (FEM) Discretization

Given the weak form of the 1D Burgers' equation:

$$
\int_0^L \left( \frac{\partial u}{\partial t} v + u \frac{\partial u}{\partial x} v + \nu \frac{\partial u}{\partial x} \frac{\partial v}{\partial x} - f v \right) \, dx = 0
$$

we will discretize it using the Finite Element Method.

### Choice of Basis Functions

First, we approximate the solution $u(x,t)$ and the test function $v(x)$ using basis functions.

$$
u(x,t) \approx \sum_{j=1}^{N} U_j(t) N_j(x)
$$

where $N_j(x)$ are the basis functions and $U_j(t)$ are the nodal values at time $t$.

Similarly, the test function $v(x)$ can be written as:

$$
v(x) = N_i(x)
$$

where $N_i(x)$ is also a basis function.

### Substitute Approximations into the Weak Form

Substitute these approximations into the weak form:

$$
\int_0^L \left( \frac{\partial}{\partial t} \left( \sum_{j=1}^{N} U_j N_j \right) v + \left( \sum_{j=1}^{N} U_j N_j \right) \frac{\partial}{\partial x} \left( \sum_{k=1}^{N} U_k N_k \right) v + \nu \frac{\partial}{\partial x} \left( \sum_{j=1}^{N} U_j N_j \right) \frac{\partial v}{\partial x} - f v \right) dx = 0
$$

### Expand and Simplify

Now expand and simplify each term:

#### Time Derivative Term:

$$
\int_0^L \frac{\partial}{\partial t} \left( \sum_{j=1}^{N} U_j N_j \right) N_i \, dx = \int_0^L \sum_{j=1}^{N} \frac{d U_j}{d t} N_j N_i \, dx
$$

This can be written as:

$$
\sum_{j=1}^{N} \frac{d U_j}{d t} \int_0^L N_j N_i \, dx
$$

Define the **mass matrix** $M$ as:

$$
M_{ij} = \int_0^L N_i N_j \, dx
$$

Thus, the time derivative term becomes:

$$
\sum_{j=1}^{N} M_{ij} \frac{d U_j}{d t}
$$

#### Convection Term:

$$
\int_0^L \left( \sum_{j=1}^{N} U_j N_j \right) \frac{\partial}{\partial x} \left( \sum_{k=1}^{N} U_k N_k \right) N_i \, dx = \int_0^L \sum_{j=1}^{N} \sum_{k=1}^{N} U_j U_k N_j \frac{\partial N_k}{\partial x} N_i \, dx
$$

This can be written as:

$$
\sum_{j=1}^{N} \sum_{k=1}^{N} U_j U_k \int_0^L N_j \frac{\partial N_k}{\partial x} N_i \, dx
$$

Define the **convection matrix** $C$ as:

$$
C_{ijk} = \int_0^L N_i N_j \frac{\partial N_k}{\partial x} \, dx
$$

Thus, the convection term becomes:

$$
\sum_{j=1}^{N} \sum_{k=1}^{N} U_j U_k C_{ijk}
$$

#### Contraction to Simplify the Tensor

To avoid dealing with a third-order tensor, we perform a contraction by expressing the convection term as:

$$
\mathbf{U}^T \mathbf{C} \mathbf{U}
$$

Where:

$$
\mathbf{C}_{jk} = \int_0^L N_j \frac{\partial N_k}{\partial x} N_i \, dx
$$

#### Diffusion Term:

$$
\int_0^L \nu \frac{\partial}{\partial x} \left( \sum_{j=1}^{N} U_j N_j \right) \frac{\partial N_i}{\partial x} \, dx = \int_0^L \nu \sum_{j=1}^{N} U_j \frac{\partial N_j}{\partial x} \frac{\partial N_i}{\partial x} \, dx
$$

This can be written as:

$$
\nu \sum_{j=1}^{N} U_j \int_0^L \frac{\partial N_j}{\partial x} \frac{\partial N_i}{\partial x} \, dx
$$

Define the **diffusion matrix** $K$ as:

$$
K_{ij} = \int_0^L \nu \frac{\partial N_i}{\partial x} \frac{\partial N_j}{\partial x} \, dx
$$

Thus, the diffusion term becomes:

$$
\nu \sum_{j=1}^{N} K_{ij} U_j
$$

#### Source Term:

$$
\int_0^L f N_i \, dx
$$

Define the **load vector** \( F \) as:

$$
F_i = \int_0^L f N_i \, dx
$$

### Discretized Weak Form

Combining these terms, we obtain the discretized weak form:

$$
\sum_{j=1}^{N} M_{ij} \frac{d U_j}{d t} + \mathbf{U}^T \mathbf{C} \mathbf{U} + \nu \sum_{j=1}^{N} K_{ij} U_j = F_i
$$

This leads to the following system of equations for each basis function \( N_i \):

$$
\mathbf{M} \frac{d \mathbf{U}}{dt} + \mathbf{C}(\mathbf{U}) \mathbf{U} + \mathbf{K} \mathbf{U} = \mathbf{F}
$$

where:

- \( \mathbf{M} \) is the mass matrix.
- \( \mathbf{C}(\mathbf{U}) \) is the convection matrix, which depends on the solution \( \mathbf{U} \).
- \( \mathbf{K} \) is the diffusion matrix.
- \( \mathbf{F} \) is the load vector.

#### Contraction to Simplify the Tensor

To simplify the handling of the convective term, we use a contraction to avoid the third-order tensor:

1. **Evaluate the Integral:**

$$
C_{ij} = \int_0^L N_i(x) \frac{\partial N_j(x)}{\partial x} \, dx
$$

This matrix \( C_{ij} \) captures the interaction between shape functions and their derivatives over the domain.

2. **Form the Convection Matrix:**

The convection matrix \( C(U) \) is formed by multiplying the precomputed integral matrix \( C_{ij} \) with the current solution vector \( U \):

$$
(C(U))_{ij} = \sum_{k=1}^{N} U_k C_{ijk} = U_j \sum_{k=1}^{N} U_k C_{ijk}
$$
