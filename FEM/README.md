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

