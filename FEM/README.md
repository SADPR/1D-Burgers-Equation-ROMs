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

