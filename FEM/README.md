## Strong Form of the 1D Burgers' Equation

The 1D Burgers' equation with viscosity and a source term is given by:

$$
\frac{\partial u(x,t)}{\partial t} + u(x,t) \frac{\partial u(x,t)}{\partial x} - \nu \frac{\partial^2 u(x,t)}{\partial x^2} = f(x,t), \ \ \ \forall x \in [0, L], \ \forall t \in [0, T]
$$

where:
- \( u(x,t) \) is the velocity field.
- \( \nu \) is the kinematic viscosity.
- \( f(x,t) \) is the source term.
- \( L \) is the length of the spatial domain.
- \( T \) is the final time.

### Boundary and Initial Conditions

The equation is subject to the following boundary and initial conditions:

$$
\begin{aligned}
&u(0, t) = u_{\text{left}}(t)=\mu_1, &&\forall t \in [0, T], \\
&u(x, 0) = u_{\text{initial}}(x)=1, &&\forall x \in [0, L].
\end{aligned}
$$

### Source Term

In the specific case we are considering, the source term \( f(x,t) \) is given by:

$$
f(x,t) = 0.02 e^{\mu_2 x}
$$

where \( \mu_2 \) is a parameter that defines the spatial variation of the source term.
