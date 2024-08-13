## Mass Matrix in Finite Element Method (FEM)

### 1. Definition of the Mass Matrix

In FEM, the mass matrix $\mathbf{M}$ is defined as:

$$
\mathbf{M}_{ij} = \int_{\Omega} N_i(x) N_j(x) \, dx
$$

where:
- $N_i(x)$ and $N_j(x)$ are the shape functions associated with the $i$th and $j$th nodes, respectively.
- $\Omega$ is the domain of the element over which the integration is performed.

### 2. Transformation to the Reference Element

To simplify the integration process, the physical element $[x_1, x_2]$ is mapped to a reference element $[-1, 1]$ using a linear transformation:

$$
x(\xi) = \frac{(1 - \xi)}{2} x_1 + \frac{(1 + \xi)}{2} x_2
$$

Here, $\xi \in [-1, 1]$ is the coordinate in the reference element.

The integral over the physical element then transforms into an integral over the reference element:

$$
\mathbf{M}_{ij} = \int_{-1}^{1} N_i(\xi) N_j(\xi) \left|\frac{dx}{d\xi}\right| \, d\xi
$$

where $\frac{dx}{d\xi}$ is the Jacobian of the transformation. The Jacobian represents how much the reference element is "stretched" or "compressed" to fit the physical element.

### 3. The Jacobian

In 1D, for a linear element, the Jacobian $J(\xi)$ is constant and represents the ratio of the length of the physical element to the reference element:

$$
J = \frac{dx}{d\xi} = \frac{x_2 - x_1}{2}
$$

This Jacobian $J$ or its absolute value $|J|$ is used to transform the differential $d\xi$ in the reference element to the corresponding differential $dx$ in the physical element.

### 4. Gaussian Quadrature

Gaussian quadrature is used to approximate the integral. For an integral over the interval $[-1, 1]$, Gaussian quadrature approximates the integral as:

$$
\int_{-1}^{1} f(\xi) \, d\xi \approx \sum_{k=1}^{n_{\text{gauss}}} w_k f(\xi_k)
$$

where:
- $\xi_k$ are the Gauss points.
- $w_k$ are the corresponding weights.

For example, in 1D with 2-point Gaussian quadrature:
- The Gauss points are $\xi_1 = -\frac{\sqrt{3}}{3}$ and $\xi_2 = \frac{\sqrt{3}}{3}$.
- The weights are both $w_1 = w_2 = 1$.

### 5. Applying Gaussian Quadrature to the Mass Matrix

Using Gaussian quadrature, the integral for the mass matrix over each element can be approximated as:

$$
\mathbf{M}_{ij} \approx \sum_{k=1}^{n_{\text{gauss}}} w_k N_i(\xi_k) N_j(\xi_k) |J|
$$

Here:
- $N_i(\xi_k)$ and $N_j(\xi_k)$ are the values of the shape functions evaluated at the Gauss points.
- $|J|$ is the absolute value of the Jacobian, accounting for the scaling between the reference and physical element.
- $w_k$ is the weight associated with the Gauss point.

### 6. Differential Volume

The term $w_k \cdot |J|$ is the "differential volume" at the Gauss point, adjusting the integrand to account for the size of the physical element and the numerical integration rule:

- $|J|$ transforms the differential element from the reference space to the physical space.
- $w_k$ ensures that the contribution of each Gauss point is weighted correctly according to the integration scheme.

### 7. Local to Global Matrix Assembly

Once the mass matrix for each element is computed, these local matrices are assembled into the global mass matrix by summing up the contributions from all elements.

### Summary

- **Mass Matrix**: Represents the distribution of mass in the system.
- **Transformation to Reference Element**: Simplifies integration by mapping the physical element to a standard reference element.
- **Jacobian Determinant**: A scaling factor that accounts for the transformation from reference to physical coordinates.
- **Gaussian Quadrature**: A method to numerically approximate integrals using weighted sums at specific points.
- **Differential Volume**: Combines the Gauss weight and the Jacobian determinant to correctly weight each Gauss point's contribution to the integral.
- **Assembly**: The mass matrices from individual elements are combined to form the global mass matrix for the entire domain.

