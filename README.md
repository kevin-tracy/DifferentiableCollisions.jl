<p align="center">
  <img width="320" src="https://github.com/kevin-tracy/DCOL.jl/blob/master/extras/images/DCOL_logo.png">
</p>

A library for differential collision detection between the following convex primitives:
- polytopes 
- capsules
- cylinders
- cones
- spheres
- padded polygons 

For more details, see our paper on [arXiv](https://arxiv.org/abs/2207.00669). 

## Interface
DCOL works by creating a struct for each shape, and calling a function to query a proximity value between them. 
#### Primitives 
Each primitive is implemented as a struct in DCOL. The defining dimensions for each primitive is described in the [paper](https://arxiv.org/abs/2207.00669), and the primitives can be constructed as the following:
```julia
import DCOL as dc

polytype = dc.Polytope(A, b)   # polytope is described by Ax <= b
capsule  = dc.Capsule(R, L)    # radius R, length L 
cylinder = dc.Cylinder(R, L)   # radius R, length L 
cone     = dc.Cone(H, β)       # height H, half angle β
sphere   = dc.Sphere(R)        # radius R
polygon  = dc.Polygon(A, b, R) # polygon is described by Ay <= b, cushion radius R
```
where all of these structs are ```::AbstractPrimitive```, and use a quaternion for attitude. The position and attitude of a primitive `P1::AbstractPrimitive` are updated in the following way:
```julia
using StaticArrays
P1 =  dc.Polytope(A,  b)::AbstractPrimitive
P1.r = SA[1, 2, 3.0]     # position in world frame W
P1.q = SA[1.0, 0, 0, 0]  # quaternion ᵂqᴮ
```
#### MRP Support
In cases where a three-parameter attitude parameterization is more convenient, a Modified Rodrigues Parameter (MRP) can be used in the following way:
```julia
P1 = dc.PolytopeMRP(A, b)::AbstractPrimitiveMRP
P1.r = SA[1,2,3.0]    # position in world frame W
P1.p = SA[0.0,0,0]    # MRP ᵂpᴮ
```
#### Proximity Functions
DCOL exposes a function `proximity` for collision detection, as well as `proximity_jacobian` for collision detection and derivatives. Two optional arguments are included that pertain to the optimization solver under the hood,  `verbose` turns on logging for this solver, and `pdip_tol` is the termination criteria. 
```julia
α,x = dc.proximity(P1, P2; verbose = false, pdip_tol = 1e-6)
α,x,J = dc.proximity_jacobian(P1, P2; verbose = false, pdip_tol = 1e-6)
```
These functions output $\alpha$ as the proximity value, with the following significance:
- $\alpha \leq 1$ means there **is** a collision between the two primitives 
- $\alpha >1$ means there **is not** a collision between the two primitives 

Also, returned is $x$ which is the intersection point between the scaled shapes (see algorithm for significance), and a Jacobian $J$ which is the following:

$$
\begin{align*}
J &= \frac{\partial (x,\alpha) }{\partial (r_1,q_1,r_2,q_2)}
\end{align*}
$$

In the case where `AbstractPrimitiveMRP`'s are used, `proximity_jacobian` will automatically return the following Jacobian:

$$
\begin{align*}
J &= \frac{\partial (x,\alpha) }{\partial (r_1,p_1,r_2,p_2)}
\end{align*}
$$

## Algorithm
DCOL calculates the collision information between two primitives by solving for the minimum scaling applied to both primitives that result in an intersection. This is done by forming an optimization problem over a scalar $\alpha \in \mathbb{R}$, and a point $x \in \mathbb{R}^3$. The following optimization problem solves for the minimum scaling $\alpha$ such that a point $x$ exists in the scaled versions of two primitives $P_1$ and $P_2$. 

$$
\begin{align*}
\underset{x,\alpha}{\text{minimize}} & \quad \alpha \\
\text{subject to} & \quad  α \geq 0, \\
                  & \quad  x \in P_1(\alpha),\\
                  & \quad  x \in P_2(\alpha)
\end{align*}
$$

This problem is a convex optimization problem with conic constraints, and is solved with a custom primal-dual interior-point method inspired by [cvxopt](http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf). If the minimum scaling $\alpha^* > 1$, then there is no collision because each primitive had to be scaled up in order to find an intersection. Alternatively, this means that if $\alpha^* \leq 1$, the two primitives are in contact. The solution to this optimization problem can be differentiated with respect to the position and orientation of each primitive using the implicit function theorem. By using a primal-dual interior-point method and returning a solution at a `pdip_tol` of $[10^{-4},10^{-8}]$, the log barrier will effectively smooth out the corners of the primitives to return useful and smooth derivatives.  


