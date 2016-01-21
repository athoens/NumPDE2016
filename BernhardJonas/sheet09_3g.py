# Solutions to exercise 3g from sheet 9 in Numerics of PDEs
# authors: Bernhard Aigner (359706)
#          Jonas Gienger   (370058)


# Import the following modules:
import meshes as mesh
import FEM
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import math as m
import matplotlib.pyplot as plt


# ------- SOLUTION OF THE HOMOGENEOUS NEUMANN PROBLEM ------------
# We want to solve the problem
#   -Laplace(u) + u = f    in Omega
#         grad(u).n = 0    on the boundary of Omega
# where Omega = (0,1)².

# Definition of RHS f:
#f = lambda x, y: 1.0;
# special choices:
sol = lambda x, y: m.cos(2*m.pi*x)*m.cos(2*m.pi*y);
f = lambda x, y: (1 + 8*(m.pi**2))*sol(x,y);

# Order n of numerical integration:
n = 3;

# Maximal mesh width h0 of the grid:
h0 = 0.1;

# Mesh generation:
p, t, be = mesh.grid_square(1, h0);
e = mesh.edgeIndex(p,t);
N = np.shape(p)[0];

# Computing the discretized system:
A = FEM.stiffnessP2(p, t, e);	# stiffness-matrix
M = FEM.massP2(p, t, e);	# mass-matrix
F = FEM.loadP2(p, t, e, n, f);	# load-vector

# Solving the discretized system (A + M)u = F:
A = A.tocsr();
M = M.tocsr();
u = spla.spsolve(A + M, F);
u = u[0:N];

# Exact solution of the problem:
usol = [None]*N;
for i in range(0, N):
   usol[i] = sol(p[i][0], p[i][1]);

# Visualization:
fig1 = plt.figure();
ax1 = FEM.plot(p, t, u, fig1, 121);		# approximated solution
ax1.set_title('Approximation');
ax2 = FEM.plot(p, t, u - usol, fig1, 122);	# approximation error
ax2.set_title('Approximation Error');
plt.show()
