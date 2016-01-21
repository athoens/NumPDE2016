# Solutions to exercise 3h from sheet 9 in Numerics of PDEs
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


# ---- ENERGY ERROR OF THE SOLVERS FOR THE HOMOGENEOUS NEUMANN PROBLEM -------
# We want to solve the problem
#   -Laplace(u) + u = f    in Omega
#         grad(u).n = 0    on the boundary of Omega
# where Omega = (0,1)².
#
# We also want to compute the energy error given by 
#  err = b(u - usol, u - usol) = l(u) - l(usol)
# where b is the associated bilinear form to the problem.
#       l                    linear                     .

# Definition of model solution and RHS f:
sol = lambda x, y: m.cos(2*m.pi*x)*m.cos(2*m.pi*y);	# model solution
lu = (1 + 8*(m.pi**2))/4;				# value of l(sol)
f = lambda x, y: (1 + 8*(m.pi**2))*sol(x,y);		# RHS

# Order n of numerical integration:
n = 3;

# Mesh spacings: 
Nspacings = 6;			# number of mesh widths to test (in powers of sqrt(2))		 -> CHANGE BACK TO 10!!!!
h0 = 0.3;			# maximum value of mesh width
h = [None]*Nspacings;		# vector for mesh widths (set values)
mmw = np.zeros(Nspacings);	# vector for mesh widths (true values)
discret_err_lin = [None]*Nspacings; 	# vector for discretization errors for linear FEM
discret_err_quad = [None]*Nspacings;	# vector for discretization errors for quadratic FEM

for i in range(0,Nspacings):
  # Maximal mesh width h0 of the grid:
  h[i] = h0*2**(-i/2);
  # Mesh generation:
  p, t, be = mesh.grid_unstr_square(1, h[i],'sheet08_3');
  e = mesh.edgeIndex(p, t);
  N = np.shape(p)[0];
  
  # determine actual max. mesh width
  mmw[i] = mesh.max_mesh_width(p,t);
  
  # LINEAR FINITE ELEMENTS
  # Compute the discretized system:
  A_lin = FEM.stiffness(p, t);		# stiffness-matrix
  M_lin = FEM.mass(p, t);		# mass-matrix
  F_lin = FEM.load(p, t, n, f);		# load-vector

  # Solve the discretized system (A + M)u = F:
  A_lin = A_lin.tocsr(); 		# conversion from lil to csr format
  M_lin = M_lin.tocsr();
  u_lin = spla.spsolve(A_lin + M_lin, F_lin);

  # QUADRATIC FINITE ELEMENTS
  # Compute the discretized system:
  A_quad = FEM.stiffnessP2(p, t, e);	# stiffness-matrix
  M_quad = FEM.massP2(p, t, e);		# mass-matrix
  F_quad = FEM.loadP2(p, t, e, n, f);	# load-vector
  
  # Solve the discretized system (A + M)u = F:
  A_quad = A_quad.tocsr(); 		# conversion from lil to csr format
  M_quad = M_quad.tocsr();
  u = spla.spsolve(A_quad + M_quad, F_quad);
  u_quad = u[0:N];
  
  # Discretization error
  # LINEAR FINITE ELEMENTS
  lun_lin = np.dot(u_lin,F_lin);	# value of l(u_n)
  en_lin = m.sqrt(lu-lun_lin); 		# energy norm of error vector
  discret_err_lin[i] = en_lin;
  # QUADRATIC FINITE ELEMENTS
  lun_quad = np.dot(u[0:N],F_quad[0:N]);		# value of l(u_n)
  en_quad = m.sqrt(abs(lu-lun_quad)); 	# energy norm of error vector  -> HERE SOMETHING GOES TERRIBLY WRONG!!!!
  discret_err_quad[i] = en_quad;

# Plotting the discretization error in dependence of mesh width
fig1 = plt.figure()
plt.plot(mmw, discret_err_lin,':o', label='error $e_n$ for linear FE')
plt.plot(mmw, discret_err_quad,':o', label='error $e_n$ for quadratic FE')
plt.plot(mmw, (discret_err_lin[Nspacings-1]/mmw[Nspacings-1])*mmw,'-',label = '$e_n\propto h$ for linear FE')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('actual maximum mesh width $h$')
plt.ylabel('error  $e_n = \|u_n-u*\|$')
plt.title('energy error depending on mesh width')
plt.legend()
plt.show()
