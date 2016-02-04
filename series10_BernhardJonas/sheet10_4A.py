# Solutions to exercise 4 from sheet 10 in Numerics of PDEs, using method A to generate a
# graded mesh: interpolation between mesh widths in gmsh
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

# --------- MIXED NEUMANN-DIRICHLET PROBLEM ---------------------------
# We want to solve the problem
#       -Laplace(u) = 1    in Omega = (-1,1)²
#         grad(u).n = 0    on part Gamma_N of the boundary of Omega (y>0)
#                 u = 0    on part Gamma_D of the boundary of Omega (y<0)
# using graded meshes

# --------------------- Settings ---------------------------------------
h_0 = 0.5; # mesh widths at the corners and the origin (singular points are equipped with 1/2*h_0^2)
energynorm_ref = 1.548888; # reference value for the energy norm of the exact solution (used to check convergence)
J = 5;	# number of meshes to test (k=0..J-1, where h_k = h_0*2^(-k2))

# ---------------------- Preparation -----------------------------------
# arrays for mesh properties
meshwidths = [None]*J;  # will be filled with the mesh widths h_k
nodes = [None]*J;	# ... numbers of nodes of mesh M_k
elements = [None]*J;	# ... numbers of elements of mesh M_k
ndof = [None]*J;	# ... numbers of global degrees of freedom

# arrays for solution properties
errors = [None]*J;	# will be filled with estimated discretization errors at each mesh size
energynorms = [None]*J;	# ... energy norms of the discrete solutions

# PDE and solver settings
f = lambda x, y: 1.;	# right hand side of equation is constant
n = 3; 	# Order n of numerical integration


# --------------------- Loop over mesh widths --------------------------
for k in range(0, J):
  
  # Mesh generation:
  h = h_0*2**(-k/2);
  meshwidths[k] = h;
  print('using mesh width h = '+str(meshwidths[k])+':');
  # define the mesh using gmsh (corners and singular points are explicitly included as nodes)
  p,t,be = mesh.grid_unstr_sheet10(1., 0.5*h**2, h, 'meshes/10_4A_'+str(k));	# generate graded mesh
  
  # node and element numbers 
  N = np.shape(p)[0];
  M = np.shape(t)[0];
  BE = np.shape(be)[0];
  
  nodes[k] = N;
  elements[k] = M;
  print('  ',nodes[k],'nodes,',elements[k],'elements')


  # -------- Identify nodes on the Dirichlet boundary ------------
  insideflags = np.ones((N, 1), dtype=bool);	# will be true for all "Dirichlet-inner" points,
						# i.e., points not contained in the Dirichlet boundary

  count = 0;	# counter for nodes on Dirichlet boudary
  for i in range(0, BE):	# labelling indices of "boundary nodes" (those contained in the lines fed into gmsh; this includes the line from (-1,0) to (1,0))
    n0 = be[i][0];	# starting point
    n1 = be[i][1];	# end point, we do not use this point, since all nodes occur at least twice -- as start and end point
    
    # check, whether this point has occured before
    if insideflags[n0]:
      # get node's coordinates
      x = p[n0][0];
      y = p[n0][1];
      # check, whether these coordinates are contained in the Dirichlet boundary Gamma_D
      isdirichlet = False;
      if (x==-1. or x==+1.) and y<=0.:
        isdirichlet = True;
      elif y==-1.:
        isdirichlet = True;
      if isdirichlet:
        insideflags[n0] = False;
        count +=1;
  
  # Now, label the Dirichlet-inner nodes properly
  num_inner_points = N-count;
  IN = [None]*(num_inner_points);	# list of interior indices
  count = 0;
  for i in range(0,N):		# labelling indices of interior nodes
    if (insideflags[i][0]):
      IN[count] = i;
      count += 1;
  if count!=num_inner_points: error('Number of Dirichlet-inner nodes missmatch');
  ndof[k] = num_inner_points;	# there are as many degrees of freedom as inner nodes

  # ---------------- assemble algebraic problem ------------------------------
  # Computing the discretized system:
  A = FEM.stiffness(p, t);	# stiffness-matrix
  #M = FEM.mass(p, t);		# mass-matrix
  F = FEM.load(p, t, n, f);	# load-vector

  # Assembling the reduced system:
  Ared = sp.lil_matrix((num_inner_points, num_inner_points));
  #Mred = sp.lil_matrix((num_inner_points, num_inner_points));
  Fred = np.zeros((num_inner_points, 1));
  for i in range(0, num_inner_points):
    for j in range(0, num_inner_points):
      Ared[i,j] = A[IN[i], IN[j]];	# reduced stiffness matrix
      #Mred[i,j] = M[IN[i], IN[j]];	# reduced mass matrix
    Fred[i] = F[IN[i]];			# reduced RHS

  # --------------------- solve algebraic problem -----------------------------
  # Solving the reduced system (Ared + Mred)ured = Fred:
  Ared = Ared.tocsr();
  #Mred = Mred.tocsr();
  ured = spla.spsolve(Ared, Fred);
  
  ufull = np.zeros(N);
  for i in range(0,num_inner_points):
    ufull[IN[i]] = ured[i];
  
  # --------------- estimate discretization error -----------------------------
  # the error is computed via the energy norm of u-u_n, where u is the exact solution and u_n is the 
  # discrete solution. Using the Galerkin orthogonality and the LVP, this can be written as
  # (e_n)^2 = <u,f> - ufull.F
  energynorm_u = m.sqrt(np.dot(ufull,F)); #sum(ufull[:]*F[:])[0]
  print('   energy norm of solution = '+str(energynorm_u)+'\n');
  energynorms[k] = energynorm_u;
  errors[k] = m.sqrt(energynorm_ref**2 - energynorm_u**2 );
  
  # ---------- Plot the solution for the last (and finest) mesh ----------------
  if k==J-1:
    fig1 = plt.figure(1);
    ax1 = FEM.plot(p, t, ufull, fig1, 121); # show approximated solution
    ax1.set_title('Approximation');
    ax1.axis('equal');
    ax2 = fig1.add_subplot(122);
    mesh.show_mkbnd(p, t, be, ax2);	# show mesh
    ax2.set_title('Mesh');  
    #plt.show();

# --------------------------- End loop over meshes -----------------------------


# ------------------- Plot the results -----------------------------------------
# Plot the behavior of mesh sizes and of discretization error in the energy norm:
fig2 = plt.figure(2);

# mesh size:
ax1 = fig2.add_subplot(121);
ax1.plot(meshwidths, nodes,'b o', label='nodes')
ax1.plot(meshwidths, elements, 'r x', label = 'elements')
ax1.plot(meshwidths, nodes[0]*(np.array(meshwidths)/meshwidths[0])**(-2),'g-', label = '$N\propto h^{-2}$')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('mesh width $h$')
ax1.set_ylabel('number of nodes/elements')
ax1.set_title('scaling of mesh size')
ax1.legend(loc='lower left')

# convergence in energy norm:
ax2 = fig2.add_subplot(122);
ax2.plot(ndof, errors, 'b o', label='data');
ax2.plot(ndof, errors[0]*(np.array(ndof)/ndof[0])**(-.5), 'g-', label = '$e_n\propto N^{-1/2}$')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('number of d.o.f (inner nodes) $N$')
ax2.set_ylabel('error  $e_n = \|u_n-u*\|$')
ax2.set_title('energy error depending on d.o.f')
ax2.legend(loc='upper right')
plt.show();