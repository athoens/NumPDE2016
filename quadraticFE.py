import numpy as np
from numpy import sqrt
from numpy import cos
from numpy import sin
from numpy import pi
import numpy.linalg as la
import scipy.sparse as sparse
import matplotlib.pylab as plt
from scipy.sparse.linalg import spsolve

import meshes as msh
import FEM

qo = 1

sol = lambda x, y: cos(2.0*pi*x)*cos(2.0*pi*y);
lu = (1 + 8*(pi**2))/4;                               # value of l(sol)
f   = lambda x, y: (1.0 + 8.0*(pi**2))*sol(x,y);

Nspacings = 6;                  # number of mesh widths to test (in powers of sqrt(2))           -> CHANGE BACK TO 10!!!!
h0 = 0.3;                       # maximum value of mesh width
h = [None]*Nspacings;           # vector for mesh widths (set values)
mmw = np.zeros(Nspacings);      # vector for mesh widths (true values)
discret_err_lin = [None]*Nspacings;     # vector for discretization errors for linear FEM
discret_err_quad = [None]*Nspacings;    # vector for discretization errors for quadratic FEM

for j in range(0,Nspacings):
  # Maximal mesh width h0 of the grid:
  h[j] = h0*2.0**(-j/2.0);
  # Mesh generation:

  [p,t]=msh.square(q0,h[j])

  # determine actual max. mesh width
  mmw[j] = msh.max_mesh_width(p,t);

  # LINEAR FINITE ELEMENTS
  # Compute the discretized system:
  A_lin = FEM.stiffness(p, t);          # stiffness-matrix
  M_lin = FEM.mass(p, t);               # mass-matrix
  F_lin = FEM.load(p, t, 3, f);         # load-vector

  # Solve the discretized system (A + M)u = F:
  A_lin = A_lin.tocsr();                # conversion from lil to csr format
  M_lin = M_lin.tocsr();
  u_lin = spsolve(A_lin + M_lin, F_lin);

  eIndex = msh.edgeIndex(p,t)
  N = p.shape[0];

  # Computing the discretized system:
  A = FEM.stiffnessP2(p,t,eIndex)
  M = FEM.massP2(p,t,eIndex)
  F = FEM.loadP2(p,t,eIndex,f,3)

  # Solving the discretized system (A + M)u = F:
  A = A.tocsr();
  M = M.tocsr();
  u = spsolve(A + M, F);
  # print np.shape(u), 'N= ', N
  #u = u[0:N];

  # Exact solution of the problem:
  usol = [None]*N;
  for i in range(0, N):
    usol[i] = sol(p[i][0], p[i][1]);

  # LINEAR FINITE ELEMENTS
  lun_lin = np.dot(u_lin,F_lin);        # value of l(u_n)
  en_lin = sqrt(lu-lun_lin);          # energy norm of error vector
  discret_err_lin[j] = en_lin;
  #lun_lin = np.dot(u_lin-usol,F_lin);        # value of l(u_n)
  #en_quad = sqrt(abs(lun_lin));
  #discret_err_lin[j] = en_quad;
  
  # QUADRATIC FINITE ELEMENTS
  lun_quad = np.dot(u[:],F[:]);                # value of l(u_n)
  en_quad = sqrt(abs(lu-lun_quad));   # energy norm of error vector 
  discret_err_quad[j] = en_quad;
  #lun_quad = np.dot(u-usol,F[0:N]);                # value of l(u_n)
  #en_quad = sqrt(abs(lun_quad));
  #discret_err_quad[j] = en_quad;

a = discret_err_quad[3]/mmw[3]**2
b = discret_err_lin[3]/mmw[3]
plt.plot(mmw, discret_err_lin,':*', label='energy error for linear FE')
plt.loglog(mmw, b*mmw[:], label='linear slope')
plt.loglog(mmw, discret_err_quad,':o', label='energy error for quadratic FE')
plt.loglog(mmw, a*mmw[:]**2, label='quadratic sloe')
plt.title('energy error depending on mesh width')
plt.legend()
plt.show()

## Visualization:
#FEM.plot(p, t, u);		# approximated solution
#plt.title('Approximation quadratic');
#FEM.plot(p, t, u-u_lin);                # approximated solution
#plt.title('Approximation linear error');
#FEM.plot(p, t, u-usol);	# approximation error
#plt.title('Approximation Error');
#plt.show()
