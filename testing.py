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


h0 = 0.1
qo = 1


sol = lambda x, y: cos(2.0*pi*x)*cos(2.0*pi*y);
f   = lambda x, y: (1.0 + 8.0*(pi**2))*sol(x,y);

[p,t]=msh.square(1,h0)


[e, eIndex, boundaryNodes, boundaryEdges] = msh.edgeIndex(p,t)
N = p.shape[0];

# Computing the discretized system:
A = FEM.stiffnessP2(p,t,eIndex)
M = FEM.massP2(p,t,eIndex)
F = FEM.loadP2my(p,t,eIndex,f,3)

# Solving the discretized system (A + M)u = F:
A = A.tocsr();
M = M.tocsr();
u = spsolve(A + M, F);
# print np.shape(u), 'N= ', N
u = u[0:N];

# Exact solution of the problem:
usol = [None]*N;
for i in range(0, N):
   usol[i] = sol(p[i][0], p[i][1]);

# Visualization:
#fig1 = plt.figure();
ax1 = FEM.plot(p, t, u);		# approximated solution
plt.title('Approximation');
ax2 = FEM.plot(p, t, u-usol);	# approximation error
plt.title('Approximation Error');
plt.show()
