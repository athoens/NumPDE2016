import numpy as np
from numpy import sqrt
from numpy import cos
from numpy import sin
from numpy import pi
#import numpy.linalg as la
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

import meshes as msh
import FEM


h0 = 0.1
qo = 3

r = lambda x,y: sqrt(x*x+y*y)
u = lambda x,y: x*sin(pi*r(x,y))
f = lambda x,y: pi*pi*u(x,y)-3.0*pi*(x/r(x,y))*cos(pi*r(x,y))
g = lambda x,y: -pi*x
m = lambda x,y: 1.0

[p,t]=msh.circle(1,h0,1)
be=FEM.boundaryEdges(p,t)

A=FEM.stiffness(p,t).tocsr()
F=FEM.load(p,t,qo,f)
G=FEM.loadNeumann(p,be,qo,g)
M=sparse.lil_matrix(FEM.load(p,t,1,m))
S=sparse.bmat([[A, M.transpose()], [M, None]]).tocsr()
F=np.hstack([F+G,0.0])
U_n=spsolve(S,F)
lambda_n=U_n[-1]
print "lambda", lambda_n
FEM.plot(p,t,U_n[0:-1])
U=np.zeros((p.shape[0]))
for i in range(0,p.shape[0]):
  U[i]=u(p[i,0],p[i,1])
FEM.plot(p,t,U-U_n[0:-1])
