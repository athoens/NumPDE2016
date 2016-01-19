import numpy as np
from numpy import sqrt
from numpy import cos
from numpy import sin
from numpy import pi
import numpy.linalg as la
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

import meshes as msh
import FEM

# If theta<1 the number of iterations has to be as large as n=5000 for final 
# time T=1!
kappa=1
h0 = 0.15
qo = 3
theta = 0.0
T = 1
n = 1000
time = np.linspace(0.0,T,num=n+1)
dt = T/float(n) 
u = lambda x,y: kappa*sin(pi*x)*sin(pi*y)
f = lambda x,y: (2*pi*pi)*u(x,y)
ft = 1.0-np.exp(-10*time)

[p,t]=msh.square(1,h0)

A=kappa*FEM.stiffness(p,t).tocsr()
M=FEM.mass(p,t).tocsr()
F=FEM.load(p,t,qo,f)

IN = FEM.interiorNodes(p,t)
T0 = sparse.lil_matrix((len(p),len(IN)))
for j in range(len(IN)):
  T0[IN[j],j] = 1
T0t = T0.transpose()
T0  = T0.tocsr()
T0t = T0t.tocsr()
A = T0t.dot(A.dot(T0))
M = T0t.dot(M.dot(T0))
F = T0t.dot(F)

S = (M+theta*dt*A)

U=np.zeros((len(IN),n+1))
for i in range(1,n+1):
  U[:,i]=spsolve(S,dt*(theta*ft[i]+(1.0-theta)*ft[i-1])*F+(M-dt*(1.0-theta)*A)*U[:,i-1])
  #print "time:", time[i], "norm of u:", la.norm(U[:,i])
U=T0.dot(U)
  
FEM.plot(p,t,U[:,-1])
uStat=np.zeros((p.shape[0]))
for i in range(0,p.shape[0]):
  uStat[i]=u(p[i,0],p[i,1])
FEM.plot(p,t,uStat-U[:,-1])
