import numpy as np
from numpy import sqrt
from numpy import cos
from numpy import sin
from numpy import pi
#import numpy.linalg as la
#import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

import meshes as msh
import FEM


h0 = 0.5
qo = 1

u = lambda x,y: cos(2*pi*x)*cos(2*pi*y)
f = lambda x,y: (8*pi*pi+1)*u(x,y)

lu = (8*pi*pi+1)/4.0

n=10
error = np.zeros((n))
h = np.zeros((n))
for i in range(n):
  h[i]=h0*pow(2,-i/2.0)
  [p,t]=msh.square(1,h[i])
  h[i]=msh.max_mesh_width(p,t)
  A=FEM.stiffness(p,t).tocsr()
  M=FEM.mass(p,t).tocsr()
  F=FEM.load(p,t,qo,f)
  Un=spsolve(A+M,F)
  error[i] = np.sqrt(lu-np.dot(Un,F))
  if i>0:
    print "rate", (np.log(error[i-1])-np.log(error[i]))/(np.log(h[i-1])-np.log(h[i]))

plt.loglog(h,error)
plt.grid(True)
plt.show()
