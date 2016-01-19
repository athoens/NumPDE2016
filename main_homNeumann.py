import numpy as np
from numpy import sqrt
from numpy import cos
from numpy import sin
from numpy import pi
#import numpy.linalg as la
#import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

import meshes as msh
import FEM


h0 = 0.1
qo = 1

u = lambda x,y: cos(2*pi*x)*cos(2*pi*y)
f = lambda x,y: (8*pi*pi+1)*u(x,y)
[p,t]=msh.square(1,h0)

#r = lambda x,y: sqrt(x*x+y*y)
#u = lambda x,y: cos(2*pi*r(x,y))
#f = lambda x,y: (4*pi*pi+1)*u(x,y)+2*pi/r(x,y)*sin(2*pi*r(x,y))
#def r(x,y):
  #return sqrt(x*x+y*y)
#def u(x,y):
  #return cos(2*pi*r(x,y))
#def f(x,y): 
  #return (4*pi*pi+1)*u(x,y)+2*pi/r(x,y)*sin(2*pi*r(x,y))
#[p,t]=msh.circle(1,h0,1)

A=FEM.stiffness(p,t).tocsr()
M=FEM.mass(p,t).tocsr()
F=FEM.load(p,t,qo,f)

Un=spsolve(A+M,F)

FEM.plot(p,t,Un)

U=np.zeros((p.shape[0]))
for i in range(0,p.shape[0]):
  U[i]=u(p[i,0],p[i,1])
FEM.plot(p,t,U-Un)
