import numpy as np
from numpy import sqrt
from numpy import cos
from numpy import sin
from numpy import pi
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from time import time

import meshes as msh
import FEM

# parameters
h0 = 0.1
qo = 3
theta = 0.0
T = 1
n = 1000
t = np.linspace(0.0,T,num=n+1)
dt = T/float(n) 
u = lambda x,y: cos(2*pi*x)*cos(2*pi*y)
f = lambda x,y: (8*pi*pi+1)*u(x,y)
ft = 1.0-np.exp(-10*t)

# scaling of time derivative 
dt /= 10

# FE mesh
[coord,trian] = msh.square(1,h0)

# analytical static limit solution
uASL=np.zeros((coord.shape[0]))
for i in range(0,coord.shape[0]):
  uASL[i]=u(coord[i,0],coord[i,1])

# l(u)
lofu = (8*pi*pi+1)/4.0

# FE matrices and load vector
A = FEM.stiffness(coord,trian)
M = FEM.mass(coord,trian)
F = FEM.load(coord,trian,qo,f)

# time-stepping methods
if theta > 0.0:
  S = (M+theta*dt*(A+M)).tocsr()
  
  U = np.zeros((coord.shape[0],n+1))
  t = time()
  for i in range(1,n+1):
    U[:,i] = spla.spsolve(S,dt*(theta*ft[i]+(1.0-theta)*ft[i-1])*F+(M-dt*(1.0-theta)*(A+M))*U[:,i-1])
    #print "t:", t[i], "norm of u:", la.norm(U[:,i])
  print "time of spsolve:  ", time()-t
  if not np.isfinite(U[:,-1]).all():
    print "error of spsolve:  inf"
  else:
    error2 = lofu-np.dot(U[:,-1],F)
    if error2<0.0:
      print "error of spsolve:  nan ( error^2 =", error2, ")"
    else:
      print "error of spsolve: ", np.sqrt(error2)

  V = np.zeros((coord.shape[0],n+1))
  S = S.tocsc()
  t = time()
  LU = spla.splu(S)
  for i in range(1,n+1):
    V[:,i] = LU.solve(dt*(theta*ft[i]+(1.0-theta)*ft[i-1])*F+(M-dt*(1.0-theta)*(A+M))*V[:,i-1])
  print "time of splu:  ", time()-t
  if not np.isfinite(V[:,-1]).all():
    print "error of splu:  inf"
  else:
    error2 = lofu-np.dot(V[:,-1],F)
    if error2<0.0:
      print "error of splu:  nan ( error^2 =", error2, ")"
    else:
      print "error of splu: ", np.sqrt(error2)

  W = np.zeros((coord.shape[0],n+1))
  t = time()
  for i in range(1,n+1):
    W[:,i], info = spla.cg(S,dt*(theta*ft[i]+(1.0-theta)*ft[i-1])*F+(M-dt*(1.0-theta)*(A+M))*W[:,i-1])
    if info:
      print "i =", i, "info =", info
  print "time of cg:  ", time()-t
  if not np.isfinite(W[:,-1]).all():
    print "error of cg:  inf"
  else:
    error2 = lofu-np.dot(W[:,-1],F)
    if error2<0.0:
      print "error of cg:  nan ( error^2 =", error2, ")"
    else:
      print "error of cg: ", np.sqrt(error2)
  
  #FEM.plot(coord,trian,U[:,-1]-uASL)
  #FEM.plot(coord,trian,V[:,-1]-uASL)
  #FEM.plot(coord,trian,W[:,-1]-uASL)

else: 
  t = time()
  M = FEM.mass(coord,trian)
  U = np.zeros((coord.shape[0],n+1))
  for i in range(1,n+1):
    U[:,i], info = spla.cg(M,dt*ft[i-1]*F+(M-dt*(A+M))*U[:,i-1])
    if info:
      print "i =", i, "info =", info
  print "time of cg:  ", time()-t
  if not np.isfinite(U[:,-1]).all():
    print "error of cg:  inf"
  else:
    error2 = lofu-np.dot(U[:,-1],F)
    if error2<0.0:
      print "error of cg:  nan ( error^2 =", error2, ")"
    else:
      print "error of cg: ", np.sqrt(error2)
  
  t = time()
  diagM = FEM.massLumping(coord,trian)
  M = sp.spdiags(diagM, 0, coord.shape[0], coord.shape[0], format="csr")
  V = np.zeros((coord.shape[0],n+1))
  for i in range(1,n+1):
    V[:,i] = (dt*ft[i-1]*F+(M-dt*(A+M))*V[:,i-1])/diagM
  print "time of mass lumping:  ", time()-t
  if not np.isfinite(V[:,-1]).all():
    print "error of mass lumping:  inf"
  else:
    error2 = lofu-np.dot(V[:,-1],F)
    if error2<0.0:
      print "error of mass lumping:  nan ( error^2 =", error2, ")"
    else:
      print "error of mass lumping: ", np.sqrt(error2)
  
  #FEM.plot(coord,trian,U[:,-1]-uASL)
  #FEM.plot(coord,trian,V[:,-1]-uASL)
