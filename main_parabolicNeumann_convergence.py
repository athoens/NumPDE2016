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

# which time discretization to compute
EED = True
EEI = False
EEM = True
CND = True
CNI = False
EID = True
EII = False

# static limit as reference solution
SLARS = False 

# own reference solution for mass lumping
MLRS = True

# FE parameters
h0 = 0.3
qo = 3

# time stepping parameters  
T  = 10
n0 = 16
nSamples = 10
CG_tol = 1e-8

# scaling of time derivative and source term
scale_ft = 1

# source term and static limit solution
u  = lambda x,y: cos(2*pi*x)*cos(2*pi*y)
f  = lambda x,y: (8*pi*pi+1)*u(x,y)
#ft = lambda t: 1.0
ft = lambda t: 1.0-np.exp(-scale_ft*t)

# FE mesh
[coord,trian] = msh.square(1,h0)
print "number of nodes in FE mesh:", coord.shape[0]

# FE matrices and load vector
A = FEM.stiffness(coord,trian)
M = FEM.mass(coord,trian)
F = FEM.load(coord,trian,qo,f)

# mass lumping
diagM = FEM.massLumping(coord,trian)
ML = sp.spdiags(diagM, 0, coord.shape[0], coord.shape[0], format="csr")

# reference solution
uRefStat = spla.spsolve(A+M,F)
if SLARS:
  print "take FE solution of static limit as reference solution"
  uRef = uRefStat
else:
  theta = .5
  n = int(n0*pow(2,nSamples))
  print "compute reference solution with theta =", theta, " and n =", n, "time steps"
  dt = T/float(n)
  t = np.linspace(0.0,T,num=n+1)
  uRef = np.zeros((coord.shape[0],n+1))
  LU = spla.splu((M+theta*dt*(A+M)).tocsc())
  for i in range(1,n+1):
    uRef[:,i] = LU.solve(dt*(theta*ft(t[i])+(1.0-theta)*ft(t[i-1]))*F+(M-dt*(1.0-theta)*(A+M))*uRef[:,i-1])
  uRef = uRef[:,-1]
  lofu = (8*pi*pi+1)/4.0
  if np.isfinite(uRef).all():
    error2 = lofu-np.dot(uRef,F)
    if error2<0.0:
      print "error of reference solution compared to analytical solution of static limit  nan ( error^2 =", error2, ")"
    else:
      print "error of reference solution compared to analytical solution of static limit ", np.sqrt(error2)
    error2 = np.dot((uRefStat-uRef),(A+M)*(uRefStat-uRef))
    print "error of reference solution compared to FE solution of static limit: ", np.sqrt(error2)
  else:
    print "error of reference solution compared to analytical solution of static limit:  inf"
    print "error of reference solution compared to FE solution of static limit:  inf"

if EEM and MLRS and not SLARS:
  print "compute reference solution for mass lumping with n =", n, "time steps"
  uRefML = np.zeros((coord.shape[0],n+1))
  for i in range(1,n+1):
    uRefML[:,i] = (dt*ft(t[i-1])*F+(ML-dt*(A+M))*uRefML[:,i-1])/diagM
  if np.isfinite(uRefML[:,-1]).all():
    uRefML = uRefML[:,-1]
    error2 = lofu-np.dot(uRefML,F)
    if error2<0.0:
      print "error of mass lumping reference solution compared to analytical solution of static limit  nan ( error^2 =", error2, ")"
    else:
      print "error of mass lumping reference solution compared to analytical solution of static limit ", np.sqrt(error2)
    error2 = np.dot((uRefStat-uRefML),(A+M)*(uRefStat-uRefML))
    print "error of mass lumping reference solution compared to FE solution of static limit: ", np.sqrt(error2)
  else:
    print "mass lumping solution with n =", n, "time steps is not finite -> take usual reference solution"
    uRefML = uRef
elif EEM and not MLRS:
  print "take reference solution also for mass lumping"  
  uRefML = uRef
elif EEM:
  print "take FE solution of static limit as reference solution for mass lumping"

# start convergence analysis
if EED or EEI or EEM or CND or CNI or EID or EII:

  n  = np.asarray(n0*pow(2,np.linspace(0,nSamples-1,num=nSamples)),dtype=int)
  dt = np.zeros((nSamples))
  print "number of time steps:", n
  
  EEDC=np.zeros((nSamples))
  EEDE=np.zeros((nSamples))
  EEIC=np.zeros((nSamples))
  EEIE=np.zeros((nSamples))
  EEMC=np.zeros((nSamples))
  EEME=np.zeros((nSamples))

  CNDC=np.zeros((nSamples))
  CNDE=np.zeros((nSamples))
  CNIC=np.zeros((nSamples))
  CNIE=np.zeros((nSamples))

  EIDC=np.zeros((nSamples))
  EIDE=np.zeros((nSamples))
  EIIC=np.zeros((nSamples))
  EIIE=np.zeros((nSamples))

  for j in range(nSamples):
    
    # create equidistant time discretization
    t     = np.linspace(0.0,T,num=n[j]+1)
    dt[j] = T/float(n[j])
    
    print "number of time steps:", n[j]
    
    # Euler explicit (theta=0), LU decomposition
    if EED:
      #tmp = time()
      #M  = FEM.mass(coord,trian)
      U  = np.zeros((coord.shape[0],n[j]+1))
      tmp = time()
      LU = spla.splu(M.tocsc())
      for i in range(1,n[j]+1):
        U[:,i] = LU.solve(dt[j]*ft(t[i-1])*F+(M-dt[j]*(A+M))*U[:,i-1])
      EEDC[j] = time()-tmp
      if np.isfinite(U[:,-1]).all():
        EEDE[j] = np.sqrt(np.dot((U[:,-1]-uRef),(A+M)*(U[:,-1]-uRef)))
      else:
        EEDE[j] = float('nan')
    
    # Euler explicit (theta=0), CG
    if EEI:
      #tmp = time()
      #M = FEM.mass(coord,trian)
      U  = np.zeros((coord.shape[0],n[j]+1))
      tmp = time()
      for i in range(1,n[j]+1):
        #U[:,i], info = spla.cg(M,dt[j]*ft(t[i-1])*F+(M-dt[j]*(A+M))*U[:,i-1])
        U[:,i], info = spla.cg(M,dt[j]*ft(t[i-1])*F+(M-dt[j]*(A+M))*U[:,i-1],tol=CG_tol)
      EEIC[j] = time()-tmp
      if np.isfinite(U[:,-1]).all():
        EEIE[j] = np.sqrt(np.dot((U[:,-1]-uRef),(A+M)*(U[:,-1]-uRef)))
      else:
        EEIE[j] = float('nan')
    
    # Euler explicit (theta=0), mass lumping
    if EEM:
      #tmp = time()
      #diagM = FEM.massLumping(coord,trian)
      #ML = sp.spdiags(diagM, 0, coord.shape[0], coord.shape[0], format="csr")
      U  = np.zeros((coord.shape[0],n[j]+1))
      tmp = time()
      for i in range(1,n[j]+1):
        U[:,i] = (dt[j]*ft(t[i-1])*F+(ML-dt[j]*(A+M))*U[:,i-1])/diagM
      EEMC[j] = time()-tmp
      if np.isfinite(U[:,-1]).all():
        EEME[j] = np.sqrt(np.dot((U[:,-1]-uRefML),(A+M)*(U[:,-1]-uRefML)))
      else:
        EEME[j] = float('nan')
    
    # Crank-Nicolson (theta=1/2), LU decomposition
    if CND:
      theta = 0.5
      S = (M+theta*dt[j]*(A+M)).tocsc()
      U  = np.zeros((coord.shape[0],n[j]+1))
      tmp = time()
      LU = spla.splu(S)
      for i in range(1,n[j]+1):
        U[:,i] = LU.solve(dt[j]*(theta*ft(t[i])+(1.0-theta)*ft(t[i-1]))*F+(M-dt[j]*(1.0-theta)*(A+M))*U[:,i-1])
      CNDC[j] = time()-tmp
      if np.isfinite(U[:,-1]).all():
        CNDE[j] = np.sqrt(np.dot((U[:,-1]-uRef),(A+M)*(U[:,-1]-uRef)))
      else:
        CNDE[j] = float('nan')
    
    # Crank-Nicolson (theta=1/2), CG
    if CNI:
      theta = 0.5
      S = (M+theta*dt[j]*(A+M)).tocsr()
      U  = np.zeros((coord.shape[0],n[j]+1))
      tmp = time()
      for i in range(1,n[j]+1):
        #U[:,i], info = spla.cg(S,dt[j]*(theta*ft(t[i])+(1.0-theta)*ft(t[i-1]))*F+(M-dt[j]*(1.0-theta)*(A+M))*U[:,i-1])
        U[:,i], info = spla.cg(S,dt[j]*(theta*ft(t[i])+(1.0-theta)*ft(t[i-1]))*F+(M-dt[j]*(1.0-theta)*(A+M))*U[:,i-1],tol=CG_tol)
      CNIC[j] = time()-tmp
      if np.isfinite(U[:,-1]).all():
        CNIE[j] = np.sqrt(np.dot((U[:,-1]-uRef),(A+M)*(U[:,-1]-uRef)))
      else:
        CNIE[j] = float('nan')
    
    # Euler implicit (theta=1), LU decomposition
    if EID:
      theta = 1.0
      S = (M+theta*dt[j]*(A+M)).tocsc()
      U  = np.zeros((coord.shape[0],n[j]+1))
      tmp = time()
      LU = spla.splu(S)
      for i in range(1,n[j]+1):
        U[:,i] = LU.solve(dt[j]*(theta*ft(t[i])+(1.0-theta)*ft(t[i-1]))*F+(M-dt[j]*(1.0-theta)*(A+M))*U[:,i-1])
      EIDC[j] = time()-tmp
      if np.isfinite(U[:,-1]).all():
        EIDE[j] = np.sqrt(np.dot((U[:,-1]-uRef),(A+M)*(U[:,-1]-uRef)))
      else:
        EIDE[j] = float('nan')
    
    # Euler implicit (theta=1), CG
    if EII:  
      theta = 1.0
      S = (M+theta*dt[j]*(A+M)).tocsr()
      U  = np.zeros((coord.shape[0],n[j]+1))
      tmp = time()
      for i in range(1,n[j]+1):
        #U[:,i], info = spla.cg(S,dt[j]*(theta*ft(t[i])+(1.0-theta)*ft(t[i-1]))*F+(M-dt[j]*(1.0-theta)*(A+M))*U[:,i-1])
        U[:,i], info = spla.cg(S,dt[j]*(theta*ft(t[i])+(1.0-theta)*ft(t[i-1]))*F+(M-dt[j]*(1.0-theta)*(A+M))*U[:,i-1],tol=CG_tol)
      EIIC[j] = time()-tmp
      if np.isfinite(U[:,-1]).all():
        EIIE[j] = np.sqrt(np.dot((U[:,-1]-uRef),(A+M)*(U[:,-1]-uRef)))
      else:
        CNIE[j] = float('nan')

  import matplotlib.pyplot as plt
  fig1 = plt.figure()
  fig2 = plt.figure()
  fig3 = plt.figure()
  ax1  = fig1.add_subplot(111)
  ax2  = fig2.add_subplot(111)
  ax3  = fig3.add_subplot(111)

  if EED:
    ax1.loglog(dt,EEDE,'r-o',label="EE LU")
  if EEI:
    ax1.loglog(dt,EEIE,'r-+',label="EE CG")
  if EEM:
    ax1.loglog(dt,EEME,'r-x',label="EEM")
  if CND:
    ax1.loglog(dt,CNDE,'g-o',label="CN LU")
  if CNI:
    ax1.loglog(dt,CNIE,'g-+',label="CN CG")
  if EID:
    ax1.loglog(dt,EIDE,'b-o',label="EI LU")
  if EII:
    ax1.loglog(dt,EIIE,'b-+',label="EI CG")
  ax1.set_xlabel("time step size")
  ax1.set_ylabel("error in energy norm")
  ax1.legend()
  ax1.grid(True)

  if EED:
    ax2.loglog(dt,EEDC,'r-o',label="EE LU")
  if EEI:
    ax2.loglog(dt,EEIC,'r-+',label="EE CG")
  if EEM:
    ax2.loglog(dt,EEMC,'r-x',label="EE ML")
  if CND:
    ax2.loglog(dt,CNDC,'g-o',label="CN LU")
  if CNI:
    ax2.loglog(dt,CNIC,'g-+',label="CN CG")
  if EID:
    ax2.loglog(dt,EIDC,'b-o',label="EI LU")
  if EII:
    ax2.loglog(dt,EIIC,'b-+',label="EI CG")
  ax2.set_xlabel("time step size")
  ax2.set_ylabel("computation time")
  ax2.legend()
  ax2.grid(True)

  if EED:
    ax3.loglog(EEDC,EEDE,'r-o',label="EE LU")
  if EEI:
    ax3.loglog(EEIC,EEIE,'r-+',label="EE CG")
  if EEM:
    ax3.loglog(EEMC,EEME,'r-x',label="EE ML")
  if CND:
    ax3.loglog(CNDC,CNDE,'g-o',label="CN LU")
  if CNI:
    ax3.loglog(CNIC,CNIE,'g-+',label="CN CG")
  if EID:
    ax3.loglog(EIDC,EIDE,'b-o',label="EI LU")
  if EII:
    ax3.loglog(EIIC,EIIE,'b-+',label="EI CG")
  ax3.set_xlabel("computation time")
  ax3.set_ylabel("error in energy norm")
  ax3.legend()
  ax3.grid(True)

  plt.show()
