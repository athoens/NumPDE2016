#  
# gaussTriangle(n)
# 
# returns abscissas and weights for "Gauss integration" in the triangle with 
# vertices (-1,-1), (1,-1), (-1,1)
#
# input:
# n - order of the numerical integration (1 <= n <= 5)
#
# output:
# x - 1xp-array of abscissas, that are 1x2-arrays (p denotes the number of 
#     abscissas/weights)
# w - 1xp-array of weights (p denotes the number of abscissas/weights)
#
def gaussTriangle(n):

  if n == 1:
      x = [[-1/3., -1/3.]];
      w = [2.];
  elif n == 2:
      x = [[-2/3., -2/3.],
           [-2/3.,  1/3.],
           [ 1/3., -2/3.]];
      w = [2/3.,
           2/3.,
           2/3.];
  elif n == 3:
      x = [[-1/3., -1/3.],
           [-0.6, -0.6],
           [-0.6,  0.2],
           [ 0.2, -0.6]];
      w = [-1.125,
            1.041666666666667,
            1.041666666666667,
            1.041666666666667];
  elif n == 4:
      x = [[-0.108103018168070, -0.108103018168070],
           [-0.108103018168070, -0.783793963663860],
           [-0.783793963663860, -0.108103018168070],
           [-0.816847572980458, -0.816847572980458],
           [-0.816847572980458,  0.633695145960918],
           [ 0.633695145960918, -0.816847572980458]];
      w = [0.446763179356022,
           0.446763179356022,
           0.446763179356022,
           0.219903487310644,
           0.219903487310644,
           0.219903487310644];
  elif n == 5:
      x = [[-0.333333333333333, -0.333333333333333],
           [-0.059715871789770, -0.059715871789770],
           [-0.059715871789770, -0.880568256420460],
           [-0.880568256420460, -0.059715871789770],
           [-0.797426985353088, -0.797426985353088],
           [-0.797426985353088,  0.594853970706174],
           [ 0.594853970706174, -0.797426985353088]];
      w = [0.450000000000000,
           0.264788305577012,
           0.264788305577012,
           0.264788305577012,
           0.251878361089654,
           0.251878361089654,
           0.251878361089654];
  else:
      print 'numerical integration of order ' + str(n) + 'not available';
      
  return x, w


#
# plot(p,t,u)
#
# plots the linear FE function u on the triangulation t with nodes p.
#
# input:
# p  - Nx2 matrix with coordinates of the nodes
# t  - Mx3 matrix with indices of nodes of the triangles
# u  - Nx1 coefficient vector
#
def plot(p,t,u):
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_trisurf(p[:, 0], p[:, 1], t, u, cmap=plt.cm.jet)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('u')
  plt.show()


# 
# import other modules
# 
import numpy as np
import numpy.linalg as la
from scipy.sparse import lil_matrix


# 
# elemStiffness
# 
# computes the element stiffness matrix
# 
# input:
# p - 3x2-matrix of the coordinates of the triangle nodes
# 
# output:
# Ak - element stiffness matrix
#
def elemStiffness(p):
  # vertices of the triangle
  P0 = p[0,:];
  P1 = p[1,:];
  P2 = p[2,:];

  # Jacobian of the element map
  Fk = np.c_[P1-P0,P2-P0];
  
  # coordinate difference matrix
  Dk = np.array([[ P1[1]-P2[1], P2[1]-P0[1], P0[1]-P1[1] ],
                 [ P2[0]-P1[0], P0[0]-P2[0], P1[0]-P0[0] ]]);

  # element stiffness matrix
  Ak = (0.5/np.linalg.det(Fk)) * np.dot(Dk.T,Dk);
  
  return Ak;


# 
# elemMass
# 
# computes the element mass matrix
# 
# input:
# p - 3x2-matrix of the coordinates of the triangle nodes
# 
# output:
# Mk - element mass matrix
#
def elemMass(p):
  # vertices of the triangle
  P0 = p[0,:];
  P1 = p[1,:];
  P2 = p[2,:];

  # Jacobian of the element map
  Fk = np.c_[P1-P0,P2-P0];
  
  # element mass matrix
  Mk = (0.5*np.linalg.det(Fk)) * (1.0/(12.0*np.ones((3,3))-6.0*np.eye(3)));
  
  return Mk;


# 
# elemLoad
#
# computes the element load vector
#
# input:
# p - 3x2-matrix of the coordinates of the triangle nodes
# n - order of the numerical integration rule (1 <= n <= 5)
# f - source term function
#
# output:
# phi - element load vector (3x1-matrix)
#
def elemLoad(p,n,f):
  
  # read quadrature points
  x, w = gaussTriangle(n);
  x = (np.asarray(x) + 1.0)/2.0;
  w = np.asarray(w)/4.0;

  # number of quadrature points
  k = np.size(w);

  # vertices of the triangle
  P0 = p[0,:];
  P1 = p[1,:];
  P2 = p[2,:];

  # Jacobian of the element map and its determinant
  Fk = np.c_[P1-P0,P2-P0];
  detFk = np.linalg.det(Fk);

  # numerical integration
  #if k == 1:
    #phi = detFk*w*f(P0 + np.dot(Fk,x))*np.array(([1-x[0]-x[1],x[0],x[1]]));
  #else:
  phi = np.zeros((3));
  for i in range(0,k):
      y = P0 + np.dot(Fk,x[i,:]);
      phi[0] = phi[0] + detFk * w[i] * f(y[0],y[1]) * (1-x[i,0]-x[i,1]);
      phi[1] = phi[1] + detFk * w[i] * f(y[0],y[1]) * x[i,0];
      phi[2] = phi[2] + detFk * w[i] * f(y[0],y[1]) * x[i,1];

  return phi;


# 
# assembleMatrix
#
# Assembles the matrix related to the element matrix bf
#
# input:
# p  - Nx2 matrix with coordinates of the nodes
# t  - Mx3 matrix with indices of nodes of the triangles
# bf - Name of callable function for the computation of the element matrix
#
# output:
# A  - NxN matrix in scipy's sparse lil format
#
def assembleMatrix(p, t, bf):
  A = lil_matrix((p.shape[0],p.shape[0]));
  for m in range(0,t.shape[0]):
    AK = bf(p[t[m,:],:]);
    for i in range(0,3):
      for j in range(0,3):
        A[t[m,i],t[m,j]] += AK[i,j];
  return A;
  

# 
# stiffness
#
# Returns the stiffness matrix A for 
#   int_Omega grad u . grad v dx
# for linear FEM on triangles
#
# input:
# p - Nx2 matrix with coordinates of the nodes
# t - Mx3 matrix with indices of nodes of the triangles
#
# output:
# A - NxN stiffness matrix in scipy's sparse lil format
#   
def stiffness(p,t):
  def bf(x):
    return elemStiffness(x);
  return assembleMatrix(p,t,bf);
  

# 
# mass
#
# Returns the mass matrix M for 
#   int_Omega u v dx
# for linear FEM on triangles
#
# input:
# p - Nx2 matrix with coordinates of the nodes
# t - Mx3 matrix with indices of nodes of the triangles
#
# output:
# M - NxN mass matrix in scipy's sparse lil format
#
def mass(p,t):
  def bf(x):
    return elemMass(x);
  return assembleMatrix(p,t,bf);
  

# 
# assembleVector
#
# Assembles the vector related to the element vector lf
#
# input:
# p  - Nx2 matrix with coordinates of the nodes
# t  - Mx3 matrix with indices of nodes of the triangles
# lf - Name of callable function for the computation of the element vector
#
# output:
# A  - Nx1 numpy-array
#
def assembleVector(p, t, lf):
  b = np.zeros((p.shape[0]));
  for m in range(0,t.shape[0]):
    bK = lf(p[t[m,:],:]);
    for i in range(0,3):
      b[t[m,i]] += bK[i];
  return b;

  
# 
# load
#
# Returns the load vector b for 
#   int_Omega f v dx
# for linear FEM on triangles
#
# input:
# p - Nx2 matrix with coordinates of the nodes
# t - Mx3 matrix with indices of nodes of the triangles
# n - order of the numerical quadrature
# f - source term function
#
# output:
# b - Nx1 load vector as numpy-array
#
def load(p,t,qo,f):
  def lf(x):
    return elemLoad(x,qo,f);
  return assembleVector(p,t,lf);


# 
# boundaryEdges(p, t)
# 
# returns the endpoints of boundary edges as indices into p.
#
# input:
# p - Nx2 array with coordinates of the nodes
# t - Mx3 array with indices of nodes of the triangles
#
# output:
# be - Bx2 array of nodes as indices into p that are endpoints of boundary 
#      edges
#
def boundaryEdges(p,t):
  edgematrix = np.zeros((3*t.shape[0],3), dtype=np.int)
  index = 0
  for i in range(t.shape[0]):
    for j in range(3):
      k = (j+1)-np.floor((j+1)/3)*3
      p0 = min([t[i,j],t[i,k]])
      p1 = max([t[i,j],t[i,k]])
      a = edgematrix[:,0]==p0
      b = edgematrix[:,1]==p1
      if (np.dot(a*b,np.ones(a.shape))):
        edgematrix[a*b,2] += 1
      else:
        edgematrix[index,0] = p0
        edgematrix[index,1] = p1
        edgematrix[index,2] = 1
        index += 1
  be=edgematrix[edgematrix[:,2]==1,0:2]
  return be


# 
# interiorNodes(p, t)
# 
# returns the interior nodes as indices into p.
#
# input:
# p - Nx2 array with coordinates of the nodes
# t - Mx3 array with indices of nodes of the triangles
#
# output:
# in - Ix1 array of nodes as indices into p that do not lie on the 
#      boundary
#
def interiorNodes(p,t):
  bn = np.unique(boundaryEdges(p,t))
  nodes = range(0,len(p))
  return np.delete(nodes,np.unique(boundaryEdges(p,t)))


# 
# elemLoadNeumann(p, n, g)
# 
# returns the element vector related to the Neumann boundary data
#   int_I g v ds
# for linear FEM on the straight boundary edge I.
#
# input:
# p - 2x2 matrix of the coordinates of the nodes on the boundary edge
# n - order of the numerical quadrature
# g - Neumann data as standard Python function or Python's lambda 
#     function
#
# output:
# gK - element vector (2x1 array)
#
def elemLoadNeumann(p, n, g):
  
  # vertices of the interval
  P0 = p[0,:]
  P1 = p[1,:]

  # length of interval
  L = la.norm(P0-P1)

  # read quadrature points and weights
  from numpy.polynomial.legendre import leggauss
  x, w = leggauss(n)
  x    = (x + 1.0)/2.0  # transform quadrature points to interval [0,1]
  w    = w/2.0          # scale weights according to transformation

  # numerical integration
  gK = np.zeros((2))
  for i in range(n):
    # transform quadrature points to interval [P0,P1]
    y = P0 + x[i]*(P1-P0)
    # add weight w(i) multiplied with function g at y multiplied with
    # element shape function multiplied with length L of interval
    gK[0] += L * w[i] * g(y[0],y[1]) * (1-x[i])
    gK[1] += L * w[i] * g(y[0],y[1]) * x[i]
  
  # return gK
  return gK


# 
# loadNeumann(p, be, n, g)
# 
# returns the vector related to the Neumann boundary data
#   int_dOmega g v ds
# for linear FEM on straight boundary edges.
#
# input:
# p  - Nx2 matrix with coordinates of the nodes
# be - Bx2 matrix with the indices of the nodes of boundary edges
# n  - order of the numerical quadrature
# g  - Neumann data as standard Python function or Python's lambda 
#      function
#
# output:
# LoadNeumann - Nx1 vector as numpy-array
#
def loadNeumann(p, be, n, g):
  
  # number of nodes
  N = p.shape[0]

  # number of boundaryEdges
  B = be.shape[0]

  # assemble vector
  LoadNeumann = np.zeros((N))
  for i in range(B):
    LoadNeumannK=elemLoadNeumann(p[be[i,:],:],n,g)
    for j in range(2):
      LoadNeumann[be[i,j]] += LoadNeumannK[j]
  
  # return LoadNeumann
  return LoadNeumann

#
# solveD0(p,t,matrix,vector)
#
# returns the solution of a BVP with homogeneous Dirichlet boundary conditions
#
def solveD0(p,t,matrix,vector):
  from scipy.sparse.linalg import spsolve
  IN = interiorNodes(p,t)
  T0 = sparse.lil_matrix((len(p),len(IN)))
  for j in range(len(IN)):
    T0[IN[j],j] = 1
  T0t = T0.transpose()
  T0  = T0.tocsr()
  T0t = T0t.tocsr()
  matrix0 = T0t.dot(matrix.dot(T0))
  vector0 = T0t.dot(vector)
  solution0 = spsolve(matrix0,vector0)
  return T0.dot(solution0)


# 
# elemMassLumping
# 
# computes the mass lumping approximation to the element mass matrix
# 
# input:
# p - 3x2-matrix of the coordinates of the triangle nodes
# 
# output:
# Mk - element mass matrix
#
def elemMassLumping(p):
  # vertices of the triangle
  P0 = p[0,:];
  P1 = p[1,:];
  P2 = p[2,:];

  # Jacobian of the element map
  Fk = np.c_[P1-P0,P2-P0];
  
  # approximated element mass matrix
  Mk = np.linalg.det(Fk)/3.0
  
  return Mk;


# 
# massLumping
#
# Returns the mass lumping approximation to the mass matrix M for 
#   int_Omega u v dx
# for linear FEM on triangles
#
# input:
# p - Nx2 matrix with coordinates of the nodes
# t - Mx3 matrix with indices of nodes of the triangles
#
# output:
# M - Nx1 vector of the diagonal entries of the approximated mass matrix
#
def massLumping(p,t):
  M = np.zeros((p.shape[0]))
  for m in range(0,t.shape[0]):
    MK = elemMassLumping(p[t[m,:],:]);
    for i in range(0,3):
      M[t[m,i]] += MK;
  return M;


# 
# assembleMatrixParallel
#
# Assembles the matrix related to the element matrix bf in parallel
#
# input:
# p  - Nx2 matrix with coordinates of the nodes
# t  - Mx3 matrix with indices of nodes of the triangles
# bf - Name of callable function for the computation of the element matrix
#
# output:
# A  - NxN matrix in scipy's sparse lil format
#
#def assembleMatrixParallel(p, t, bf):
  #import multiprocessing as mp
  #processes = 2#mp.cpu_count()
  #tindex=np.zeros((processes+1),dtype=int)
  #m=int(t.shape[1]/processes)
  #for i in range(2,processes):
    #tindex[i]=i*m
  #tindex[-1]=t.shape[1]
  
  #def assembleMatrixParallelHelper(pid):
    #print "assembling of process", pid
    #return assembleMatrix(p, t[tindex[pid]:tindex[pid+1],:], bf)
    
  #print "start assembling using", processes, "processes"
  #pool = mp.Pool(processes)
  #results = pool.map(assembleMatrix, ((p,t[:m,:],bf),(p,t[m+1:,:],bf),))
  #A = results[0]
  #for i in range(2,len(results)):
    #A += results[i]

  #return A


#def massParallel(p,t,processes):
  #import multiprocessing as mp
  ##processes = int(0.5*mp.cpu_count())
  #print "start assembling of", t.shape[0], "triangles using", processes, "processes"
  #tpp = int(t.shape[0]/processes)+1
  #process = []
  #queue = mp.Queue(processes)
  #for i in range(processes):
    #if i < processes-1:
      #ti=t[i*tpp:(i+1)*tpp,:]
    #else:
      #ti=t[i*tpp:,:]        
    #process.append(mp.Process(target=assembleMassQueue,args=(p,ti,queue)))
    #print "process", i, "created with", ti.shape[0], "triangles"  
  #for i in range(processes):
    #process[i].start()
    #print "process", i, "started" 
  #A = lil_matrix((p.shape[0],p.shape[0]))
  #for i in range(processes):
    #A += queue.get()
  ##for i in range(processes):
    ##process[i].join()
    ##print "process", i, "finished" 
    ##A += queue.get()
  #return A, processes

  
def massParallel(p,t,processes):
  import multiprocessing as mp
  #processes = int(0.5*mp.cpu_count())
  tpp = int(t.shape[0]/processes)+1
  queue = mp.Queue(processes)
  for i in range(processes):
    if i < processes-1:
      ti=t[i*tpp:(i+1)*tpp,:]
    else:
      ti=t[i*tpp:,:]        
    mp.Process(target=assembleMassQueue,args=(p,ti,queue)).start()
  A = lil_matrix((p.shape[0],p.shape[0]))
  for i in range(processes):
    A += queue.get()
  return A, processes


def assembleMassQueue(p,t,q):
  A = lil_matrix((p.shape[0],p.shape[0]));
  for m in range(0,t.shape[0]):
    AK = elemMass(p[t[m,:],:])
    for i in range(0,3):
      for j in range(0,3):
        A[t[m,i],t[m,j]] += AK[i,j]
  q.put(A)

  
#def massPool(p,t):
  #import multiprocessing as mp
  #processes = mp.cpu_count()
  #print "start assembling of", t.shape[0], "triangles using", processes, "processes"
  #sharedp0 = mp.Array('d',p[:,0])
  #sharedp1 = mp.Array('d',p[:,1])
  #sharedt0 = mp.Array('i',t[:,0])
  #sharedt1 = mp.Array('i',t[:,1])
  #sharedt2 = mp.Array('i',t[:,2])
  #sharedi1 = mp.Array('i',np.zeros((processes),dtype=int))
  #sharedi2 = mp.Array('i',np.zeros((processes),dtype=int))
  #tpp = int(t.shape[1]/processes)+1
  #for i in range(processes):
    #sharedi1[i] = i*tpp
    #if i < processes-1:
      #sharedi2[i] = (i+1)*tpp
    #else:
      #sharedi2[i] = t.shape[0]
  #pool = mp.Pool(processes)
  #results = pool.map(assembleMassPool, range(processes))
  #A = results[0]
  #for i in range(2,len(results)):
    #A += results[i]
  #return A, processes


#def assembleMassPool(process):
  #A = lil_matrix((len(sharedp0),len(sharedp0)))
  ##t = sharedt[sharedt1[process]:sharedt2[process],:]
  #for m in range(sharedi1[process],sharedi2[process]):
    #points =                  [sharedp0[sharedt0[m]],sharedp1[sharedt0[m]]]
    #points = np.hstack(points,[sharedp0[sharedt1[m]],sharedp1[sharedt1[m]]])
    #points = np.hstack(points,[sharedp0[sharedt2[m]],sharedp1[sharedt2[m]]])             
    #AK = elemMass(points)
    #for i in range(0,3):
      #for j in range(0,3):
        #A[t[m,i],t[m,j]] += AK[i,j]
  #return A
  