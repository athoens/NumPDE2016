#Martin Plonka 337 266
#Richard Luetzke 334 012
import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.sparse as sps
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
      print ('numerical integration of order ' + str(n) + 'not available');
      
  return x, w


#
# plot(p,t,u)
#
# plots the linear FE function u on the triangulation t with nodes p
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
#subplot
def subplot(splt,p,t,u):
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = splt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_trisurf(p[:, 0], p[:, 1], t, u, cmap=plt.cm.jet)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('u')
  
#return T matrix for an element K=[node 1,node 2,node 3] as sparse matrix
def getTMatrix(element,N):
     T = sps.lil_matrix((3,N))
     T[0,element[0]] = 1
     T[1,element[1]] = 1
     T[2,element[2]] = 1
     T1 = T.tocsr()
     return T1

#returns the D_k matrix for a point p
def getDK(p):
     D = np.zeros([2,3],float)
     D[0,0] = p[1,1] - p[2,1]
     D[0,1] = p[2,1] - p[0,1]
     D[0,2] = p[0,1] - p[1,1]
     D[1,0] = p[2,0] - p[1,0]
     D[1,1] = p[0,0] - p[2,0]
     D[1,2] = p[1,0] - p[0,0]
     return D

#returns the transformation matrix F_K
#for a triangle (a,b,c)
def getFK(p):
     F = np.zeros([2,2],float)
     F[0,0] = p[1,0] - p[0,0]
     F[0,1] = p[2,0] - p[0,0]
     F[1,0] = p[1,1] - p[0,1]
     F[1,1] = p[2,1] - p[0,1]
     return F
#computes the element stiffness matrix related to the bilinear form
#a_K(u,v) = int_K grad u . grad v dx
#for LINEAR FEM on triangles (aK = 1)
#input p = 3x2 matrix of the coordinates of the triangle nodes
#output AK element stiffness 3x3 matrix
def elemStiffness(p):
     #D_K matrix
     D = getDK(p)
     #F_K matrix
     F = getFK(p)
     Vol = la.det(F)
     A = 0.5*np.dot(np.transpose(D),D)/Vol
     return A
     
#computes the element mass matrix related to the bilinear form
#m_K(u,v) = int_K u.v dx
#for LINEAR FEM on triangles (cK = 1)
#input p = 3x2 matrix of the coordinates of the triangle nodes
#output MK element mass matrix 3x3
def elemMass(p):
     F = getFK(p)
     Vol = la.det(F)/2
     M = np.zeros((3,3))
     for i in range(0,3):
          for j in range(0,3):
               if i!=j :
                    M[i,j] = 1/12 * Vol
               else :
                    M[i,j] = 1/6 * Vol
     
     return M
#returns the element load vevtor related to the linear form
#l_K(v) = int_K f.v dx
#for LINEAR FEM
#input p= 3x2 node coordinates, n = order of quadrature (1<=n<=5),f source term function
#output fK element load vector (3x1)

#f should be a standard Python function
def elemLoad(p,n,f):
     u,w = gaussTriangle(n)
     F = getFK(p)
     Vol = la.det(F)
     L = np.zeros((3,1))
     def phiK(x,y):
          return np.transpose(F.dot([x,y])) + p[0,:]
     sum = 0
     for i in range(0,len(w)):
          a = phiK((u[i][0]+1)/2,(u[i][1]+1)/2)
          L[2,0] = L[2,0] + w[i]*f(a[0],a[1])*(u[i][1]+1)*Vol/8
          L[1,0] = L[1,0] + w[i]*f(a[0],a[1])*(u[i][0]+1)*Vol/8
          sum = sum + w[i]*f(a[0],a[1])*Vol/4
     L[0,0] = sum - L[1,0] - L[2,0]
     return L
#returns the stiffness matrix related to the bilinear form
#int_omega grad u . grad v dx
#for LINEAR FEM on triangles
#input p = Nx2, t = Mx3
#output Stiff = NxN stiffness matrix in scipy's sparse lil format
def stiffness(p,t):
     #uses elemStiffness for each element
     A = sps.lil_matrix((len(p),len(p)))
     for i in range(0,len(t)):
          element = np.zeros((3,2))
          element[0,0] = p[t[i,0],0]
          element[0,1] = p[t[i,0],1]
          element[1,0] = p[t[i,1],0]
          element[1,1] = p[t[i,1],1]
          element[2,0] = p[t[i,2],0]
          element[2,1] = p[t[i,2],1]
          TK = getTMatrix(t[i,:],len(p))
          AK = sps.csr_matrix(elemStiffness(element))
          A = A + TK.transpose().dot(AK.dot(TK))
     return A
#returns the mass matrix related to the bilinear form
#int_omega u . v dx
#for LINEAR FEM on trianlges
#input p = Nx2, t = Mx3
#output Mass = NxN mass matrix in scipy's sparse lil format
def mass(p,t):
     #uses elemMass
     M = sps.lil_matrix((len(p),len(p)))
     for i in range(0,len(t)) :
          element = np.zeros((3,2))
          element[0,0] = p[t[i,0],0]
          element[0,1] = p[t[i,0],1]
          element[1,0] = p[t[i,1],0]
          element[1,1] = p[t[i,1],1]
          element[2,0] = p[t[i,2],0]
          element[2,1] = p[t[i,2],1]
          TK = getTMatrix(t[i,:],len(p))
          MK = sps.csr_matrix(elemMass(element))
          M = M + TK.transpose().dot(MK.dot(TK))
          
     return M
     
#returns the load vector related to the linear form
#int_omega f v dx
#input p = Nx2, t = Mx3, n order in [1,5], f source term function
#output Load Nx1 = load vector as numpy array
def load(p,t,n,f):
     #uses elemLoad
     L = np.zeros((len(p),1))
     for i in range (0,len(t)) :
          element = np.zeros((3,2))
          element[0,0] = p[t[i,0],0]
          element[0,1] = p[t[i,0],1]
          element[1,0] = p[t[i,1],0]
          element[1,1] = p[t[i,1],1]
          element[2,0] = p[t[i,2],0]
          element[2,1] = p[t[i,2],1]
          TK = getTMatrix(t[i,:],len(p)).transpose()
          LK = elemLoad(element,n,f)
          L = L + TK.todense().dot(LK)
     return L
#returns the interior nodes as indices into p
#input p = Nx2, t = Mx3, be = Bx2
#output Ix1 array of nodes as indices to p that do not lie on the boundary
def interiorNodes(p,t,be):
     IN = np.array(range(0,len(p)))
     IN = np.delete(IN,be[:,0])
     return IN

#returns element vector related to Neumann boundary data
#int_I g.v dS      I sraight boundary edge
#input:p = 2x2 nodes of the boundary edge, n order, g Neumann data
#output: gK element vector (2x1 array)
def elemLoadNeumann(p,n,g):
     u,w = np.polynomial.legendre.leggauss(n)
     mE = np.zeros((2,1))
     sum1 = 0
     sum2 = 0
     c = 0.25*la.norm(p[1,:]-p[0,:])
     for i in range(0,len(w)):
          a = p[0,:] + (p[1,:]-p[0,:])*(1+u[i])/2
          sum1 = sum1 + w[i]*g(a[0],a[1])
          sum2 = sum2 + w[i]*g(a[0],a[1])*u[i]
     mE[0,0] = (sum1 - sum2)*c
     mE[1,0] = (sum1 + sum2)*c
     return mE
     
#returns the m vector for Neumann data
# int_dOmega g.v dS
#input: p Nx2 nodes, be = Bx2 boundary edges, n order, g Neumann data
#output: Nx1 np.array
def loadNeumann(p,be,n,g):
     #uses elemLoad
     LN = np.zeros((len(p),1))
     for i in range (0,len(be)) :
          element = np.zeros((2,2))
          element[0,0] = p[be[i,0],0]
          element[0,1] = p[be[i,0],1]
          element[1,0] = p[be[i,1],0]
          element[1,1] = p[be[i,1],1]
          mE = elemLoadNeumann(element,n,g) #mass edge
          LN[be[i,0],0] = LN[be[i,0],0] + mE[0,0]
          LN[be[i,1],0] = LN[be[i,1],0] + mE[1,0]
     return LN

#elemStiffnessP2(p)
#input p = 3x2 triangle nodes
#returns 6x6 matrix corresponding
#a_K (u,v) = int_K grad u . grad v
def elemStiffnessP2(p):
     Ak = np.zeros((6,6))
     #D_K matrix
     Dk = getDK(p)
     #F_K matrix
     F = getFK(p)
     Vol = la.det(F)/2 # |K|
     #left upper 3x3 block of Ak
     G = 0.25*np.dot(np.transpose(Dk),Dk)/(Vol*Vol)
     #right upper and left lower (transposed) 3x3 block of Ak
     Pk = np.zeros((3,3))
     Pk[0,0] = G[0,0] + G[0,2]
     Pk[0,1] = G[0,0] + G[0,1]
     Pk[0,2] = G[0,1] + G[0,2]
     Pk[1,0] = G[1,0] + G[1,2]
     Pk[1,1] = G[1,0] + G[1,1]
     Pk[1,2] = G[1,1] + G[1,2]
     Pk[2,0] = G[2,0] + G[2,2]
     Pk[2,1] = G[2,0] + G[2,1]
     Pk[2,2] = G[2,1] + G[2,2]
     #lower right 3x3 block of Ak
     Sk = np.zeros((3,3))
     Sk[0,0] = G[0,0] + G[2,2] + G[0,2]
     Sk[0,1] = (G[0,0] + G[0,1] + G[0,2])/2 + G[1,2]
     Sk[0,2] = (G[0,2] + G[1,2] + G[2,2])/2 + G[0,1]
     Sk[1,0] = (G[0,0] + G[0,1] + G[0,2])/2 + G[1,2]
     Sk[1,1] = G[0,0] + G[1,1] + G[0,1]
     Sk[1,2] = (G[0,1] + G[1,1] + G[1,2])/2 + G[0,2]
     Sk[2,0] = (G[0,2] + G[1,2] + G[2,2])/2 + G[0,1]
     Sk[2,1] = (G[0,1] + G[1,1] + G[1,2])/2 + G[0,2]
     Sk[2,2] = G[1,1] + G[1,2] + G[2,2]
     #writing blocks into Ak
     Ak[0:3,0:3] = G*Vol
     Ak[0:3,3:6] = Vol*Pk/3
     Ak[3:6,0:3] = Vol* np.transpose(Pk)/3
     Ak[3:6,3:6] = Vol*Sk/6
     return Ak

#elemMassP2(p)
#input p = 3x2 triangle nodes
#returns 6x6 matrix corresponding
#m_K (u,v) = int_K u . v
def elemMassP2(p):
     F = getFK(p)
     Vol = la.det(F)/2 # |K|
     Mk = np.zeros((6,6))
     #calculated on paper, could be wrapped up with for-loops
     Mk[0,0] = 1
     Mk[0,1] = 1/2
     Mk[1,0] = 1/2
     Mk[0,2] = 1/2
     Mk[2,0] = 1/2
     Mk[0,3] = 1/5
     Mk[3,0] = 1/5
     Mk[0,4] = 1/5
     Mk[4,0] = 1/5
     Mk[0,5] = 1/10
     Mk[5,0] = 1/10
     Mk[1,1] = 1
     Mk[1,2] = 1/2
     Mk[2,1] = 1/2
     Mk[1,3] = 1/10
     Mk[3,1] = 1/10
     Mk[1,4] = 1/5
     Mk[4,1] = 1/5
     Mk[1,5] = 1/5
     Mk[5,1] = 1/5
     Mk[2,2] = 1
     Mk[2,3] = 1/5
     Mk[3,2] = 1/5
     Mk[2,4] = 1/10
     Mk[4,2] = 1/10
     Mk[2,5] = 1/5
     Mk[5,2] = 1/5
     Mk[3,3] = 1/15
     Mk[3,4] = 1/30
     Mk[4,3] = 1/30
     Mk[3,5] = 1/30
     Mk[5,3] = 1/30
     Mk[4,4] = 1/15
     Mk[4,5] = 1/30
     Mk[5,4] = 1/30
     Mk[5,5] = 1/15
     return Vol*Mk/6


#elemLoadP2(p,n,f)
#input p = 3x2 triangle nodes
#returns 6x1 matrix corresponding
#l_K (v) = int_K f . v 
def elemLoadP2(p,n,f):
     Lk = np.zeros((6,1))
     u,w = gaussTriangle(n)
     F = getFK(p)
     Vol = la.det(F)/2 # |K|
     #transform from reference triangle to element
     def phiK(x,y):
          return np.transpose(F.dot([x,y])) + p[0,:]
     
     for i in range(0,len(w)):
          a = phiK((u[i][0]+1)/2,(u[i][1]+1)/2) #to plug into f
          Lk[2,0] = Lk[2,0] + w[i]*f(a[0],a[1])*(u[i][1]+1)*Vol/4 # x2
          Lk[1,0] = Lk[1,0] + w[i]*f(a[0],a[1])*(u[i][0]+1)*Vol/4 # x2
          Lk[0,0] = Lk[0,0] + w[i]*f(a[0],a[1])*(1-(u[i][0]+1)/2-(u[i][1]+1)/2)*Vol/2 # 1-x1-x2
          Lk[3,0] = Lk[3,0] + w[i]*f(a[0],a[1])*(1-(u[i][0]+1)/2-(u[i][1]+1)/2)*(u[i][1]+1)*Vol/4 #(1-x1-x2)x2
          Lk[4,0] = Lk[4,0] + w[i]*f(a[0],a[1])*(1-(u[i][0]+1)/2-(u[i][1]+1)/2)*(u[i][0]+1)*Vol/4 #(1-x1-x2)x1
          Lk[5,0] = Lk[5,0] + w[i]*f(a[0],a[1])*(u[i][0]+1)*(u[i][1]+1)*Vol/8 #x1 x2
     return Lk
#stiffnessP2(p,t,E)
#input p=nodes, t=elements, E edges
#returns the assembled stiffness matrix for quadratic FE corresponding to
# int_Omega grad u . grad v dx
def stiffnessP2(p,t,E):
     M = len(p) # number of nodes
     N = E[0,0] # number of edges
     A = sps.lil_matrix((M+N,M+N))
     for i in range(0,len(t)):
          nodes = [t[i,0],t[i,1],t[i,2]] #nodes of the elements counted counter clockwise
          edges = [E[nodes[0],nodes[1]],E[nodes[1],nodes[2]],E[nodes[2],nodes[0]]] #edges counter clock wise
          element = np.zeros((3,2)) #physical element
          element[0,0] = p[t[i,0],0]
          element[0,1] = p[t[i,0],1]
          element[1,0] = p[t[i,1],0]
          element[1,1] = p[t[i,1],1]
          element[2,0] = p[t[i,2],0]
          element[2,1] = p[t[i,2],1]
          AK = sps.csr_matrix(elemStiffnessP2(element))
          TK = getT(nodes,edges,M,N)
          A = A + TK.transpose().dot(AK.dot(TK))
     return A
#massP2(p,t,E)
#input p=nodes, t=elements, E edges
#returns the assembled mass matrix for quadratic FE corresponding to
# int_Omega u.v dx
def massP2(p,t,E):
     m = len(p) #number of nodes
     N = E[0,0] #number of edges
     M = sps.lil_matrix((m+N,m+N))
     for i in range(0,len(t)):
          nodes = [t[i,0],t[i,1],t[i,2]]
          edges = [E[nodes[0],nodes[1]],E[nodes[1],nodes[2]],E[nodes[2],nodes[0]]]
          element = np.zeros((3,2)) #physical elements
          element[0,0] = p[t[i,0],0]
          element[0,1] = p[t[i,0],1]
          element[1,0] = p[t[i,1],0]
          element[1,1] = p[t[i,1],1]
          element[2,0] = p[t[i,2],0]
          element[2,1] = p[t[i,2],1]
          MK = sps.csr_matrix(elemMassP2(element))
          TK = getT(nodes,edges,m,N)
          M = M + TK.transpose().dot(MK.dot(TK))
     return M
#loadP2(p,t,E,n,f)
#input p=nodes,t=elements,E=edges,n=quadrature,f=rhs
#returns the load vector for quadratic FE corresponding to
#int_Omega f.v dx
def loadP2(p,t,E,n,f):
     M = len(p) #number of nodes
     N = E[0,0] #number of edges
     L= np.zeros((M+N,1))
     for i in range(0,len(t)):
          nodes = [t[i,0],t[i,1],t[i,2]]
          edges = [E[nodes[0],nodes[1]],E[nodes[1],nodes[2]],E[nodes[2],nodes[0]]]
          element = np.zeros((3,2)) #physical element
          element[0,0] = p[t[i,0],0]
          element[0,1] = p[t[i,0],1]
          element[1,0] = p[t[i,1],0]
          element[1,1] = p[t[i,1],1]
          element[2,0] = p[t[i,2],0]
          element[2,1] = p[t[i,2],1]
          LK = elemLoadP2(element,n,f)
          TK = getT(nodes,edges,M,N).transpose()
          L = L + TK.todense().dot(LK)
     return L
#getT(nodes,edges,M,N)
#input M=global number of nodes, N=global number of edges
#input nodes=global number of local nodes, edges= global number of local edges
#returns returns T matrix needed for P2 FE
def getT(nodes,edges,M,N):
     T = sps.lil_matrix((6,M+N))
     T[0,nodes[0]] = 1
     T[1,nodes[1]] = 1
     T[2,nodes[2]] = 1
     T[3,M+edges[2]-1] = 1
     T[4,M+edges[0]-1] = 1
     T[5,M+edges[1]-1] = 1
     T1 = T.tocsr()
     return T1