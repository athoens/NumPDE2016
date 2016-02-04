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
 


import numpy as np
import numpy.linalg as la
import math as m
import scipy.sparse as sp



#  
# elemStiffness(p)
# 
# computes the element stiffness matrix to the bilinear form 
#  a_K(u,v)= int_K grad u . grad v dx
#
# input:
# p - 3x2-matrix of the coordinates of the triangle nodes
#
# output:
# AK - element stiffness matrix
#
def elemStiffness(p):

   D=np.zeros((2,3))    #Initialisieren der D-Matrix aus der Uebung.     
   D[0,0]=p[1,1]-p[2,1]
   D[0,1]=p[2,1]-p[0,1]
   D[0,2]=p[0,1]-p[1,1]
   D[1,0]=p[2,0]-p[1,0]
   D[1,1]=p[0,0]-p[2,0]
   D[1,2]=p[1,0]-p[0,0]

   F=np.ones((3,3))
   F[0:3,1:3]=p        # Matrix dessen Determinante doppelte Dreiecksflaeche ist
   
   AK=np.dot(D.transpose(),D)*1/(2*m.fabs(la.det(F))) # Berechnen der Element Stiffness Matrix
   return AK



#  
# elemMass(p)
# 
# computes the element mass matrix to the bilinear form 
#  m_K(u,v)= int_K u v dx
#
# input:
# p - 3x2-matrix of the coordinates of the triangle nodes
#
# output:
# MK - element mass matrix
#
def elemMass(p):

   A=np.ones((3,3))*(1/24)
   A=A+np.diag(np.diag(A))                       # Initialiseren der 'grundlegenden' Massematrix
   F=p[1:3,0:2].transpose()
   F[0:2,0:1]=F[0:2,0:1]-p[0:1,0:2].transpose()
   F[0:2,1:2]=F[0:2,1:2]-p[0:1,0:2].transpose()  # Erstellung der Transformationsmatrix F die Referenzdreieck auf das gegebene Dreieck abbildet
   MK=A*m.fabs(la.det(F))                        # Berechnung der Element Mass Matrix 
   return MK
     
#  
# elemLoad(p,n,f)
# 
# returns the element load vector related to the linear form 
#  l_K(v)= int_K f v dx
# for linear FEM on triangle
#
# input:
# p - 3x2-matrix of the coordinates of the triangle nodes
# n - order of the numerical quadrature (1<=n<=5)
# f - source term function
#
# output:
# fK - element load vector (3x1 array)
#
def elemLoad(p, n, f):

   fK=np.zeros((3,1))
   F=p[1:3,0:2].transpose()                       
   F[0:2,0:1]=F[0:2,0:1]-p[0:1,0:2].transpose()  # Erstellen der Transformationsmatrix 
   F[0:2,1:2]=F[0:2,1:2]-p[0:1,0:2].transpose()
   x,w=gaussTriangle(n)
      

   for j in range(len(x)):
     b=x[j]
     b[0]=(b[0]+1)/2            # Transformation der Stuetzstellen auf Referenzdreieck
     b[1]=(b[1]+1)/2
     bf=np.dot(F,b)+p[0:1,0:2]  # Stuetzstellen nach Transformation auf urspruengliches Dreieck, notwendig fuer f
     
     fK[0]=fK[0]+w[j]*f(bf[0,0],bf[0,1])*(1-b[0]-b[1])   # Aufsummieren der einzelnen Stuetzstellen mal Gewichte

     fK[1]=fK[1]+w[j]*f(bf[0,0],bf[0,1])*(b[0])

     fK[2]=fK[2]+w[j]*f(bf[0,0],bf[0,1])*(b[1])
     

   fK=fK*m.fabs(la.det(F))*1/4     # Berechnen des Element Load Vectors.
   return fK     
   
   

#  
# stiffness(p,t)
# 
# returns the stiffness matrix to the bilinear form 
#  a(u,v)= int_Omega grad u . grad v dx
# for linear FEM on trinagles
#
# input:
# p - Nx2 matrix with coordinates of the nodes
# t - Mx3 matrix with indices if nodes of the triangles
#
# output:
# Stiff - NxN stiffness matrix scipy's sparse lil format
#

def stiffness(p, t):
   N=p[0:,0].size
   Stiff=sp.lil_matrix((N,N))
   for j in range(t[0:,0].size):     #Assemblierung der globalen Stiffness Matrix ueber die T Matrizen
     TK=createTK(p,t[j,0:3])
     pK=TK.dot(p)                                   
     AK=sp.lil_matrix(elemStiffness(pK))          # Berechnung der einzelnen Element Stiffness Matrizen
     Stiff=Stiff+TK.transpose().dot(AK).dot(TK)   # Assemblierung der einzelnen in die globale Stiffness Matrix
 
   return Stiff


#  
# mass(p,t)
# 
# returns the mass matrix to the bilinear form 
#  m(u,v)= int_Omega u v dx
# for linear FEM on trinagles
#
# input:
# p - Nx2 matrix with coordinates of the nodes
# t - Mx3 matrix with indices if nodes of the triangles
#
# output:
# Mass - NxN mass matrix scipy's sparse lil format
#
def mass(p, t):
   N=p[0:,0].size
   Mass=sp.lil_matrix((N,N))
   for j in range(t[0:,0].size):
     TK=createTK(p,t[j,0:3])
     pK=TK.dot(p)
     MK=sp.lil_matrix(elemMass(pK))
     Mass=Mass+TK.transpose().dot(MK).dot(TK)
     
   return Mass
  

#  
# stiffness(p,t,n,f)
# 
# returns the load vector related to the linear form 
#  l(v)= int_Omega f v dx
# for linear FEM on trinagles
#
# input:
# p - Nx2 matrix with coordinates of the nodes
# t - Mx3 matrix with indices if nodes of the triangles
# n - order of the numerical quadrature (1<=n<=5)
# f - source term function
#
# output:
# Load - Nx1 load vector as numpy-array 
#

def load (p, t, n, f):
  N=p[0:,0].size
  Load=np.zeros((N,1))
  for j in range(t[0:,0].size):
   TK=createTK(p,t[j,0:3])
   pK=TK.dot(p)
   lK=elemLoad(pK,n,f)
   Load=Load+TK.transpose().dot(lK)
  
  
  return Load

#
# createTK(p,t)
#
# returns the T matrix related to the triangle that is given by t.
#
# input:
# p - Nx2 array with coordinates of the nodes
# t - 1x3 array with indices of nodes of a triangle 
#
# output:
# TK - T matrix that helps to assemble the global stiffness and mass matrix
#

def createTK (p, t):
  
   TK=sp.lil_matrix((3,p[0:,0].size))
   TK[0,t[0,0]]=1
   TK[1,t[0,1]]=1
   TK[2,t[0,2]]=1
   
   return TK	


#
# interiorNodes(p,t,be)
#
# returns the interior nodes as indices into p.
#
# input:
# p  - Nx2 array with coordinates of the nodes
# t  - Mx3 array with indices of nodes of the triangles
# be - Bx2 array with indices of nodes on boundary edges
#
# output:
# IN - Ix1 array of nodes as indices into p that do not lie on
#      the boundary
#
def interiorNodes(p, t, be):
  check=np.ones((p[0:,0].size))    # Array zur Ueberpruefung der Boundary Nodes 1=Interior, 0=Boundary
  
  for j in range(be[0:,0].size):   # 'Auslesen' der Boundary Nodes aus dem be Array.
     check[be[j,0]]=0
     check[be[j,1]]=0

  IN=np.zeros((check.nonzero()[0].size)) # Array der Interior Nodes von der Groeße der Anzahl aller Einsen (Interior Nodes) in Check
  i=0                                    # Laufindex zur Indizierung in IN
  for j in range(p[0:,0].size):
      if check[j]==1:                    # Wenn j der Index eines Interior Node ist, dann ist der naechste Eintrag in IN dieses j.
         IN[i]=j
         i=i+1
  return IN


def elemLoadNeumann(p,n,g):
    (x,w)=np.polynomial.legendre.leggauss(n)
    gK=np.zeros((2))
    for j in range(len(x)):
     b=(x[j]+1)/2
     bx=p[0,0]+b*(p[1,0]-p[0,0])
     by=p[0,1]+b*(p[1,1]-p[0,1])
     
     gK[0]=gK[0]+w[j]*g(bx,by)*(1-b)
     gK[1]=gK[1]+w[j]*g(bx,by)*b   
    
    return gK*np.sqrt(m.pow(p[0,0]-p[1,0],2)+m.pow(p[0,1]-p[1,1],2))*0.5


def loadNeumann(p,be,n, g):  
    N=p[:,0].size

    gLoad=np.zeros((N,1))
    pK=np.zeros((2,2))
    for j in range(be[0:,0].size):
        pK[0,0:2]=p[be[j,0],0:2]
        pK[1,0:2]=p[be[j,1],0:2]
        gK=elemLoadNeumann(pK,n,g  )
        gLoad[be[j,0],0]=gLoad[be[j,0],0]+gK[0]
        gLoad[be[j,1],0]=gLoad[be[j,1],0]+gK[1]
        
    return gLoad
  



def elemStiffnessP2(p):
    
   D=np.zeros((2,3))    #Initialisieren der D-Matrix aus der Uebung.     
   D[0,0]=p[1,1]-p[2,1]
   D[0,1]=p[2,1]-p[0,1]
   D[0,2]=p[0,1]-p[1,1]
   D[1,0]=p[2,0]-p[1,0]
   D[1,1]=p[0,0]-p[2,0]
   D[1,2]=p[1,0]-p[0,0]
   
   F=np.ones((3,3))
   F[0:3,1:3]=p 
   Area=m.fabs(la.det(F))*0.5
   G=(1/(Area*Area))*0.25*D.transpose().dot(D)
   
   AK=np.zeros((6,6))
   AK[0:3,0:3]=G*Area
   for j in range(3):
        AK[j,3]=(G[j,1]+G[j,2])*Area*(1/3)
        AK[j,4]=(G[j,0]+G[j,2])*Area*(1/3)
        AK[j,5]=(G[j,0]+G[j,1])*Area*(1/3)
        
     
   AK[3:6,0:3]=AK[0:3,3:6].transpose()   
  
   AK[3,3]=(G[1,1]+G[2,2]+G[2,1])*Area*(1/6)
   AK[4,4]=(G[0,0]+G[2,2]+G[0,2])*Area*(1/6)
   AK[5,5]=(G[0,0]+G[1,1]+G[0,1])*Area*(1/6) 
   
   AK[3,4]=(G[0,1]+G[0,2]+G[1,2]   )*Area*(1/12)+G[2,2]*Area*(1/6)
   AK[4,3]=AK[3,4]
   
   AK[3,5]=(G[0,1]+G[0,2]+G[1,2])*Area*(1/12)+G[1,1]*Area*(1/6)
   AK[5,3]=AK[3,5]
   
   AK[4,5]=(G[0,1]+G[2,1]+G[2,0] )*Area*(1/12)+G[0,0]*Area*(1/6)
   AK[5,4]=AK[4,5]

   return AK


def elemMassP2(p):
    
    MK=np.zeros((6,6))
    
    F=np.ones((3,3))
    F[0:3,1:3]=p 
    Area=m.fabs(la.det(F))*0.5
    
    K=np.ones((3,3))*(1/12)    
    K=K+np.eye(3)*(1/12)
    K=K*Area
    
    MK[0:3,0:3]=K
     
    K=np.ones((3,3))*(1/180)
    K=K+np.eye(3)*(1/180)
    K=K*Area

    MK[3:6,3:6]=K    
    
    
    K=np.ones((3,3))*(1/30)
    K=K-np.eye(3)*(1/60)
    K=K*Area 

    MK[0:3,3:6]=K
    MK[3:6,0:3]=K.transpose()

    return MK

def elemLoadP2(p,n,f):
   fK=np.zeros((6,1))
   F=p[1:3,0:2].transpose()                       
   F[0:2,0:1]=F[0:2,0:1]-p[0:1,0:2].transpose()  # Erstellen der Transformationsmatrix 
   F[0:2,1:2]=F[0:2,1:2]-p[0:1,0:2].transpose()
   x,w=gaussTriangle(n)
   Area=m.fabs(la.det(F))*0.5

   for j in range(len(x)):
     b=x[j]
     b[0]=(b[0]+1)/2            # Transformation der Stuetzstellen auf Referenzdreieck
     b[1]=(b[1]+1)/2
     bf=np.dot(F,b)+p[0:1,0:2]  # Stuetzstellen nach Transformation auf urspruengliches Dreieck, notwendig fuer f
     
     fK[0]=fK[0]+w[j]*f(bf[0,0],bf[0,1])*(1-b[0]-b[1])   # Aufsummieren der einzelnen Stuetzstellen mal Gewichte
     
     fK[1]=fK[1]+w[j]*f(bf[0,0],bf[0,1])*(b[0])

     fK[2]=fK[2]+w[j]*f(bf[0,0],bf[0,1])*(b[1])
     
     fK[3]=fK[3]+w[j]*f(bf[0,0],bf[0,1])*(b[0])*(b[1])
     
     fK[4]=fK[4]+w[j]*f(bf[0,0],bf[0,1])*(b[1])*(1-b[0]-b[1])
     
     fK[5]=fK[5]+w[j]*f(bf[0,0],bf[0,1])*(1-b[0]-b[1])*(b[0])
     

   fK=fK*Area*0.5     # Berechnen des Element Load Vectors.
   return fK  
   
   
def createTKP2(p,t,E): 
   N=p[:,0].size
   M=m.floor(E.max())
   TK=sp.lil_matrix((6,N+M))
   TK[0:3,0:N]=createTK(p,t)
   TK[3,E[t[0,1],t[0,2]]+N-1]=1
   TK[4,E[t[0,0],t[0,2]]+N-1]=1
   TK[5,E[t[0,0],t[0,1]]+N-1]=1
   #TK[5,E[t[0',0],t[0,2]]]=1
   
   return TK	

   
def stiffnessP2(p,t,E):

   N=p[0:,0].size
   M=m.floor(E.max())
   Stiff=sp.lil_matrix((N+M,N+M))
   for j in range(t[0:,0].size): #Assemblierung der globalen Stiffness Matrix ueber die T Matrizen
     TK=createTKP2(p,t[j,0:3],E)
     pK=TK[0:3,0:N].dot(p)
     AK=sp.lil_matrix(elemStiffnessP2(pK))          # Berechnung der einzelnen Element Stiffness Matrizen
     Stiff=Stiff+TK.transpose().dot(AK).dot(TK)   # Assemblierung der einzelnen in die globale Stiffness Matrix
 
   return Stiff
    
    

def massP2(p,t,E):
    
   N=p[0:,0].size
   M=m.floor(E.max())
   Mass=sp.lil_matrix((N+M,N+M))
   for j in range(t[0:,0].size):     #Assemblierung der globalen Stiffness Matrix ueber die T Matrizen
     TK=createTKP2(p,t[j,0:3],E)
     pK=TK[0:3,0:N].dot(p)                                 
     MK=sp.lil_matrix(elemMassP2(pK))          # Berechnung der einzelnen Element Stiffness Matrizen
     Mass=Mass+TK.transpose().dot(MK).dot(TK)   # Assemblierung der einzelnen in die globale Stiffness Matrix
 
   return Mass
    

def loadP2(p,t,n,f,E):
    
  N=p[0:,0].size
  M=m.floor(E.max())
  Load=np.zeros((N+M,1))
  for j in range(t[0:,0].size):
   TK=createTKP2(p,t[j,0:3],E)
   pK=TK[0:3,0:N].dot(p)
   lK=elemLoadP2(pK,n,f)
   Load=Load+TK.transpose().dot(lK)
  
  return Load
    
def notDiricNodes(p, t, be):
  check=np.ones((p[0:,0].size))    # Array zur Ueberpruefung der Boundary Nodes 1=Interior, 0=Boundary
  
  for j in range(be[0:,0].size):   # 'Auslesen' der Boundary Nodes aus dem be Array.
     if p[be[j,0],1]==-1:   #-1
        check[be[j,0]]=0
     elif p[be[j,0],0]==1 and p[be[j,0],1]<=0: #1,0
        check[be[j,0]]=0
     elif p[be[j,0],0]==-1 and p[be[j,0],1]<=0: #-1,0
        check[be[j,0]]=0
     
     if p[be[j,1],1]==-1:
        check[be[j,1]]=0
     elif p[be[j,1],0]==1 and p[be[j,1],1]<=0:
        check[be[j,1]]=0
     elif p[be[j,1],0]==-1 and p[be[j,1],1]<=0:
        check[be[j,1]]=0

  nDN=np.zeros((check.nonzero()[0].size)) # Array der Interior Nodes von der Groeße der Anzahl aller Einsen (Interior Nodes) in Check
  i=0                                    # Laufindex zur Indizierung in IN
  for j in range(p[0:,0].size):
      if check[j]==1:                    # Wenn j der Index eines Interior Node ist, dann ist der naechste Eintrag in IN dieses j.
         nDN[i]=j
         i=i+1
  return nDN
    
    