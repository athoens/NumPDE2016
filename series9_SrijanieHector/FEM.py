# coding=utf-8
# Héctor Andrade Loarca
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

import numpy as np
import scipy.sparse as sparse
import math
import meshes as msh

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
# plots the linear FE function u on the triangulation t with nodes p
#
# input:
# p  - Nx2 matrix with coordinates of the nodes
# t  - Mx3 matrix with indices of nodes of the triangles
# u  - Nx1 coefficient vector
#
# I changed it a little to generate the plot as an image
# in the work directory

def plot(p,t,u,file,title):
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_trisurf(p[:, 0], p[:, 1], t, u, cmap=plt.cm.jet)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('u')
  plt.title(title)
  plt.savefig(file)
  plt.close()

# Function that computes the elemStiffness in a triangle

# input
# p - 3x2-matrix of the coordinates of the triangle nodes

#First lets define a function that calculates the area of a triangle
# with input the point of the vertices

def area(p):
  return 1./2.*(np.abs(p[1,0]*p[2,1]-p[2,0]*p[1,1]+
                       p[0,0]*p[1,1]-p[1,0]*p[0,1]+
                       p[2,0]*p[0,1]-p[0,0]*p[2,1]))

def elemStiffness(p):
  #We compute the area of the triangle using shoelace formula
  Area=area(p)
  # We compute the coordinate difference matrix DK in the triangle K
  DK=np.array([[p[1,1]-p[2,1],p[2,1]-p[0,1],p[0,1]-p[1,1]],
               [p[2,0]-p[1,0],p[0,0]-p[2,0],p[1,0]-p[0,0]]])
  # Finally we compute the element stiffness matrix
  return (1/(4*Area))*np.transpose(DK).dot(DK)


#Function that computes the elemMass of element mass matrix
# for a constant coeficcient cK=1

# input
# p - 3x2-matrix of the coordinates of the triangle nodes

def elemMass(p):
  #We compute the area of the triangle using shoelace formula
  Area=area(p)
  #We compute the element mass matrix which have 1/6 in the diagonal and 
  # and 1/12 otherwise
  return (1./12*np.ones((3,3))+1./12*np.eye(3))*Area

# Function that computes the element Load vector
 
# Input:
# p - 3x2 matrix of the coordinates of the triangle nodes
# n - order of the numerical quadrature (1 <= n <= 5)
# f - source term function

# First lets define the function that transform the coordinates from the
# the triangle with points coordinates in p to the reference triangle 

#Input: p - 3x2 matrix of the coordinates of the triangle nodes
#       (x1,x2)- coordinates of evaluation in the reference triangle

def transform(p,x1,x2):
  return p[0]+x1*(p[1]-p[0])+x2*(p[2]-p[0])

# Lets define the shape functions in the reference triangle 
#Input:
#       j - the index of the function
#       (x1,x2)- coordinates of evaluation in the reference triangle

def Nshape(j,x1,x2):
  return [1-x1-x2,x1,x2][j]

# Lets define the function that computes the element load vector

# input:
  # p - 3x2 matrix of the coordinates of the triangle nodes
  # n - order of the numerical quadrature (1 <= n <= 5)
  # f - source term function


def elemLoad(p,n,f):
  #Lets get the vectors and weights in the quadrature
  quad=gaussTriangle(n)
  xquad=quad[0]
  wquad=quad[1]
  #Transformed each element of the quadrature points to np.array
  xquad=map(lambda x: np.array(x),xquad)
  #Lets generate a list of the vector transformed to the reference triangle
  #to integrate in the triangle where the quadrature is defined
  xref=map(lambda x: (x+1)/2,xquad)
  #Lets transformed to the original triangle the new coordinates
  xtrans=map(lambda x:transform(p,x[0],x[1]), xref)
  #The determinant of the Jacobian
  detJ=2*area(p)
  #Finally we obtain the element load vector
  Load=map(lambda i:sum(map(lambda j:wquad[j]*f(xtrans[j][0],xtrans[j][1])
                 *Nshape(i,xref[j][0],xref[j][1])*np.abs(detJ),range(0,len(wquad)))),range(0,3))
  return 1./4*np.array(Load).reshape(3,1)


# Now we gonna assemble the whole elements to obtain the total stiffenss
# matrix


# First lets define a function that gives you the T matrix of a triangle K

# input:
  # t : Mx3 matrix with the triangle-node numbering
  # K : the number of the triangle in the t matrix

def T(t,K):
  # Start the T matrix with just zeros
  n=t.max()
  Tm=np.zeros((3,n))
  # extraxt nodes index in the K triangle
  index=t[K]
  # Put the ones
  Tm[0,index[0]-1]=1
  Tm[1,index[1]-1]=1
  Tm[2,index[2]-1]=1
  #Returning the matrix T in lil sparse format
  return sparse.lil_matrix(Tm)

# Now lets get the global stiffnes matrix

# input:
# p - Nx2 matrix with coordinates of the nodes
# t - Mx3 matrix with indices of nodes of the triangles

def stiffness(p,t):
  # Sum up the matrix element stiffness weighted with the T matrices (the conecction)
  # in lil_matrix format
  stiff=sum(map(lambda K:(T(t,K).transpose()
    .dot(elemStiffness(np.array([p[i-1] for i in t[K]]))))
  .dot(T(t,K).toarray()),range(0,len(t))))
  return sparse.lil_matrix(stiff)

# Now lets get the global mass matrix in the same way that the stiffness

# input:
# p - Nx2 matrix with coordinates of the nodes
# t - Mx3 matrix with indices of nodes of the triangles


def mass(p,t):
  # Sum up the matrix element mass weighted with the T matrices (the conecction)
  # in lil_matrix format
  massm=sum(map(lambda K:(T(t,K).transpose()
    .dot(elemMass(np.array([p[i-1] for i in t[K]]))))
  .dot(T(t,K).toarray()),range(0,len(t))))
  return sparse.lil_matrix(massm)

# Now lets get the global load vector 

# input:
# p - Nx2 matrix with coordinates of the nodes
# t - Mx3 matrix with indices of nodes of the triangles
# n - order of the numerical quadrature (1 <= n <= 5)
# f - source term function

def load(p,t,n,f):
  # Sum up the matrix element mass weighted with the T matrices (the conecction)
  return sum(map(lambda K:(T(t,K).transpose()
    .dot(elemLoad(np.array([p[i-1] for i in t[K]]),n,f))),range(0,len(t))))
  
# Lets define the function to get the interior nodes as indices into p

# input:
# p  - Nx2 array with coordinates of the nodes
# t  - Mx3 array with indices of nodes of the triangles
# be - Bx2 array with indices of nodes on boundary edges
  
def interiorNodes(p, t, be):
  # First get a list of the indices of the nodes in the boundary
  bound=sum(be.tolist(),[])
  # We take out the duplicates
  bound=set(bound)
  # Generate a set of the whole indices of the vertices in the mesh
  indices=set([i for i in range(1,len(p)+1)])
  # Finally we return the substraction of the sets to get the 
  # interior points indices
  return list(indices-bound)



# Function elemLoadNeumman, that returns the element vector related to the
# Neumann boundary data

# Lets define the boundary shape functions in the reference triangle 
#Input:
#       j - the index of the function
#       x - point of evaluation

def Nbound(i,x):
  return [1-x,x][i]

# Lets define the function g parametrized in the line with points
# in p

# input:
# p - 2x2 matrix of the coordinates of the nodes on the boundary
#     edge
# g - Neumann data as standard Python function or Python’s
#     lambda function
# x - point where we want to evaluate the parametrization

def gp(p,g,x):
	# get the point to evaluate 
	point=p[0,:]+x*(p[1,:]-p[0,:])
	return g(point[0],point[1])

#Now the element Load vector for Neumann data

# input:
# p - 2x2 matrix of the coordinates of the nodes on the boundary
#     edge
# n - order of the numerical quadrature
# g - Neumann data as standard Python function or Python’s
#     lambda function

def elemLoadNeumann(p,n,g):
  # Lets get the vectors and weights in the Guass-Legendre quadrature
  x,w=np.polynomial.legendre.leggauss(n)
  # now lets define the final determinant of the transformation in the integral
  det=np.sqrt((p[1,0]-p[0,0])**2+(p[1,1]-p[0,1])**2)/2
  #Finally we can compute the boundary element value
  LoadNeumann=map(lambda i:sum(map(lambda j:w[j]*gp(p,g,(x[j]+1)/2)
  	*Nbound(i,(x[j]+1)/2),range(0,len(w)))),range(0,2))
  return det*np.array(LoadNeumann).reshape(2,1)



# Now lets get the global load vector of the Neumann boundary data

# For that we create a new connection T-matrix for the boundary edges


def TB(be,p,K):
	# Start the TB matrix with just zeros
	nn=len(p)
	Tm=np.zeros((2,nn))
	# extraxt nodes index in the K boundary edge
	index=be[K]
	# Put the ones
	Tm[0,index[0]-1]=1
	Tm[1,index[1]-1]=1
	#Returning the matrix T in lil sparse format
	return sparse.lil_matrix(Tm)


# input:
# p - Nx2 matrix with coordinates of the nodes
# be - Bx3 matrix with indices of nodes of the boundary edges
# n - order of the numerical quadrature
# g - Neumann data

def loadNeumann(p,be,n,g):
	# Sum up the matrix element mass weighted with the T matrices (the conecction)
	return sum(map(lambda K:(TB(be,p,K).transpose()
		.dot(elemLoadNeumann(np.array([p[i-1] for i in be[K]]),n,g))),range(0,len(be))))  



## Nonlinear elements 

## Lets define a function that calculates the integral in a triangle
## with coordinates in the 3x2 array p 
## int_K λ1^(β1)*λ2^(β2)*λ3^(β3) using the formula (1)

def baryint(p,b1,b2,b3):
	return 2*area(p)*(math.factorial(b1)*math.factorial(b2)
		*math.factorial(b3))/math.factorial(b1+b2+b3+2)

## Lets define the function elemStiffnessP2 matrix that returns
## the 6x6 element stiffness quadratic matrix for a triangle K
## with coordinates in the 3x2 array p

# input:
# p - 3x2-matrix of the coordinates of the triangle nodes
#
# output:
# AK - element stiffness matrix
#

def elemStiffnessP2(p):
	#We compute the area of the triangle using shoelace formula
	K=area(p)
	# We compute the coordinate difference matrix DK in the triangle K
	DK=np.array([[p[1,1]-p[2,1],p[2,1]-p[0,1],p[0,1]-p[1,1]],
	           [p[2,0]-p[1,0],p[0,0]-p[2,0],p[1,0]-p[0,0]]])
	# We compute the G_K matrix with the entris (GK)_{i,j}=grad(λ_j)⋅grad(λ_i)
	GK=(1./(4*K**2))*np.transpose(DK).dot(DK)
	#We initialize the entries of the element stiffnessmatrix
	ElemStiff=np.zeros((6,6))
	ElemStiff[0,0]=GK[0,0]*K
	ElemStiff[0,1]=GK[0,1]*K
	ElemStiff[1,0]=ElemStiff[0,1]
	ElemStiff[0,2]=GK[0,2]*K
	ElemStiff[2,0]=ElemStiff[0,2]
	ElemStiff[0,3]=GK[0,1]*baryint(p,0,0,1)+GK[0,2]*baryint(p,0,1,0)
	ElemStiff[3,0]=ElemStiff[0,3]
	ElemStiff[0,4]=GK[0,0]*baryint(p,0,0,1)+GK[0,2]*baryint(p,1,0,0)
	ElemStiff[4,0]=ElemStiff[0,4]
	ElemStiff[0,5]=GK[0,0]*baryint(p,0,1,0)+GK[0,1]*baryint(p,1,0,0)
	ElemStiff[5,0]=ElemStiff[0,5]
	ElemStiff[1,1]=GK[1,1]*K
	ElemStiff[1,2]=GK[1,2]*K
	ElemStiff[2,1]=ElemStiff[1,2]
	ElemStiff[1,3]=GK[1,1]*baryint(p,0,0,1)+GK[1,2]*baryint(p,0,1,0)
	ElemStiff[3,1]=ElemStiff[1,3]
	ElemStiff[1,4]=GK[1,0]*baryint(p,0,0,1)+GK[1,2]*baryint(p,1,0,0)
	ElemStiff[4,1]=ElemStiff[1,4]
	ElemStiff[1,5]=GK[1,0]*baryint(p,0,1,0)+GK[1,1]*baryint(p,1,0,0)
	ElemStiff[5,1]=ElemStiff[1,5]
	ElemStiff[2,2]=GK[2,2]*K
	ElemStiff[2,3]=GK[2,1]*baryint(p,0,0,1)+GK[2,2]*baryint(p,0,1,0)
	ElemStiff[3,2]=ElemStiff[2,3]
	ElemStiff[2,4]=GK[2,0]*baryint(p,0,0,1)+GK[2,2]*baryint(p,1,0,0)
	ElemStiff[4,2]=ElemStiff[2,4]
	ElemStiff[2,5]=GK[2,0]*baryint(p,0,1,0)+GK[2,1]*baryint(p,1,0,0)
	ElemStiff[5,2]=ElemStiff[2,5]
	ElemStiff[3,3]=GK[1,1]*baryint(p,0,0,2)+2*GK[1,2]*baryint(p,0,1,1)+GK[2,2]*baryint(p,0,2,0)
	ElemStiff[3,4]=GK[0,1]*baryint(p,0,0,2)+GK[1,2]*baryint(p,1,0,1)+GK[2,0]*baryint(p,0,1,1)+GK[2,2]*baryint(p,1,1,0)
	ElemStiff[4,3]=ElemStiff[3,4]
	ElemStiff[3,5]=GK[2,1]*baryint(p,1,1,0)+GK[2,0]*baryint(p,0,2,0)+GK[1,1]*baryint(p,1,0,1)+GK[1,0]*baryint(p,0,1,1)
	ElemStiff[5,3]=ElemStiff[3,5]
	ElemStiff[4,4]=GK[2,2]*baryint(p,2,0,0)+2*GK[0,2]*baryint(p,1,0,1)+GK[0,0]*baryint(p,0,0,2)
	ElemStiff[4,5]=GK[2,1]*baryint(p,2,0,0)+GK[2,0]*baryint(p,1,1,0)+GK[0,1]*baryint(p,1,0,1)+GK[0,0]*baryint(p,0,1,1)
	ElemStiff[5,4]=ElemStiff[4,5]
	ElemStiff[5,5]=GK[1,1]*baryint(p,2,0,0)+2*GK[0,1]*baryint(p,1,1,0)+GK[0,0]*baryint(p,0,2,0)
	return ElemStiff


## Now we will define a function that compute the element mass matrix in a triangle
## with vertices at p

# input:
# p - 3x2-matrix of the coordinates of the triangle nodes
#
# output:
# MK - element stiffness matrix
#

def elemMassP2(p):
	# We define an array with the exponents in λ_i for the shape fucntions
	Shape=np.array([[1,0,0],[0,1,0],[0,0,1],[0,1,1],[1,0,1],[1,1,0]])
	# We initialize the element mass matrix
	ElemMass=np.zeros((6,6))
	# Now we insert the entries in the matrix using the integral formula for 
	# the barycentric coordinates
	for i in range(6):
		for j in range(6):
			# Compute the exponents of the λs in 
			exp=Shape[i]+Shape[j]
			ElemMass[i,j]=baryint(p,exp[0],exp[1],exp[2])
	return ElemMass



# Function that computes the element Load vector for quadratic elements
 
# Input:
# p - 3x2 matrix of the coordinates of the triangle nodes
# n - order of the numerical quadrature (1 <= n <= 5)
# f - source term function

# Function that calculates the orthogonal of a two-dimensional vector

def ort(v):
	return np.array([-v[1],v[0]]) 

# Function that computes the j-th barycentric coordinates for a triangle
# with indices in p, at point x and j going from 1 to 3

def lamb(j,p,x1,x2):
	# Compute the area
	K=area(p)
	# Return the value
	return (1./(2*K))*(np.array([x1,x2])-p[j%3]).dot(ort(p[(j-2)%3]-p[j%3]))

# Lets define the quadratic finite elements functions with j=0,...,5

def Nquad(j,p,x1,x2):
	#Define the function
	nquad=[lamb(1,p,x1,x2),lamb(2,p,x1,x2),lamb(3,p,x1,x2),
	lamb(2,p,x1,x2)*lamb(3,p,x1,x2),lamb(1,p,x1,x2)*lamb(3,p,x1,x2),
	lamb(1,p,x1,x2)*lamb(2,p,x1,x2)]
	return nquad[j]

# Lets define the function that computes the element load vector

# input:
  # p - 3x2 matrix of the coordinates of the triangle nodes
  # n - order of the numerical quadrature (1 <= n <= 5)
  # f - source term function


def elemLoadP2(p,n,f):
	#Lets get the vectors and weights in the quadrature
	quad=gaussTriangle(n)
	xquad=quad[0]
	wquad=quad[1]
	#Transformed each element of the quadrature points to np.array
	xquad=map(lambda x: np.array(x),xquad)
	#Lets generate a list of the vector transformed to the reference triangle
	#to integrate in the triangle where the quadrature is defined
	xref=map(lambda x: (x+1)/2,xquad)
	#Lets transformed to the original triangle the new coordinates
	xtrans=map(lambda x:transform(p,x[0],x[1]), xref)
	#The determinant of the Jacobian
	detJ=2*area(p)
	#Finally we obtain the element load vector
	Load=map(lambda i:sum(map(lambda j:wquad[j]*f(xtrans[j][0],xtrans[j][1])
	             *Nquad(i,p,xtrans[j][0],xtrans[j][1])*np.abs(detJ),range(0,len(wquad)))),range(0,6))
	return 1./4*np.array(Load).reshape(6,1)



# Now we gonna assemble the whole quadratic elements to obtain the total stiffenss
# matrix

# Lets define the T matrix correspondent to the quadratic element methods

# input:

  # t : array with the triangles
  # emi : matrix with incidences of edges in nodes
  # K : the number of the triangle in the t matrix

def TP2(t,emi,K):
	#First lets calculate the matrix correspondent to the nodes in a triangle
	Tnode=T(t,K)
	# Number of nodes
	n=Tnode.shape[1]
	# Start the Tedge matrix with just zeros
	m=int(emi.max())
	Tedge=np.zeros((3,m))
	# extract nodes index in the K triangle
	index=t[K]
	# Get the index in the emi array of the correspondence edge numbering
	edgenumb1=int(emi[index[0]-1,index[1]-1])
	edgenumb2=int(emi[index[1]-1,index[2]-1])
	edgenumb3=int(emi[index[2]-1,index[0]-1])
	# Put the ones
	Tedge[0,edgenumb1-1]=1
	Tedge[1,edgenumb2-1]=1
	Tedge[2,edgenumb3-1]=1
	# Now create a block matrix with one left upper block the Tnode matrix
	# and the right lower block with Tedge matrix
	Ttot=np.zeros((6,n+m))
	#Insert the slices
	Ttot[0:3,0:n]=Tnode.toarray()
	Ttot[3:6,0:m]=Tedge
	#Returning the matrix T in lil sparse format
	return sparse.lil_matrix(Ttot)

# Now lets get the global stiffnes matrix for quadratic elements

# input:
# p - Nx2 matrix with coordinates of the nodes
# t - Mx3 matrix with indices of nodes of the triangles

def stiffnessP2(p,t):
  #Compute the matrix with node-edge indices
  emi=msh.edgeIndex(p,t)[1]
  # Sum up the matrix element stiffness weighted with the T matrices (the conecction)
  # in lil_matrix format
  stiff=sum(map(lambda K:(TP2(t,emi,K).transpose()
  	.dot(elemStiffnessP2(np.array([p[i-1] for i in t[K]]))))
  .dot(TP2(t,emi,K).toarray()),range(0,len(t))))
  return sparse.lil_matrix(stiff)

# Now lets get the global mass matrix in the same way that the stiffness

# input:
# p - Nx2 matrix with coordinates of the nodes
# t - Mx3 matrix with indices of nodes of the triangles


def massP2(p,t):
  #Compute the matrix with node-edge indices
  emi=msh.edgeIndex(p,t)[1]
  # Sum up the matrix element mass weighted with the T matrices (the conecction)
  # in lil_matrix format
  massm=sum(map(lambda K:(TP2(t,emi,K).transpose()
  	.dot(elemMassP2(np.array([p[i-1] for i in t[K]]))))
  .dot(TP2(t,emi,K).toarray()),range(0,len(t))))
  return sparse.lil_matrix(massm)

# Now lets get the global load vector 

# input:
# p - Nx2 matrix with coordinates of the nodes
# t - Mx3 matrix with indices of nodes of the triangles
# n - order of the numerical quadrature (1 <= n <= 5)
# f - source term function

def loadP2(p,t,n,f):
  #Compute the matrix with node-edge indices
  emi=msh.edgeIndex(p,t)[1]
  # Sum up the matrix element mass weighted with the T matrices (the conecction)
  return sum(map(lambda K:(TP2(t,emi,K).transpose()
  	.dot(elemLoadP2(np.array([p[i-1] for i in t[K]]),n,f))),range(0,len(t))))


