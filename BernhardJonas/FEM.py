# FUNCTIONS FOR A FINITE ELEMENT SOLVER
# authors: Bernhard Aigner (359706)
#          Jonas Gienger   (370058)

# Import needed modules
import numpy as np
import scipy.sparse as sp
import math as m
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ---------------------------MESH STUFF---------------------------------
# interiorNodes(p, t, be):
# 
# returns the interior nodes of a triangulation as indices of p
# 
# input:
# p - Nx2-matrix with coordinates of the nodes
# t - Mx3-matrix with inidces of nodes of the triangles
# be - Bx2-matrix with indices of nodes on boundary edges
# 
# output:
# IN - Ix1-array of nodes as indices into p that do not lie on the boundary

def interiorNodes(p, t, be):
   
   N = np.shape(p)[0];
   NBE = np.shape(be)[0];
   bla = np.ones((N, 1), dtype=bool);
   for i in range(0, NBE):	# labelling indices of boundary nodes
      bla[be[i][0]] = False;
      bla[be[i][1]] = False;
   IN = [None]*(N - NBE);	# list of interior indices
   count = 0;
   for i in range(0,N):		# labelling indices of interior nodes
      if (bla[i][0]):
         IN[count] = i;
         count += 1;
   
   return IN


# -----------------------NUMERICAL QUADRATURE---------------------------
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
      print('numerical integration of order ' + str(n) + 'not available');
      
  return x, w


# -----------------------ELEMENT STUFF----------------------------
# elemStiffness(p)
# 
# returns element stiffness matrix related to a bilinear form a_K(u, v) = int_K grad u . grad v dx for linear FEM on triangles.
# 
# input:
# p - 3x2-matrix of the coordinates of the triangle nodes
# 
# output:
# AK - 3x3-element stiffness matrix

def elemStiffness(p):
   
   FK = [[p[1][0] - p[0][0], p[2][0] - p[0][0]],
         [p[1][1] - p[0][1], p[2][1] - p[0][1]]];	# transformation matrix
   det = FK[0][0]*FK[1][1] - FK[0][1]*FK[1][0];		# computing the determinant
   val = (1/2)*abs(det);				# area of the triangle
   DK = [[p[1][1] - p[2][1], p[2][1] - p[0][1], p[0][1] - p[1][1]], 
         [p[2][0] - p[1][0], p[0][0] - p[2][0], p[1][0] - p[0][0]]];
   DK = np.array(DK);
   AK = (1/(4*val))*np.dot(DK.transpose(), DK);		# element stiffness matrix
   
   return AK


# elemStiffnessP2(p)
# 
# computes the element stiffness matrix related to the bilinear form
#   a_K(u,v) = int_K grad u . grad v dx
# for quadratic FEM on triangles.
# 
# input:
# p - 3x2-matrix of the coordinates of the triangle nodes
# 
# output:
# AK - 6x6 element stiffness matrix

def elemStiffnessP2(p):
  
   FK = [[p[1][0] - p[0][0], p[2][0] - p[0][0]],
         [p[1][1] - p[0][1], p[2][1] - p[0][1]]];	# transformation matrix
   detFK = FK[0][0]*FK[1][1] - FK[0][1]*FK[1][0];	# compute the determinant = 2*area of triangle = 2|K|
   detFK = abs(detFK);					# functional determinant of transformation
   
   # coordinate difference matrix
   DK = [[p[1][1] - p[2][1], p[2][1] - p[0][1], p[0][1] - p[1][1]], 
         [p[2][0] - p[1][0], p[0][0] - p[2][0], p[1][0] - p[0][0]]];
   
   DK = np.array(DK);
    
   # matrix containing the inner product of the gradients of the barycentric coordinates
   GK = np.dot(DK.transpose(),DK)/detFK;		# this is GK multiplied by detFK
   
   # creating the element striffness matrix
   AK = np.zeros((6,6)); 				# allocation
   for i in range(0,3):
     for j in range(0,3):
       # CASE 1: i=0,1,2 and j=0,1,2 (linear shape functions)
       AK[i,j] = GK[i,j]/2.;
       # CASE 2: i=0,1,2 and j=3,4,5 (mixed terms, linear shape functions in i, quadratic shape functions in j)
       AK[i,j+3] = (GK[i,(j+1)%3] + GK[i,(j-1)%3])/6.;
       # CASE 3: i=3,4,5 and j=0,1,2 (mixed terms, linear shape functions in j, quadratic shape functions in i)
       AK[i+3,j] = (GK[(i+1)%3,j] + GK[(i-1)%3,j])/6.;
       # CASE 4: i=3,4,5 and j=3,4,5 (quadratic shape functions)
       AK[i+3,j+3] = (GK[(i+1)%3,(j+1)%3] + GK[(i-1)%3,(j-1)%3]) * (1 + int(i==j));		# int(i==j) is a Kronecker delta
       AK[i+3,j+3] +=           GK[(i-1)%3,(j+1)%3]              * (1 + int( i==(j+1)%3 ));
       AK[i+3,j+3] +=           GK[(i+1)%3,(j-1)%3]              * (1 + int( i==(j+2)%3 ));
       AK[i+3,j+3] /= 24.;
   
   return AK


# elemMass(p)
# 
# returns element mass matrix related to a bilinear form m_K(u, v) = int_K u v dx for linear FEM on triangles.
# 
# input:
# p - 3x2-matrix of the coordinates of the triangle nodes
# 
# output:
# MK - 3x3-element mass matrix

def elemMass(p):
   
   FK = [[p[1][0] - p[0][0], p[2][0] - p[0][0]],
         [p[1][1] - p[0][1], p[2][1] - p[0][1]]];	# transformation matrix
   det = FK[0][0]*FK[1][1] - FK[0][1]*FK[1][0];		# computing the determinant
   val = abs(det);					# functional determinant
   MK = [[2*val, val, val],
         [val, 2*val, val],
         [val, val, 2*val]];
   MK = (1/24)*np.array(MK);				# element mass matrix
   
   return MK


# elemMassP2(p)
# 
# computes the element mass matrix related to the bilinear form
#   m_K(u,v) = int_K u v dx
# for quadratic FEM on triangles.
# 
# input:
# p - 3x2-matrix of the coordinates of the triangle nodes
# 
# output:
# MK - 6x6 element mass matrix

def elemMassP2(p):
   
   FK = [[p[1][0] - p[0][0], p[2][0] - p[0][0]],
         [p[1][1] - p[0][1], p[2][1] - p[0][1]]];	# transformation matrix
   detFK = FK[0][0]*FK[1][1] - FK[0][1]*FK[1][0];	# compute the determinant = 2*area of triangle = 2|K|
   detFK = abs(detFK);					# functional determinant of transformation
   
   # defining the element mass matrix
   MK = np.zeros((6,6));				# allocation
   for i in range(0,3):
     for j in range(0,3):
       # CASE 1: i,j = 0,1,2 (linear shape functions)
       MK[i,j] = (1 + int(i==j))/24.;			# int(i==j) is a Kronecker delta
       # CASE 2: i=0,1,2 and j=3,4,5 (mixed terms, linear shape functions in i, quadratic shape functions in j)
       MK[i+3,j] = (1 + int(i!=j))/120.;
       # CASE 3: i=3,4,5 and j=0,1,2 (mixed terms, linear shape functions in j, quadratic shape functions in i)
       MK[i,j+3] = (1 + int(i!=j))/120.;
       # CASE 4: i=3,4,5 and j=3,4,5 (quadratic shape functions)
       MK[i+3,j+3] = (1 + int(i==j))/360.;
   MK *= detFK;
   
   return MK


# elemLoad(p, n, f)
# 
# returns element load vector related to a linear form l_K(v) = int_K f v dx for linear FEM on triangles.
# 
# input:
# p - 3x2-matrix of the coordinates of the triangle nodes
# n - order of the numerical quadrature (1 <= n <= 5)
# f - source term function
# 
# output:
# fK - 3x1-element load array

def elemLoad(p, n, f):
   
   FK = [[p[1][0] - p[0][0], p[2][0] - p[0][0]],
         [p[1][1] - p[0][1], p[2][1] - p[0][1]]];	# transformation matrix
   det = FK[0][0]*FK[1][1] - FK[0][1]*FK[1][0];		# computing the determinant
   val = abs(det);					# functional determinant
   g1 = lambda x, y: f(FK[0][0]*x + FK[0][1]*y + p[0][0], FK[1][0]*x + FK[1][1]*y + p[0][1])*(1 - x - y);	# integrand 1
   g2 = lambda x, y: f(FK[0][0]*x + FK[0][1]*y + p[0][0], FK[1][0]*x + FK[1][1]*y + p[0][1])*x;			# integrand 2
   g3 = lambda x, y: f(FK[0][0]*x + FK[0][1]*y + p[0][0], FK[1][0]*x + FK[1][1]*y + p[0][1])*y;			# integrand 3
   fK = np.zeros((3,1));
   x, w = gaussTriangle(n);	# points and weights for numerical quadrature
   w = np.array(w);
   steps = np.shape(w)[0];	# number of weights
   for i in range(0, steps):	# numerical quadrature
      fK[0][0] += 0.25*val*w[i]*g1((x[i][0] + 1)/2, (x[i][1] + 1)/2);
      fK[1][0] += 0.25*val*w[i]*g2((x[i][0] + 1)/2, (x[i][1] + 1)/2);
      fK[2][0] += 0.25*val*w[i]*g3((x[i][0] + 1)/2, (x[i][1] + 1)/2);
   
   return fK


# elemLoadP2(p, n, f)
# 
# returns element load vector related to a linear form l_K(v) = int_K f v dx for quadratic FEM on triangles.
# 
# input:
# p - 3x2-matrix of the coordinates of the triangle nodes
# n - order of the numerical quadrature (1 <= n <= 5)
# f - source term function
# 
# output:
# fK - 6x1-element load array

def elemLoadP2(p, n, f):
   
   FK = [[p[1][0] - p[0][0], p[2][0] - p[0][0]],
         [p[1][1] - p[0][1], p[2][1] - p[0][1]]];	# transformation matrix
   detFK = FK[0][0]*FK[1][1] - FK[0][1]*FK[1][0];	# computing the determinant
   detFK = abs(detFK);					# functional determinant
   
   # Transformation to reference triangle
   f_ref = lambda x, y: f(FK[0][0]*x + FK[0][1]*y + p[0][0], FK[1][0]*x + FK[1][1]*y + p[0][1]);
   
   # Integrands corresponding to linear shape functions
   l1 = lambda x, y: f_ref(x,y)*(1 - x - y);		# integrand 1
   l2 = lambda x, y: f_ref(x,y)*x;			# integrand 2
   l3 = lambda x, y: f_ref(x,y)*y;			# integrand 3
   
   # Integrands corresponding to quadratic shape functions
   q1 = lambda x, y: f_ref(x,y)*x*y;			# integrand 1
   q2 = lambda x, y: f_ref(x,y)*(1 - x - y)*y;		# integrand 2
   q3 = lambda x, y: f_ref(x,y)*(1 - x - y)*x;		# integrand 3
   
   # Defining the element load vector
   fK = np.zeros((6,1));	# allocation
   x, w = gaussTriangle(n);	# points and weights for numerical quadrature
   w = np.array(w);
   steps = np.shape(w)[0];	# number of weights
   # numerical quadrature
   for i in range(0, steps):
      # integrands corresponding to linear shape functions
      fK[0][0] += 0.25*detFK*w[i]*l1((x[i][0] + 1)/2, (x[i][1] + 1)/2);
      fK[1][0] += 0.25*detFK*w[i]*l2((x[i][0] + 1)/2, (x[i][1] + 1)/2);
      fK[2][0] += 0.25*detFK*w[i]*l3((x[i][0] + 1)/2, (x[i][1] + 1)/2);
      # integrands corresponding to quadratic shape functions
      fK[3][0] += 0.25*detFK*w[i]*q1((x[i][0] + 1)/2, (x[i][1] + 1)/2);
      fK[4][0] += 0.25*detFK*w[i]*q2((x[i][0] + 1)/2, (x[i][1] + 1)/2);
      fK[5][0] += 0.25*detFK*w[i]*q3((x[i][0] + 1)/2, (x[i][1] + 1)/2);
   
   return fK


# elemLoadNeumann(p, n, g)
# 
# returns element vector related to the Neumann boundary data int_I g v ds for linear FEM on triangles.
# 
# input:
# p - 2x2-matrix of the coordinates of the nodes on the boundary edge
# n - order of the numerical quadrature
# g - Neumann data function
# 
# output:
# gE - 2x1-element load array

def elemLoadNeumann(p, n, g):
   
   E = [[p[1][0] - p[0][0]],
        [p[1][1] - p[0][1]]];			# edge vector
   length = m.sqrt(E[0][0]**2 + E[1][0]**2);	# edge length
   g1 = lambda x: g(E[0][0]*x + p[0][0], E[1][0]*x + p[0][1])*(1 - x);		# integrand 1
   g2 = lambda x: g(E[0][0]*x + p[0][0], E[1][0]*x + p[0][1])*x;		# integrand 2
   gE = np.zeros((2,1));			# initializing output
   x, w = np.polynomial.legendre.leggauss(n);	# points and weights for numerical quadrature
   w = np.array(w);				
   steps = np.shape(w)[0];			# number of quadrature points
   for i in range(0, steps):			# numerical quadrature
      gE[0][0] += length*0.5*w[i]*g1((x[i] + 1)/2);
      gE[1][0] += length*0.5*w[i]*g2((x[i] + 1)/2);
   
   return gE


# ----------------------GLOBAL STUFF------------------------------
# stiffness(p, t)
# 
# returns stiffness matrix related to a bilinear form a(u, v) = int_Omega grad u . grad v dx for linear FEM on triangles.
# 
# input:
# p - Nx2-matrix of the coordinates of the triangle nodes
# t - Mx3-matrix of indices of nodes of the triangle
# 
# output:
# Stiff - NxN-stiffness matrix in scipy's lil format

def stiffness(p, t):
   
   M = np.shape(t)[0];
   N = np.shape(p)[0];
   Stiff = sp.lil_matrix((N, N));			# creates an NxN-sparse zero matrix in lil-format
   for m in range(0, M):
      tK = [t[m][0], t[m][1], t[m][2]];			# indices
      pK = [[p[tK[0]][0], p[tK[0]][1]],			# coordinates
            [p[tK[1]][0], p[tK[1]][1]],
            [p[tK[2]][0], p[tK[2]][1]]];
      AK = elemStiffness(pK);
      for i in range(0, 3):
         for j in range(0, 3):
            Stiff[tK[i], tK[j]] += AK[j][i];		# defining the stiffness-matrix
   
   return Stiff


# stiffnessP2(p, t, e)
# 
# returns stiffness matrix related to a bilinear form a(u, v) = int_Omega grad u . grad v dx for quadratic FEM on triangles.
# 
# input:
# p - Nx2-matrix of the coordinates of the triangle nodes
# t - Mx3-matrix of indices of nodes of the triangle
# e - NxN-matrix in scipy's lil format, adjacency matrix of the undirected mesh, where non-zero entries
#     are the index of the edges between node. Since the matrix is symmetric with no diagonal entries,
#     it is stored as an upper triangular matrix. 
# 
# output:
# Stiff - (N+E)x(N+E)-stiffness matrix in scipy's lil format, where E is the number of edges
#
# numbering convention: the first entries 0,..,N-1 in each direction of 'Stiff' are for the linear basis functions localized on the nodes,
#                       the following entries N,..,N+E-1 are for the quadratic basis functions localized on the edges

def stiffnessP2(p, t,  e):
   
   M = np.shape(t)[0];
   N = np.shape(p)[0];
   E = e.nnz;					# number of non-zero entries = number of edges
   
   Stiff = sp.lil_matrix((N+E, N+E));		# creates an (N+E)x(N+E)-sparse zero matrix in lil-format
   for m in range(0, M):			# loop over all triangles
      tK = [t[m][0], t[m][1], t[m][2]];		# indices
      pK = [[p[tK[0]][0], p[tK[0]][1]],		# coordinates
            [p[tK[1]][0], p[tK[1]][1]],
            [p[tK[2]][0], p[tK[2]][1]]];      
      AK = elemStiffnessP2(pK);  		# 6x6 matrix: indices 0,1,2 for linear basis functions
                                                #             indices 3,4,5 for quadratic basis functions
      # two loops to loop over all nodes and edges
      for i in range(0, 3):
         # find correct edge index for edge (tK[i], tK[i+1]):
         edge = [tK[i], tK[(i+1)%3]];
         edge.sort();	        		# sort such that edge[0]<edge[1]
         ei = e[edge[0],edge[1]];		# global index of this edge
         
         for j in range(0, 3):
            # find correct edge index for edge (tK[j], tK[j+1]):
            edge = [tK[j], tK[(j+1)%3]];
            edge.sort();	        	# sort such that edge[0]<edge[1]
            ej = e[edge[0],edge[1]];		# global index of this edge
            
            Stiff[ tK[i]   , tK[j]    ] += AK[i  , j  ];	# linear-linear terms in stiffness-matrix (indices in tK start from 0)
            Stiff[ tK[i]   , N + ej-1 ] += AK[i  , j+3];	# linear-quadratic terms in stiffness-matrix (indices in e start from 1)
            Stiff[ N + ei-1, tK[j]    ] += AK[i+3, j  ];	# linear-quadratic terms in stiffness-matrix 
            Stiff[ N + ei-1, N + ej-1 ] += AK[i+3, j+3];	# quadratic-quadratic terms in stiffness-matrix  
   
   return Stiff


# mass(p, t)
# 
# returns mass matrix related to a bilinear form m(u, v) = int_Omega u v dx for linear FEM on triangles.
# 
# input:
# p - Nx2-matrix of the coordinates of the triangle nodes
# t - Mx3-matrix of indices of nodes of the triangle
# 
# output:
# Mass - NxN-mass matrix in scipy's sparse lil format

def mass(p, t):
   
   M = np.shape(t)[0];
   N = np.shape(p)[0];
   Mass = sp.lil_matrix((N, N));			# creates an NxN-sparse zero matrix in lil-format
   for m in range(0, M):
      tK = [t[m][0], t[m][1], t[m][2]];			# indices
      pK = [[p[tK[0]][0], p[tK[0]][1]],			# coordinates
            [p[tK[1]][0], p[tK[1]][1]],
            [p[tK[2]][0], p[tK[2]][1]]];
      MK = elemMass(pK);
      for i in range(0, 3):
         for j in range(0, 3):
            Mass[tK[i], tK[j]] += MK[j][i];		# defining the mass-matrix
   
   return Mass


# massP2(p, t, e)
# 
# returns mass matrix related to a bilinear form m(u, v) = int_Omega u v dx for quadratic FEM on triangles.
# 
# input:
# p - Nx2-matrix of the coordinates of the triangle nodes
# t - Mx3-matrix of indices of nodes of the triangle
# e - NxN-matrix in scipy's lil format, adjacency matrix of the undirected mesh, where non-zero entries
#     are the index of the edges between node. Since the matrix is symmetric with no diagonal entries,
#     it is stored as an upper triangular matrix. 
# 
# output:
# Mass - (N+E)x(N+E)-stiffness matrix in scipy's lil format, where E is the number of edges
#
# numbering convention: the first entries 0,..,N-1 in each direction of 'Stiff' are for the linear basis functions localized on the nodes,
#                       the following entries N,..,N+E-1 are for the quadratic basis functions localized on the edges

def massP2(p, t, e):
   
   M = np.shape(t)[0];
   N = np.shape(p)[0];
   E = e.nnz;						# number of non-zero entries = number of edges
   
   Mass = sp.lil_matrix((N+E, N+E));			# creates an (N+E)x(N+E)-sparse zero matrix in lil-format
   for m in range(0, M):
      tK = [t[m][0], t[m][1], t[m][2]];			# indices
      pK = [[p[tK[0]][0], p[tK[0]][1]],			# coordinates
            [p[tK[1]][0], p[tK[1]][1]],
            [p[tK[2]][0], p[tK[2]][1]]];
      MK = elemMassP2(pK);
      
      # loop over all nodes and edges of the triangle
      for i in range(0, 3): 
         # find correct edge index for edge (tK[j], tK[j+1]):
         edge = [tK[i], tK[(i+1)%3]];
         edge.sort();	        			# sort such that edge[0]<edge[1]
         ei = e[edge[0],edge[1]];			# global index of this edge
         
         for j in range(0, 3):
            # find correct edge index for edge (tK[j], tK[j+1]):
            edge = [tK[j], tK[(j+1)%3]];
            edge.sort();	        		# sort such that edge[0]<edge[1]
            ej = e[edge[0],edge[1]];			# global index of this edge
            
            Mass[ tK[i]   , tK[j]    ] += MK[i  , j  ];	# linear-linear terms in mass-matrix (indices in tK start from 0)
            Mass[ tK[i]   , N + ej-1 ] += MK[i  , j+3];	# linear-quadratic terms in mass-matrix (indices in e start from 1)
            Mass[ N + ei-1, tK[j]    ] += MK[i+3, j  ];	# linear-quadratic terms in mass-matrix       
            Mass[ N + ei-1, N + ej-1 ] += MK[i+3, j+3];	# quadratic-quadratic terms in mass-matrix
            
   return Mass


# load(p, t, n, f)
# 
# returns load vector related to a linear form l(v) = int_Omega f v dx for linear FEM on triangles.
# 
# input:
# p - Nx2-matrix of the coordinates of the triangle nodes
# t - Mx3-matrix of indices of nodes of the triangle
# n - order of the numerical quadrature (1 <= n <= 5)
# f - source term function
# 
# output:
# Load - Nx1-load numpy-array

def load(p, t, n, f):
   
   M = np.shape(t)[0];
   N = np.shape(p)[0];
   
   Load = np.zeros((N,1), dtype=np.float_);
   for m in range(0, M):
      tK = [t[m][0], t[m][1], t[m][2]];			# indices
      pK = [[p[tK[0]][0], p[tK[0]][1]],			# coordinates
            [p[tK[1]][0], p[tK[1]][1]],
            [p[tK[2]][0], p[tK[2]][1]]];
      fK = elemLoad(pK, n, f);
      for i in range(0, 3):
         Load[tK[i]][0] += fK[i];			# defining the load-vector
   
   return Load


# loadP2(p, t, e, n, f)
# 
# returns load vector related to a linear form l(v) = int_Omega f v dx for quadratic FEM on triangles.
# 
# input:
# p - Nx2-matrix of the coordinates of the triangle nodes
# t - Mx3-matrix of indices of nodes of the triangle
# e - NxN-matrix in scipy's lil format, adjacency matrix of the undirected mesh, where non-zero entries
#     are the index of the edges between node. Since the matrix is symmetric with no diagonal entries,
#     it is stored as an upper triangular matrix. 
# n - order of the numerical quadrature (1 <= n <= 5)
# f - source term function
# 
# output:
# Load - (N+E)x1-load numpy-array
# 
# numbering convention: the first entries 0,..,N-1 in each direction of 'Stiff' are for the linear basis functions localized on the nodes,
#                       the following entries N,..,N+E-1 are for the quadratic basis functions localized on the edges

def loadP2(p, t, e, n, f):
   
   M = np.shape(t)[0];
   N = np.shape(p)[0];
   E = e.nnz;						# number of non-zero entries = number of edges
   
   Load = np.zeros((N+E,1));
   for m in range(0, M):
      tK = [t[m][0], t[m][1], t[m][2]];			# indices
      pK = [[p[tK[0]][0], p[tK[0]][1]],			# coordinates
            [p[tK[1]][0], p[tK[1]][1]],
            [p[tK[2]][0], p[tK[2]][1]]];
      fK = elemLoadP2(pK, n, f);

      # loop over all nodes and edges of the triangle
      for i in range(0, 3): 
         # find correct edge index for edge (tK[i], tK[i+1]):
         edge = [tK[i], tK[(i+1)%3]];
         edge.sort();	        			# sort such that edge[0]<edge[1]
         ei = e[edge[0],edge[1]];			# global index of this edge
         
         # defining the load-vector
         Load[tK[i]] += fK[i];				# corresponding to nodes
         Load[N + ei-1] += fK[i+3];			# corresponding to edges
   
   return Load


# loadNeumann(p, be, n, g)
# 
# returns vector related to the Neumann boundary data int_Omega g v ds for linear FEM on straight boundary edges.
# 
# input:
# p - Nx2-matrix of the coordinates of the triangle nodes
# be - Bx2-matrix of indices of the nodes on the boundary edges
# n - order of the numerical quadrature
# g - Neumann data function
# 
# output:
# LoadNeumann - Nx1-load numpy-array

def loadNeumann(p, be, n, g):
   
   N = np.shape(p)[0];
   B = np.shape(be)[0];
   LoadNeumann = np.zeros((N,1));			# initializing output
   for m in range(0, B):
      tE = [be[m][0], be[m][1]];			# indices of edge-nodes
      pE = [[p[tE[0]][0], p[tE[0]][1]],
            [p[tE[1]][0], p[tE[1]][1]]];		# coordinates of edge-nodes
      gE = elemLoadNeumann(pE, n, g);
      for i in range(0, 2):
         LoadNeumann[tE[i]][0] += gE[i][0];		# defining the load-vector
   
   return LoadNeumann


# ----------------------GRAPHICAL OUTPUT-------------------------
# plot(p,t,u)
#
# plots the linear FE function u on the triangulation t with nodes p
#
# input:
# p   -  Nx2 matrix with coordinates of the nodes
# t   -  Mx3 matrix with indices of nodes of the triangles
# u   -  Nx1 coefficient vector
# fig -  figure for plotting
# s   -  number indicating indices for subplots
# 
# output:
# ax  -  return axis for graphical output

def plot(p,t,u,fig,s):

  ax = fig.add_subplot(s, projection='3d')
  ax.plot_trisurf(p[:, 0], p[:, 1], t, u, cmap=plt.cm.jet)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('u')

  return ax
