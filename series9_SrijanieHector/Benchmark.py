import FEM as fem 
import meshes as msh
import numpy as np 
import scipy as sp 
import scipy.sparse as sparse
import scipy.sparse.linalg as spla

#We create the mesh
h0=np.sqrt(2)/5
mesh=msh.grid_square(1,h0)
# Matrix of nodes
pi=mesh[0]
# Matrix of triangles index
t=mesh[1]
# Matrix of boudary elements vertices
be=mesh[2]


# Lets get the coordinates of a triangle
K=0
p=np.array([pi[i-1] for i in t[K]])

#Lets define first the function f
f= lambda x1,x2: (8*(np.pi**2)+1)*np.cos(2*np.pi*x1)*np.cos(2*np.pi*x2)
# The closest value of h0 to 1 to be able to generate a regular 
h0=np.sqrt(2)/14
n=3


# Lets get another time the mesh
h0=np.sqrt(2)/10
mesh=msh.grid_square(1,h0)
# Matrix of nodes
p=mesh[0]
# Matrix of triangles index
t=mesh[1]
# Matrix of boudary elements vertices
be=mesh[2]
