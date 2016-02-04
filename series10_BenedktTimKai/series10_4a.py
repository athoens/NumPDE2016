import FEM as FE
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import math as m
import meshes as mesh
import matplotlib.pyplot as plt


(p,t,e,z,be)=mesh.generate_quad_adapt_triangulation(0.5,1,False)
Diff=np.zeros((8,1))
M=p[:,0].size+t[:,0].size

for j in range(7):
  h=0.5*m.pow(0.5,(j+1)/2.)  
  (p,t,e,z,be)=mesh.generate_quad_adapt_triangulation(h,1,False)
  N=p[:,0].size+t[:,0].size
  Diff[j,0]=N/M
  M=N
  
  
print('The average multiplier of nodes and cells when dividing the mesh width with sqRoot(2) over 7 steps is:')
print(np.mean(Diff))
