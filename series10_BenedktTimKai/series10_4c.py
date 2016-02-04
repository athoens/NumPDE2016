import FEM as FE
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import math as m
import meshes as mesh
import matplotlib.pyplot as plt
import series10_3




def f(x,y):
    return 1
    
    
def discSol(h):
    
  (p,t,e,z,be)=mesh.generate_quad_adapt_triangulation(h,1,False)
  
  width=mesh.max_mesh_width()
  nDN=FE.notDiricNodes(p,t,be)
    
  Stiff=FE.stiffness(p,t).tocsr()
  #Mass=FE.mass(p,t).tocsr()
  Load=FE.load(p,t,3,f)

  N=sp.lil_matrix((nDN.size,p[0:,0].size))

  for j in range(nDN.size):         # Initialisierung von Reduktionsmatrizen, aehnlich der T Matrizen.
    N[j,nDN[j]]=1                 # Dies sind quasi NxN Einheitsmatrizen bei denen die Zeilen entfernt
                                 # sind, deren Indizes mit denen der Boundary Nodes korrelieren.

  
 
  rStiff=N.dot(Stiff).dot(N.transpose()) # Durch Multiplikation der N Matrizen von Links und Rechts werden die
  #rMass=N.dot(Mass).dot(N.transpose())
  rLoad=N.dot(Load)

  run=spla.spsolve(rStiff,rLoad)
  un=N.transpose().dot(run)  
  


  """ 
  if h==0.2:
      FE.plot(p,t,un)
      plt.title('Diskrete Loesung')
      plt.show()
  #"""  
    
  return (np.dot(un, Load),width) #nDN.size  
  

M=5  #Anzahl an Iterationen
e=np.zeros((M,1))
dof=np.zeros((M,1))
fu=1.548888           #Wert des Integrals über f*u.


for i in range(M):
    
    (fun,width)=discSol((0.4)*(1/m.pow(2,i/3.0))) #Integral über diskrete Lösung sowie des tatsächliche Gitterweite werden berechnet
    dof[i,0]=width
    e[i,0]=m.sqrt(m.fabs(fu*fu-fun))                 # Fehler in der Energienorm
    

plt.loglog((dof),(e))
plt.title('Fehler in Energienorm gegen die Gitterweite')
plt.show()
print('The solution on a graded mesh has a smaller discretization error and also does converges faster than the solution on the uniform grid.')




