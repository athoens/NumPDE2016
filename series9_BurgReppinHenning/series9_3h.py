import FEM as FE
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import math as m
import meshes as mesh
import matplotlib.pyplot as plt



def u(x,y):
  return m.cos(m.pi*2*x)*m.cos(m.pi*2*y)

def f(x,y):
  return (8*m.pi*m.pi+1)*u(x,y)


def discSol(h):
    
  (p,t,v,z,be)=mesh.generate_gmsh(h,1,True)
  E=mesh.edgeIndex(p,t)
  width=mesh.max_mesh_width()
  N=p[:,0].size
  Stiff=FE.stiffnessP2(p,t,E).tocsr()
  Mass=FE.massP2(p,t,E).tocsr()
  Load=FE.loadP2(p,t,3,f,E)

  un=spla.spsolve(Mass+Stiff,Load)

  #uc=np.zeros((p[0:,0].size))
  #for j in range(p[0:,0].size):
  #  uc[j]=u(p[j,0],p[j,1])
  
  return (np.dot(un[0:N], Load[0:N]),width)
  
  

e=np.zeros((6,1))
h=np.zeros((6,1))
fu=(8*m.pi*m.pi+1)/4            #Wert des Integrals über f*u.


for i in range(6):
    
    (fun,hActual)=discSol((0.1)*(1/m.pow(2,i/3))) #Integral über diskrete Lösung sowie des tatsächliche Gitterweite werden berechnet
    h[i,0]=hActual
    e[i,0]=m.sqrt(m.fabs(fu-fun))                  # Fehler in der Energienorm
    

plt.loglog((h),(e),'o')

plt.title('Fehler in Energienorm gegen Gitterweite')
plt.show()
print('Die Konvergenzrate ist quadratisch.')






