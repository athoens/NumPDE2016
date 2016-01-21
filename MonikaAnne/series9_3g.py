import meshes as mesh
import FEM as fem
import scipy.sparse as sp
import numpy as np
import scipy.sparse.linalg as spla
def main(h0,n):
    p,t,be=mesh.grid_square(1,h0)
    e,l=mesh.edgeIndex(p,t)
    S=fem.stiffnessP2(p,t,e,l) #assemble matrices with p2 functions
    M=fem.massP2(p,t,e,l)
    L=fem.loadP2(p,t,n,f,e,l)
    S=S.tocsr()
    M=M.tocsr()
    L=L.tocsr()
    un=spla.spsolve(S+M,L)    
    u=np.zeros(len(p))
    for i in range(0,len(p)): #remove edges from solution
        u[i]=un[i]
    fem.plot(p,t,u)
    
    #calculating and plotting error
    fehler=np.zeros(len(u))
    for i in range (0,len(p)):
        fehler[i]=np.cos(2*np.pi*p[i,0])*np.cos(2*np.pi*p[i,1])-u[i]
    fem.plot(p,t,fehler)
    
    return u

def f(x,y):
    return (8*np.pi**2+1)*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)
#main(0.1,3)
