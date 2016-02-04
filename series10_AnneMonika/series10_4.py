import meshes as mesh
import FEM as fem
import scipy.sparse as sp
import numpy as np
import scipy.sparse.linalg as spla
print('comparing mesh sizes (4a)')
p,t,be=mesh.read_gmsh('bgmesh0.msh')
print(len(p))
p,t,be=mesh.read_gmsh('bgmesh1.msh')
print(len(p))


def main(msh,n):
    p,t,be=mesh.read_gmsh(msh)
    h=mesh.max_mesh_width(p,t)
    IN=fem.interiorNodes(p,t,be)
    S=fem.stiffness(p,t)          
    M=fem.mass(p,t)
    L=fem.load(p,t,n,lambda x,y:1)
    S=S.tocsr()
    M=M.tocsr()
    L=L.tocsr()
    
    #neumannnodes
    N=np.zeros(len(p))
    j=0
    for i in range (0,len(p)):
        if (p[i,0]==-1 or p[i,0]==1):
            if 1>p[i,1] and p[i,1]>=0:
                N[j]=i
                j=j+1
        if p[i,1]==1:
            N[j]=i
            j=j+1
    j=j-1 #number of neumannnodes

    Sr=sp.csr_matrix((len(IN)+j,len(IN)+j)) #adjusting solution to boundary condition
    Mr=sp.csr_matrix((len(IN)+j,len(IN)+j))
    Lr=sp.csr_matrix((len(IN)+j,1))
    A=sp.csr_matrix((len(IN)+j,len(p)))
    for i in range(0,len(IN)):
        A[i,IN[i]]=1
        
    for i in range(0,j):
        A[i+len(IN),N[i]]=1
    Sr=A*S*A.transpose()
    Mr=A*M*A.transpose()
    Lr=A*L
    un=spla.spsolve(Sr+Mr,Lr) 
    u1=A.transpose()*un
    fem.plot(p,t,u1)
    #u=u0+sum(psi*alpha*s)
    #error=np.sqrt(abs(u-u1[0]))
    #return error,h

#as we were not able to get the direct solution for u we could not calculate an error
#so the following study of convergence cannot be used
main('mesh01.msh',3)
#e=np.zeros(4)
#ha=np.zeros(4)
#e[0],ha[0]=main('mesh01.msh',3)
#e[1],ha[1]=main('mesh0707.msh',3)
#e[2],ha[2]=main('mesh05.msh',3)
#e[3],ha[3]=main('mesh035.msh',3)
#plt.pyplot.loglog(ha,e,'-o')
#plt.pyplot.xlabel("max width h")
#plt.pyplot.ylabel("error of energy norm")
#plt.pyplot.show()
#print("rate of convergence")
#print(np.diff(np.log(e))/np.diff(np.log(ha)))
