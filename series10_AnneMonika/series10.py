import meshes as mesh
import FEM as fem
import scipy.sparse as sp
import numpy as np
import scipy.sparse.linalg as spla
def main(h0,n):
    p,t,be=mesh.grid_square(2,h0) #shift mesh to origin#
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
        if (p[i,0]==0 or p[i,0]==2):
            if 2>p[i,1] and p[i,1]>=1:
                N[j]=i
                j=j+1
        if p[i,1]==2:
            N[j]=i
            j=j+1
    j=j-1 #number of neumannnodes
    print(N)
    Sr=sp.csr_matrix((len(IN)+j,len(IN)+j))
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
main(0.1,3)
#as we were not able to get the direct solution for u we could not calculate an error
#so the following study of convergence cannot be used


#e=np.zeros(4) 
#ha=np.zeros(4)
#for i in range(0,4):
#	hi=0.1/(np.sqrt(2**i))
#	er,h=main(hi,3)
#	e[i]=er
#	ha[i]=h
#print(ha)
#plt.pyplot.loglog(ha,e,'-o')
#plt.pyplot.xlabel("max width h")
#plt.pyplot.ylabel("error of energy norm")
#plt.pyplot.show()
#print("rate of convergence")
#print(np.diff(np.log(e))/np.diff(np.log(ha)))
