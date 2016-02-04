#Martin Plonka 337 266
#Richard Lützke 334012
# reading Gmsh meshes
# generating uniform meshes for give max width for the unit square

import scipy.sparse as sps
import numpy as np
import matplotlib.pyplot as plt

#generates a uniform grid of the square (0,0) to (l,l) with max width h0
#and returns [p,t,be]
def grid_square(l,h0):
    #number of nodes on one boundary side -1
    #number of subedges per side
    N = int(np.ceil(np.sqrt(2)*l/h0))
    
    #actual height
    # longest edge per triangle
    h = np.sqrt(2) * l /N
    
    #short edge
    hs = h/np.sqrt(2)
    
    #setting up the matrizes
    p = np.zeros(((N+1)*(N+1),2)) # (N+1)*(N+1) nodes with 2 coordinates
    t = np.zeros((2*N*N,3),int) # 2 *N*N triangles with 3 nodes (indices)
    be = np.zeros((4*N,2),int) # 4*N boundary edges with 2 nodes (indices)
    
    #setting the node coordinates from bottem left (0,0) to upper right (l,l)
    for i in range(N+1):
            for j in range(N+1):
                p[j+i*(N+1),0] = j*hs
                p[j+i*(N+1),1] = i*hs
    
    #setting element/triangles indices
    #counter clock wise per triangle
    #triangles from bottom left to upper right
    for i in range(N):
        for j in range(N):
            #lower triangle /|
            t[i*2*N +j,0] = i*(N+1) +j
            t[i*2*N +j,1] = i*(N+1) +j +1
            t[i*2*N +j,2] = (i+1)*(N+1) +j +1
            
            #upper triange |/
            t[i*2*N + j +N,0] = i*(N+1) +j
            t[i*2*N + j +N,1] = (i+1)*(N+1) +j +1
            t[i*2*N + j +N,2] = (i+1)*(N+1) +j
    
    #setting the boundary edges
    for i in range(N):
        #(0,0) to (1,0)
        be[i,0] = i
        be[i,1] = i+1
        #(1,0) to (1,1)
        be[N+i,0] = N + i*(N+1)
        be[N+i,1] = N + (i+1)*(N+1)
        #(1,1) to (0,1)
        be[2*N + i,0] = N*(N+2) - i #(N+1)*(N+1) -1 -i
        be[2*N + i,1] = N*(N+2) - i - 1
        #(0,1) to (0,0)
        be[3*N + i,0] = (N+1)*(N - i)
        be[3*N + i,1] = (N+1)*(N - i - 1)
    
    #return
    return [p,t,be]

#plotting the mesh
def show(p,t):
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    #plotting nodes
    plt.plot(p[:,0],p[:,1],'r.')
    
    #plotting triangles
    for i in range(len(t)):
        plt.plot([p[t[i,0],0],p[t[i,1],0]],[p[t[i,0],1],p[t[i,1],1]],'b-')
        plt.plot([p[t[i,1],0],p[t[i,2],0]],[p[t[i,1],1],p[t[i,2],1]],'b-')
        plt.plot([p[t[i,2],0],p[t[i,0],0]],[p[t[i,2],1],p[t[i,0],1]],'b-')
        
    plt.show()

#calculates the max triangle width h for the mesh defined by p,t
def max_mesh_width(p,t):
    h=0
    #calculating all edge lengths^2 for all triangles (x,y,z) in |.|_2
    for i in range(len(t)):
        xy = np.square(p[t[i,0],0] - p[t[i,1],0]) + np.square(p[t[i,0],1] - p[t[i,1],1])
        yz = np.square(p[t[i,1],0] - p[t[i,2],0]) + np.square(p[t[i,1],1] - p[t[i,2],1])
        zx = np.square(p[t[i,2],0] - p[t[i,0],0]) + np.square(p[t[i,2],1] - p[t[i,0],1])
        h = np.maximum(np.maximum(np.maximum(xy,yz),zx),h)
    
    return np.sqrt(h)

#plots a mesh in a subfigure
#splt is an element of matplotlib.pyplot
def subplot(splt,p,t):
    splt.xlabel('x1')
    splt.ylabel('x2')
    
    #plotting nodes
    splt.plot(p[:,0],p[:,1],'r.')
    
    #plotting triangles
    for i in range(len(t)):
        splt.plot([p[t[i,0],0],p[t[i,1],0]],[p[t[i,0],1],p[t[i,1],1]],'b-')
        splt.plot([p[t[i,1],0],p[t[i,2],0]],[p[t[i,1],1],p[t[i,2],1]],'b-')
        splt.plot([p[t[i,2],0],p[t[i,0],0]],[p[t[i,2],1],p[t[i,0],1]],'b-')

#read with gmsh generated .msh-files        
def read_gmsh(filename):
    #read the file
    file= open(filename,'r')
    line = file.readline()
    
    #we expect the msh format:
    # $MeshFormat
    # version-number file-typ data-size
    # $EndMeshFormat
    # $PhysicalNames
    # number-of-names
    # ...physical-dimension physical-number "physical-name"...
    # $EndPhysicalNames
    # $Nodes
    # number-of-nodes
    # ...node-number x-co y-co z-co...
    # $EndNodes
    # $Elements
    # number-of-elemnts
    # ...el-num el-type number-of-tags <tags> node-number-list...
    # $EndElements
    
    #a lot of information we dont need
    
    #we skip everything until the nodes
    #because we know that there is only one physical '101' for boundary edges
    while(line != '$Nodes\n'):
        line = file.readline()
    # line = $Nodes
    
    #Number of Nodes N
    N = int(file.readline())
    
    #initializing node array
    p = np.zeros((N,2),float)
    i=0 #index
    
    line = file.readline() #first node
    #writing nodes into p and reading the next line
    while(line != '$EndNodes\n'):
        entries = np.asarray(line.split())
        p[i,0] = entries[1] # x-co
        p[i,1] = entries[2] # y-co
        #we only have a 2-D mesh right now
        i = i+1
        line = file.readline()
        
    #line = $EndNodes
    
    #skipping assumingly useless information
    while(line !='$Elements\n'):
        line = file.readline()
        
    #line = $Elements
    
    #Number of Elements M (stil includes nodes and edges...)
    M = int(file.readline()) #M is to big
    
    c=0 #count nodes
    d=0 #count lines
    
    line = file.readline() #first element
    entries = line.split()
    
    while(len(entries) <7):
        c = c+1
        line = file.readline()
        entries = line.split()
    
    # len(entries) > 7 means no more nodes, now edges then triangles
    #initializing be
    be = np.zeros((1,2),float)
    #write first edge
    be[d,0] = float(entries[5]) -1
    be[d,1] = float(entries[6]) -1
    line = file.readline()
    entries = line.split()
    
   #writing all other edges 
    while(len(entries)<8):
        d = d+1
        be = np.append(be,[[float(entries[5]) -1,float(entries[6])-1]], axis=0)
        line = file.readline()
        entries = line.split()
    
    d=d+1
    
    #initializing t
    t = np.zeros((M-c-d,3),float)
    i=0
    #writing the elements / trianlges
    while(line != '$EndElements\n'):
        t[i,0] = float(entries[5]) -1
        t[i,1] = float(entries[6]) -1
        t[i,2] = float(entries[7]) -1
        i=i+1
        line = file.readline()
        entries = line.split()
        
    file.close()
    return [p,t,be]

# Accepts nodes p Nx2 and elements t Mx3
#and returns a spare NxN (NOT triangle) matrix
def edgeIndex(p,t):
    E = sps.lil_matrix((len(p),len(p)),dtype=np.int_)
    e = 1 #edge index counter
    
    #iterate per element
    for i in range(0,len(t)):
        #nodes of the element
        a = t[i,0]
        b = t[i,1]
        c = t[i,2]
        #checking edges exist
        #if not write edge and increase counter
        if(E[a,b] == 0):
            E[a,b] = e
            E[b,a] = e
            e = e+1
        if(E[b,c] == 0):
            E[b,c] = e
            E[c,b] = e
            e = e+1
        if(E[a,c] == 0):
            E[a,c] = e
            E[c,a] = e
            e = e+1
    E[0,0]=e-1 # number of edges
    return E