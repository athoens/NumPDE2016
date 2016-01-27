# coding=utf-8
###-----------------------------------------------------------###
###  Name: Héctor Andrade Loarca                              ###
###  Course: Numerics of PDEs                                 ###
###  Professor: Kersten Schmidt                               ###
###                                                           ###
###               Module meshes.py                            ###
###            " Dealing with meshes "                        ###
###                                                           ###
###-----------------------------------------------------------###

import numpy as np 
import matplotlib.pyplot as plt
import scipy.sparse as sparse

#First lets define the function read_gmsh that parse a .msh file
#as input it recieve the file name

#file='square_mesh.msh'

def read_gmsh(file):
	#We read the file
	msh=open(file).read()
	#We split the big string that we get from the file by line skips \n
	msh=msh.split('\n')
	##
	#We identify the entries with Nodes by begin and end
	begin=msh.index('$Nodes')+2
	end=msh.index('$EndNodes')
	nodes=msh[begin:end]
	# We split the elements of notes by the space to get the entries
	nodes=map(lambda x:x.split(' ')[1:3],nodes)
	# Convert the entries to floats instead of strings
	p=np.array(nodes).astype(float)
	##
	#Now we identify the entries of the elements
	begin=msh.index('$Elements')+2
	end=msh.index('$EndElements')
	elements=msh[begin:end]
	#Split each element by the space between the entries
	elements=map(lambda x: x.split(' '),elements)
	##
	#We filter elements of dimension 1 and elements
	elemone=[element for element in elements if element[1]=='1']
	#We get the last two entries that represents the nodes of this one 
	#dimensional element
	elemone=[x[5:8] for x in elemone]
	#We transform to a np.array with float entries
	be=np.array(elemone).astype(int)
	#
	#We filter elements of dimension 2 and elements
	elemtwo=[element for element in elements if element[1]=='2']
	#We get the last three entries that represents the nodes of this two
	#dimensional element that is a triangle in counterclockwise order
	elemtwo=[x[5:9] for x in elemtwo]
	#We transform to a np.array with float entries
	t=np.array(elemtwo).astype(int)
	#Finally we return a list of this 3 arrays
	return [p,t,be]

#Now lets define a fucntion grid_square that produces a uniform grid of
# a square of side length a and maximal mesh widht h0, with the same outputs
# as read_gmsh

# If we want h0 to be the maximal mesh width in the square of lenght side a
# we need it to be an hypotenouse of a right triangle with equal sides that 
# are divisors of the length a, to fit perfectly 

# Thats it n*(h0)/sqrt(2)=a for some n

# Lets define the fuction grid_square
#a=1
#h0=np.sqrt(2)/10

def grid_square(a,h0):
	#Check if h0 and a fullfill the requirements for regular meshes
	if int(np.sqrt(2)*a/h0)!=np.sqrt(2)*a/h0:
		return "h0 and a do not generate a regular mesh"
	else:
		#Define the number of elements in each side
		n=int(np.sqrt(2)*a/h0)
		#We gonna order the nodes from below to the top beginnign ant the 0,0
		p=[[i*a/float(n),0.0] for i in range(0,n+1)]
		for j in range(1,n+1):
			p=p+[[i*a/float(n),j*a/float(n)] for i in range(0,n+1)]
		#We convert p to np.array
		p=np.array(p)
		##
		# Now we generate the arrays with the one dimensional elements in the border
		# We will take the nodes numerate from bellow to the top of the square grid
		# And we will take the elements in the border in counterclockwise order
		# Divided in the 4 borders
		#First border
		first=[[i,i+1] for i in range(1,n+1)]
		second= [[(n+1)*i,(n+1)*i+n+1] for i in range(1,n+1)]
		third=[[(n+1)**2-i+1,(n+1)**2-i] for i in range(1,n+1)]
		fourth=[[(n+1)**2-n-(i-1)*(n+1),(n+1)**2-n-i*(n+1)] for i in range(1,n+1)]
		#We join the borders and convert to np-array
		be=np.array(first+second+third+fourth)
		##
		# Now we gonna create the arrays with the two dimensional triangles elements 
		# nodes
		# We gonna create them in layers from bellow to top
		# We create the two type of triangles depending on the orientation of the triangle
		# First the triangles pointing down-left
		j=1
		t1=[[i+(j-1)*(n+1),i+1+(j-1)*(n+1),(i+1)+(n+1)+(j-1)*(n+1)] for i in range(1,n+1)]
		for j in range(2,n+1):
			t1=t1+[[i+(j-1)*(n+1),i+1+(j-1)*(n+1),(i+1)+(n+1)+(j-1)*(n+1)] for i in range(1,n+1)]
		# Now the triangles pointing up-right
		j=1
		t2=[[i+(j-1)*(n+1),i+1+(n+1)+(j-1)*(n+1),i+(n+1)+(j-1)*(n+1)] for i in range(1,n+1)]
		for j in range(2,n+1):
			t2=t2+[[i+(j-1)*(n+1),i+1+(n+1)+(j-1)*(n+1),i+(n+1)+(j-1)*(n+1)] for i in range(1,n+1)]
		#And then we join q1 and q2 and convert it to np.array
		t=np.array(t1+t2)
		# Now we got our three outputs
		return [p,t,be]

#Now lets define a function that draws the mesh with Matplotlib which accepts
# a numpy array p with the nodes and an array t of the triangles with the index 
# of the nodes and the filename that you want to save the image and the title

def show(p,t,file_name,title):
	# Create the loop over each triangle
	for i in range(0,len(t)):
		ti=t[i]
		# Take the points of each triangle
		pi=[p[ti[i]-1] for i in range(0,3)]
		#Define the x and y coordinates of each point
		xi=[xi[0] for xi in pi]
		yi=[yi[1] for yi in pi]
		#Close the loop in the triangle
		xi=xi+[xi[0]]
		yi=yi+[yi[0]]
		#plot each trianle
		plt.plot(xi,yi,"-k")
	plt.title(title)
	plt.xlabel('x1')
	plt.ylabel('y2')
	plt.savefig(file_name)
	plt.close()

# Now lets define a maximal mesh function that returns the maximal width
# Of a mesh with p nodes and t triangles

def max_mesh_width(p,t):
	# We save in a array the maximum distance between points in each triangle
	h=[]
	for i in range(0,len(t)):
		ti=t[i]
		# Take the points of each triangle
		pi=[p[ti[i]-1] for i in range(0,3)]
		maxi=max([np.linalg.norm(pi[0]-pi[1]),np.linalg.norm(pi[0]-pi[2]),np.linalg.norm(pi[1]-pi[2])])
		h.append(maxi)
	#Finally we get the maximum width of the mesh
	return max(h)

# Function that takes a list of edges with duplicates and has as output a list with
# unique values and with the original order

def unique_edges(edges):
	# First we check the duplicates values and put them to zero
	for edge in edges:
		#exactduplicates
		duplicates1=np.array([edge1==edge for edge1 in edges])
		#the edges that are the same just with different order
		duplicates2=np.array([edge1[0]==edge[1] and edge1[1]==edge[0] for edge1 in edges])
		duplicates=duplicates1+duplicates2
		# If there ar duplicates we put them to zero
		if sum(duplicates)!=1:
			#Get the indinces of the duplicates
			indices=np.array([i for i in range(len(duplicates))])
			indices=indices[duplicates]
			# We set to zero the duplicates but the first element
			for i in range(1,len(indices)):
				edges[indices[i]]=[0,0]
	return [edge for edge in edges if edge!=[0,0]]





# Lets define a function that has as input the array p of the N nodes coordinates 
# as output a NxN array with entry (n_1,n_2) the index of the edge that contains this two
# edges


def edgeIndexold(p,t):
	# First we initialize an array of length N
	N=len(p)
	EdgeIndex=np.zeros((N,N))
	# First we gonna do an array with the edges and the numbering of their
	# nodes
	edges=[]
	for ti in t:
		edges=edges+[[ti[0],ti[1]],[ti[1],ti[2]],[ti[2],ti[0]]]
	#We extract just the unique value in a array N
	edges=np.array(unique_edges(edges))
	# Clearly the output will be a symmetric matrix
	for i in range(len(edges)):
		EdgeIndex[edges[i][0]-1,edges[i][1]-1]=i+1
		EdgeIndex[edges[i][1]-1,edges[i][0]-1]=i+1
	#We return the EdgeIndex matrix and the array with the edges in their position
	return [edges,EdgeIndex]


def edgeIndex(p,t): 
    n =  len(p)
    #E  = lil_matrix((p.shape[0],p.shape[0]), dtype=np.int);
    
    E =  np.zeros((n,n));
    print E.shape
    m = 1
    for j in range(len(t)):
        if E[t[j,0],t[j,1]]==0 and E[t[j,1],t[j,0]]==0:
            E[t[j,0],t[j,1]]=m
            E[t[j,1],t[j,0]]=m
            m=m+1
        
        if E[t[j,1],t[j,2]]==0 and E[t[j,2],t[j,1]]==0:
            E[t[j,1],t[j,2]]=m
            E[t[j,2],t[j,1]]=m
            m=m+1
       
        if E[t[j,0],t[j,2]]==0 and E[t[j,2],t[j,0]]==0:
            E[t[j,0],t[j,2]]=m
            E[t[j,2],t[j,0]]=m
            m=m+1
    
    return E   



