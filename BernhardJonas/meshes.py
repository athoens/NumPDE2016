# FUNCTIONS FOR MESH CONSTRUCTION AND MESH PROCESSING
# authors: Bernhard Aigner (359706)
#          Jonas Gienger   (370058)


# -------------------------- REQUIRED PACKAGES --------------------------------------------------
import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.sparse as sp
import subprocess


# ------------------------------ FUNCTIONS ------------------------------------------------------
def read_gmsh(filename):
  fid = open(filename,'r');	# open file to read
  string = fid.readline();
  # look for first keyword to find the beginning of the list of nodes
  string_tar = '$Nodes\n';	# target string
  while (string != string_tar):
    string = fid.readline();
  string = fid.readline();	# read number of nodes
  N = int(string.rstrip('\n'));	# convert string to integer
  # Define p: array containing x1 and x2 coordinates of all N nodes
  p = np.zeros((N, 2));
  # populate p
  for i in range(0, N):
    string = fid.readline();		# reading string
    entries = string.split();		# splitting string up in words
    for j in range(0,2):
      p[i, j] = float(entries[j+1]);	# converting string to float

  # Looking for second keyword to find beginning of the list of elements
  string_tar = '$Elements\n';	# target string
  while (string != string_tar):
    string = fid.readline();
  string = fid.readline();	# read number of elements
  elm_num = int(string.rstrip('\n'));	# convert string to integer
  # Intialize an array large enough to hold the node-lists of all points, lines and triangles.
  # It will later be split up into the arrays 'be' and 't'
  elm = np.zeros((elm_num, 3),dtype=np.int_);	# numpy array of integers
  # initialize counters for ...
  P = 0; # ...number of 1-node points
  B = 0; # ...number of 2-node lines
  M = 0; # ...number of 3-node triangles
  # scan the file to classify the elements
  for i in range(0, elm_num):
    string = fid.readline();
    entries = string.split();	# split line up in different entries
    if entries[1] == '15': # 1-node points
      P += 1;
      numtags = int(entries[2]); # gives number of tags in list
      elm[i, 0] = int(entries[3]); # write node of 1-node point down; remaining two entries of elm[i,.] are 0
    elif entries[1] == '1':  # 2-node lines
      B += 1;	# increase counter
      numtags = int(entries[2]); # gives number of tags in list
      # second tag is the "elementary geometrical entity", i.e. edge of the original square
      # => all these edges are in the boundary 
      for j in range (0, 2):
        elm[i, j] = int(entries[3 + j + numtags]);
    elif entries[1] == '2':  # 3-node triangles
      M +=1;	# count this triangle
      numtags = int(entries[2]); # gives number of tags in list
      for j in range (0, 3):
        elm[i, j] = int(entries[3 + j + numtags]);
    
  # Now that all elements (up to triangles) are classified, create the arrays of 
  # triangles and boundary edges.
  # The classes 15, 1, 2 occur in this order in the .msh file =>
  #  this order can be used to split array 'elm'
  
  # Finalizing be
  be = np.zeros((B, 2), dtype=np.int_);
  for i in range(0, B):
    be[i, :] = elm[i+P, [0,1]];
  # Finalizing t
  t = np.zeros((M, 3), dtype=np.int_);
  for i in range(0, M):
    t[i,:] = elm[i+P+B,[0,1,2]];
  
  # to have indices starting from 0, do the following:
  t -= 1;
  be -= 1;

  return p, t, be



def grid_square(a, h0):
# Produces a structured 2D, plane triangular grid of a square with
# a...sidelength of the square
# h0...maximal mesh width
# Returns three arrays:
# p...an array of dimension Nx2 containing the plane coordinates of all nodes in the mesh 
# t...an array of dimension Mx3 containing all the indices of nodes of all triangles in the mesh
# be...an array of dimension Bx2 containing all the indices of all boundary edges in the mesh
  L = m.ceil(m.sqrt(2)*a/h0);	# number of edges along each direction
  grid_spacing = a/L;	# distance betwen nodes along the x- and y- direction
  Nx = L+1;	# number of nodes along x-direction
  Ny = L+1;	# number of nodes along y-direction
  N = Nx*Ny;	# total number of nodes in the lattice
  
  p = np.zeros((N, 2));
  # loop over all nodes: write coordinates
  for ix in range(0,Nx):
    x = ix*grid_spacing; # ranges from 0 to a in steps of grid_spacing
    for iy in range(0,Ny):
      y = iy*grid_spacing;
      index = Nx*ix + iy;
      p[index, 0] = x;
      p[index, 1] = y;
  #   create boundary edges  
  B = 4*L; # total number of boundary edges
  be = np.zeros((B, 2), dtype=np.int_);
  be_counter = 0;
  
  iy = 0; # lower side from left to right
  for ix in range(0,L):
    index0 = Nx*ix + iy;
    index1 = Nx*(ix+1) + iy;	# right neighbor
    be[be_counter,0] = index0;
    be[be_counter,1] = index1;
    be_counter += 1;
  
  ix = L; # right side from top to bottom
  for iy in range(0,L):
    index0 = Nx*ix + iy;
    index1 = Nx*ix + iy+1;	# upper neighbor
    be[be_counter,0] = index0;
    be[be_counter,1] = index1;
    be_counter += 1;
  
  iy = L; # top side from right to left
  for ix in range(L,0,-1):
    index0 = Nx*ix + iy;
    index1 = Nx*(ix-1) + iy;	# left neighbor
    be[be_counter,0] = index0;
    be[be_counter,1] = index1;
    be_counter += 1;
  
  ix = 0; # left side from top to bottom
  for iy in range(L,0,-1):
    index0 = Nx*ix + iy;
    index1 = Nx*ix + iy-1;	# lower neighbor

    be[be_counter,0] = index0;
    be[be_counter,1] = index1;
    be_counter += 1;
    
  #  create triangles  
  M = 2*L*L;	# number of triangles
  t = np.zeros((M,3), dtype=np.int_);
  t_counter = 0;
  indices = [None]*3;
  for ix in range(0,Nx-1):	# all nodes, except those at the right boundary
    for iy in range(0,Ny-1):	# all nodex, except those at the upper boundary
      # triangles of "first type" /_|
      indices[0] = Nx*ix + iy;
      indices[1] = Nx*(ix+1) + iy;
      indices[2] = Nx*(ix+1) + iy+1;
      for j in range(0,3):
        t[t_counter,j] = indices[j];
      t_counter +=1;
      
      # triangles of "second type" Г/
      indices[0] = Nx*ix + iy;
      indices[1] = Nx*(ix+1) + iy+1;
      indices[2] = Nx*ix + iy+1;
      for j in range(0,3):
        t[t_counter,j] = indices[j];
      t_counter +=1;

  return p, t, be;



def grid_unstr_square(a, h0, namestring):
# Produces an unstructured grid of a square with
# a...sidelength of the square
# h0...maximal mesh width
# namestring is the string containing the file name to use (extensions '.geo' and '.msh' will be appended)
# Returns three arrays:
# p...an array of dimension Nx2 containing the plane coordinates of all nodes in the mesh 
# t...an array of dimension Mx3 containing all the indices of nodes of all triangles in the mesh
# be..an array of dimension Bx2 containing all the indices of all boundary edges in the mesh
   
   a_str = str(a);	# conversion to string
   h0_str = str(h0);	# conversion to string
   fid = open(namestring + '.geo','w');					# create a new .geo-file
   fid.write('Point(1) = {0, 0, 0, '+ h0_str +'};\n');			# write first point of square: origin
   fid.write('Point(2) = {'+ a_str +', 0, 0, '+ h0_str +'};\n');	# write second point of square...
   fid.write('Point(3) = {'+ a_str +', '+ a_str +', 0, '+ h0_str +'};\n');
   fid.write('Point(4) = {0, '+ a_str +', 0, '+ h0_str +'};\n');
   fid.write('Line(1) = {1, 2};\nLine(2) = {2, 3};\nLine(3) = {3, 4};\nLine(4) = {4, 1};\n'); # define lines
   fid.write('Line Loop(1) = {1, 2, 3, 4};\n')	# define line loop
   fid.write('Plane Surface(1) = {1};\n');	# define surface
   fid.close()					# close file
   
   subprocess.call(['gmsh', namestring + '.geo', '-2']);	# mesh .geo-file into msh.file,
   p, t, be = read_gmsh(namestring + '.msh');			# use read_gmsh to create output

   return p, t, be



def grid_unstr_polygon(a, h0, namestring):
# Produces an unstructured grid of a polygon approximation of a circle with
# a...radius of the circle
# h0...maximal mesh width
# namestring is the string containing the file name to use (extensions '.geo' and '.msh' will be appended)
# Returns three arrays:
# p...an array of dimension Nx2 containing the plane coordinates of all nodes in the mesh 
# t...an array of dimension Mx3 containing all the indices of nodes of all triangles in the mesh
# be..an array of dimension Bx2 containing all the indices of all boundary edges in the mesh
   
   a_str = str(a);	# conversion to string
   h0_str = str(h0);	# conversion to string
   theta = 2*np.arcsin(h0/(2*a));		# angle
   n = m.ceil(2*m.pi/theta);			# number of boundary nodes -1
   theta = 2*m.pi/n				# angle correction
   fid = open(namestring + '.geo','w');		# create a new .geo-file
   for i in range(0,n):				# defining the points
      fid.write('Point(' + str(i+1) + ') = {' + str(a*m.sin(i*theta)) + ', ' + str(a*m.cos(i*theta)) + ', 0, '+ h0_str +'};\n');
   for i in range(0,n-1):			# defining the lines
      fid.write('Line(' + str(i+1) + ') = {' + str(i+1) + ', ' + str(i+2) + '};\n');
   fid.write('Line(' + str(n) + ') = {' + str(n) + ', ' + str(1) + '};\n');
   fid.write('Line Loop(1) = {');
   for i in range(0,n-1):
      fid.write(str(i+1) + ' ,');		# define line loop
   fid.write(str(n) + '};\n');
   fid.write('Plane Surface(1) = {1};\n');	# define surface
   fid.close()					# close file
   
   subprocess.call(['gmsh', namestring + '.geo', '-2']);	# mesh .geo-file into msh.file,
   p, t, be = read_gmsh(namestring + '.msh');			# use read_gmsh to create output

   return p, t, be



def show(p, t, ax):
# This function plots a mesh out of input parameters:
# p...Nx2 array containing the plane coordinates of vertices for the mesh
# t...Mx3 array containing the vertices of triangles as indices of p
  ax.triplot(p[:,0], p[:,1], t, 'ro-');	# triangular plot
  plt.xlabel('x-axis');
  plt.ylabel('y-axis');
  plt.axis('equal');
  plt.xlim([min(p[:,0])-0.2, max(p[:,0])+0.2]);
  plt.ylim([min(p[:,1])-0.2, max(p[:,1])+0.2]);
  #plt.title('triangular plot');
  #plt.show()					# show plot

  return



def show_mkbnd(p, t, be, ax):
# This function plots a mesh out of input parameters (also shows the boundary in different color):
# p...Nx2 array containing the plane coordinates of vertices for the mesh
# t...Mx3 array containing the vertices of triangles as indices of p
# be..Bx2 array containing the nodes of the boundary edges
  ax.triplot(p[:,0], p[:,1], t, 'r.-');	# triangular plot
  B = np.shape(be)[0];	# number of boundary edges
  for i in range(0,B):	# plot boundary
    plt.plot(p[be[i,:]][:,0],p[be[i,:]][:,1],'b-', linewidth=2.0);
  plt.xlabel('x-axis');
  plt.ylabel('y-axis');
  plt.axis('equal');
  plt.xlim([min(p[:,0])-0.2, max(p[:,0])+0.2]);
  plt.ylim([min(p[:,1])-0.2, max(p[:,1])+0.2]);
  #plt.title('triangular plot');
  #plt.show()					# show plot

  return



def max_mesh_width(p,t):
# find the actual maximum mesh witdh of points p and triangles t
  #N = np.shape(p)[0]; # number of points
  M = np.shape(t)[0]; # number of triangles
  # loop over all triangles to find the longest edge:
  lengths=np.zeros((M,3));
  for i in range(0,M):
    # three vectors describing the edges:
    v1 = p[t[i,1]]-p[t[i,0]];
    v2 = p[t[i,2]]-p[t[i,1]];
    v3 = p[t[i,0]]-p[t[i,2]];
    # norms of these vector = lengths of the sides
    lengths[i,0] = np.linalg.norm(v1)	# length of side 1
    lengths[i,1] = np.linalg.norm(v2)	# length of side 2
    lengths[i,2] = np.linalg.norm(v3)	# length of side 3
  # maximum mesh width = maximum of all side lengths of all triangles
  max_width = np.amax(lengths);
  
  return max_width



def edgeIndex(p,t):
# gives indices to all edges a triangular mesh
# input:
# p - Nx2-matrix of the coordinates of the triangle nodes
# t - Mx3-matrix of indices of nodes of the triangle
# 
# output:
# e - NxN-matrix in scipy's lil format, adjacency matrix of the undirected mesh, where non-zero entries
#     are the index of the edges between node. Since the matrix is symmetric with no diagonal entries,
#     it is stored as an upper triangular matrix. 

  N = np.shape(p)[0]; 		# number of points
  M = np.shape(t)[0]; 		# number of triangles
  
  e = sp.lil_matrix((N, N), dtype=np.int_);	# sparse matrix to hold the edge indices
  E = 0;			# initialize counter for number of edges
  
  # loop over all triangles
  for i in range(0,M):
    # get the three node indices
    n = [None]*3;		# empty 3-array
    for j in range(0,3):
      n[j] = t[i,j];
    
    # loop over pairs of nodes in the triangle
    for j in range(0,3):
      n1 = n[j];		# beginning node of the edge
      n2 = n[(j+1)%3];  	# next node index (modulo 3) = end node of the edge
      
      if n2<n1:    		# sort node indices to have n1<n2 --> upper triangular matrix
        aux = n2;
        n2 = n1;
        n1 = aux    
      # is this edge (n1, n2) new or did it occur in a previous triangle and is already numbered?
      if e[n1,n2] == 0:
        E+=1; 			# increase edge counter
        e[n1,n2] = E; 		# label new edge (starting with 1)    
    
  return e


# ------------------- USEFULL INFROMATION FROM THE GMSH DOCUMENTATION -----------------------------
# File format explained
# $MeshFormat
# version-number file-type data-size
# $EndMeshFormat
# $PhysicalNames
# number-of-names
# physical-dimension physical-number "physical-name"
# …
# $EndPhysicalNames
# $Nodes
# number-of-nodes
# node-number x-coord y-coord z-coord
# …
# $EndNodes
# $Elements
# number-of-elements
# elm-number elm-type number-of-tags < tag > … node-number-list
# …
# $EndElements
# $Periodic
# number-of-periodic-entities
# dimension slave-entity-tag master-entity-tag
# number-of-nodes
# slave-node-number master-node-number
# …
# $EndPeriodic
# $NodeData
# number-of-string-tags
# < "string-tag" >
# …
# number-of-real-tags
# < real-tag >
# …
# number-of-integer-tags
# < integer-tag >
# …
# node-number value …
# …
# $EndNodeData
# $ElementData
# number-of-string-tags
# < "string-tag" >
# …
# number-of-real-tags
# < real-tag >
# …
# number-of-integer-tags
# < integer-tag >
# …
# elm-number value …
# …
# $EndElementData
# $ElementNodeData
# number-of-string-tags
# < "string-tag" >
# …
# number-of-real-tags
# < real-tag >
# …
# number-of-integer-tags
# < integer-tag >
# …
# elm-number number-of-nodes-per-element value …
# …
# $EndElementNodeData
# $InterpolationScheme
# "name"
# number-of-element-topologies
# elm-topology
# number-of-interpolation-matrices
# num-rows num-columns value …
# …
# $EndInterpolationScheme
    

# Also from the gmsh documentation
# number-of-tags
    #gives the number of integer tags that follow for the n-th element.
    #By default, the first tag is the number of the physical entity to which the element belongs;
    #the second is the number of the elementary geometrical entity to which the element belongs;
    #the third is the number of mesh partitions to which the element belongs, followed by the partition ids (negative partition ids indicate ghost cells).
    #A zero tag is equivalent to no tag. Gmsh and most codes using the MSH 2 format require at least the first two tags (physical and elementary tags).
  Contains
