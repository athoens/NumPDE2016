#
# import other modules
#
import meshpy.triangle as triangle
import numpy as np
import numpy.linalg as la
from scipy.sparse import lil_matrix
import meshpy.gmsh_reader as reader
import os

# load a .msh file
def read_gmsh(fname):
    # create a receiver from the GmshMeshReceiverBase interface
    rc = reader.GmshMeshReceiverNumPy()
    # read the mesh from fname file
    rd = reader.read_gmsh(rc, fname)

    # points
    p = np.array(list(map(lambda x: x[0:2], rc.points)))

    # 3 indicies of verticies
    t = np.array(list(filter(lambda x: x.shape == (3,), rc.elements)))

    # 2 boundary edge indicies
    be = np.array(list(filter(lambda x: x.shape == (2,), rc.elements)))
    
    # Dirichlet boundary edge indicies
    # find indicies for atributes
    I0  = np.nonzero(np.array(list(rc.element_markers)) == 20);
    bd0 = np.array(list(rc.elements))[I0[0]]
    bd  = np.zeros((len(bd0),2));
    for j in range(len(bd)):
      bd[j][0] = bd0[j][0];
      bd[j][1] = bd0[j][1];
    #print list(rc.elements)
    
    return (p, t, be, bd)


def create_msh_bgm(fname):
    l = "gmsh -2 square.geo -bgm " + fname + ".pos -o " + fname + ".msh"
    print l
    os.system(l)
    return read_gmsh(fname+".msh")

def create_msh(fname, h):
    l = "gmsh -2 square.geo -clmax " + str(h) +" -o " + fname + ".msh"
    print l
    os.system(l)
    return read_gmsh(fname+".msh")



#
# round_trip_connect(a, b)
#
# Auxiliary function that returns a set of pairs of adjacent indices between
# the integers a and b. This set also includes the pair (b, a) as last entry. 
#
# input:
# a - integer
# b - integer (b>a)
#
# output:
# set of pairs of adjacent indices between a and b and the pair (b, a)
#
def round_trip_connect(a, b):
  return [(i, i+1) for i in range(a, b)] + [(b, a)]


#
# max_edge_length(vertices)
# 
# Auxiliary function that returns the maximal distance of the three vertices, 
# or equivalenty, the maximal length of all edges in the triangle spanned by 
# the three vertices.
#
# input:
# vertices - array of three 2d vertices
#
# output:
# maximal distance of the three vertices
#
def max_edge_length(vertices):
  p = np.array(vertices)
  return max(la.norm(p[0]-p[1]),la.norm(p[1]-p[2]),la.norm(p[2]-p[0]))


#
# square(a,h0)
#
# Function that produces an unstructured mesh of a square. The square has side
# length a and its lower left corner is positioned at the origin. The maximal
# mesh width of the produced mesh is smaller or equal to h0. (The maximal mesh
# width of a mesh is the maximal distance of two adjacent vertices, or 
# equivalenty, the maximal length of all edges in the mesh.)
#
# input: 
# a  - side length of square
# h0 - upper bound for maximal mesh width
#
# output:
# p - Numpy-array of size Nx2 with the x- and y-coordinates of the N vertices 
#     in the mesh
# t - Numpy-array of size Mx3, where each row contains the indices of the three
#     vertices of one of the M triangles in counterclockwise order.
#
def square(a,h0):
  # define the four vertices of the square
  points = [(0, 0), (a, 0), (a, a), (0, a)]
  # define the four edges between the four vertices of the square
  facets = round_trip_connect(0, len(points)-1)
  # initialize the mesh and set the vertices and the edges of the domain
  info = triangle.MeshInfo()
  info.set_points(points)
  info.set_facets(facets)
  # define a function that returns a boolean that is true if the triangle with
  # vertices vtx and area a has maximal edge length larger than the desired 
  # maximal mesh width h0.
  def needs_refinement(vtx, a):
    return bool(max_edge_length(vtx) > h0)
  # create the mesh giving the mesh information info and the refinement 
  # function needs_refinement as input
  mesh = triangle.build(info, refinement_func=needs_refinement)
  # read vertices and triangles of the mesh and convert the arrays to numpy 
  # arrays
  p = np.array(mesh.points)
  t = np.array(mesh.elements)
  # return the vertices and triangles of the mesh
  return (p, t)

    
#
# grid_square(a,h0)
#
# Function that produces an structured grid of a square. The square has side
# length a and its lower left corner is positioned at the origin. The maximal
# mesh width of the produced mesh is smaller or equal to h0. (The maximal mesh
# width of a mesh is the maximal distance of two adjacent vertices, or 
# equivalenty, the maximal length of all edges in the mesh.)
#
# input: 
# a  - side length of square
# h0 - upper bound for maximal mesh width
#
# output:
# p - Numpy-array of size Nx2 with the x- and y-coordinates of the N vertices 
#     in the mesh
# t - Numpy-array of size Mx3, where each row contains the indices of the three
#     vertices of one of the M triangles in counterclockwise order.
#  
def grid_square(a,h0):
  n=int(np.ceil(np.sqrt(2.0)*a/h0))
  p=np.zeros(((n+1)*(n+1),2))
  t=np.zeros((2*n*n,3),dtype=int)
  for i in range(0,n+1):
    for j in range(0,n+1):
      p[i*(n+1)+j,0] = i*a/float(n)
      p[i*(n+1)+j,1] = j*a/float(n)
  for i in range(0,n):
    for j in range(0,n):
      t[i*n+j,0]=i*(n+1)+j
      t[i*n+j,1]=i*(n+1)+j+1
      t[i*n+j,2]=(i+1)*(n+1)+j+1
      t[i*n+j+n*n,0]=i*(n+1)+j
      t[i*n+j+n*n,1]=(i+1)*(n+1)+j+1
      t[i*n+j+n*n,2]=(i+1)*(n+1)+j
  return (p, t)


#
# show(p,t)
# 
# Function that plots the mesh with points p and triangles t.
# 
# input:
# p - Numpy-array of size Nx2 with the x- and y-coordinates of the N vertices 
#     in the mesh
# t - Numpy-array of size Mx3, where each row contains the indices of the three
#     vertices of one of the M triangles in counterclockwise order.
#
def show(p,t):
  import matplotlib.pyplot as pt
  pt.triplot(p[:, 0], p[:, 1], t)
  pt.show()


#
# max_mesh_width(p,t)
#
# Function that returns the maximal mesh width of the mesh with points p and 
# triangles t.
#
# input:
# p - Numpy-array of size Nx2 with the x- and y-coordinates of the N vertices 
#     in the mesh
# t - Numpy-array of size Mx3, where each row contains the indices of the three
#     vertices of one of the M triangles in counterclockwise order.
#
# output:
# h0 - Maximal mesh width
#
def max_mesh_width(p,t):
  h0=0.0;
  for i in range(0,t.shape[0]):
    h0=max(h0,max_edge_length(p[t[i,:],:]))
  return h0


#
# circle(a,h0,n)
#
# Function that produces an unstructured mesh of a circle. The circle has 
# radius r and its center is located at the origin. The maximal mesh width of
# the produced mesh is smaller or equal to h0. The number of points on the 
# boundary is at least n. If the maximal mesh width is smaller than the 
# distance between the n points on the boundary, the number of points on the 
# boundary is increased accordingly.
#
# input: 
# r  - radius of circle
# h0 - upper bound for maximal mesh width
# n  - minimal number of points on the boundary
#
# output:
# p - Numpy-array of size Nx2 with the x- and y-coordinates of the N vertices 
#     in the mesh
# t - Numpy-array of size Mx3, where each row contains the indices of the three
#     vertices of one of the M triangles in counterclockwise order.
#
def circle(r,h0,n):
  n = max(n,int(np.ceil(2.0*np.pi*r/h0)))
  points = [(r*np.cos(2*np.pi*i/float(n)), r*np.sin(2*np.pi*i/float(n))) for i in range(0, n)]
  facets = round_trip_connect(0, len(points)-1)
  info = triangle.MeshInfo()
  info.set_points(points)
  info.set_facets(facets)
  def needs_refinement(vertices, area):
    return bool(max_edge_length(vertices) > h0)
  mesh = triangle.build(info, refinement_func=needs_refinement)
  return (np.array(mesh.points), np.array(mesh.elements))


import scipy.sparse as sparse

#
# collects indices of edges, builds a mapping from node indices to edge
# indices and identifies the boundary nodes and edges within a mesh

# input:
#   p - Nx2-array of nodal coordinates
#   t - Mx3-array of triangles as indices into p, defined with a 
#       counter-clockwise node ordering

# output:
#   eIndex      - NxN-sparse matrix of all node combinations 
#
def edgeIndex(p,t): 
    n =  len(p)
    E  = lil_matrix((p.shape[0],p.shape[0]), dtype=np.int);
    #E =  np.zeros([n,n])
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


