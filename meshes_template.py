#
# import other modules
#
import meshpy.triangle as triangle
import numpy as np
import numpy.linalg as la


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
    