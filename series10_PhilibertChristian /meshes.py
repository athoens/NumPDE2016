#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# sudo pip install meshpy
import meshpy.gmsh_reader as reader, numpy as np, os, math, string, functools
from meshpy.gmsh_reader import GmshMeshReceiverNumPy
import matplotlib
import matplotlib.pyplot as plt
import scipy, scipy.sparse as sp

# load a .msh file
def read_gmsh(fname):
	# http://documen.tician.de/meshpy/gmsh.html#reader
	rc = GmshMeshReceiverNumPy()
	rd = reader.read_gmsh(rc, fname)

	# points
	# python2: p = np.array(list(map(lambda x: x[0:2], rc.points)))
	# see http://www.diveintopython3.net/porting-code-to-python-3-with-2to3.html#map
	p = np.array(list(map(lambda x: x[0:2], rc.points)))

	# 3 indicies of verticies
	t = np.array(list(filter(lambda x: x.shape == (3,), rc.elements)))

	# 2 boundary edge indicies
	be = np.array(list(filter(lambda x: x.shape == (2,), rc.elements)))

	return (p, t, be)

# gmsh file skeletons
# set up geometry
f = '''\
// HA3 structure
h0 = {0};
a= {1};
Point(1) = {{0, 0, 0, h0}};
Point(2) = {{a, 0, 0, h0}};
Point(3) = {{a, a, 0, h0}};
Point(4) = {{0, a, 0, h0}};
Line(1) = {{3, 4}};
Line(2) = {{4, 1}};
Line(3) = {{1, 2}};
Line(4) = {{2, 3}};
Line Loop(6) = {{1, 2, 3, 4}};
Plane Surface(6) = {{6}};
'''

# structured
fS = """\
// make it regular
Transfinite Surface {6};
// do the meshing
Mesh 2;
// save
Save "tmp.msh";
"""

# unstructured
fU = """\
// do the meshing
Mesh 2;
// save
Save "tmp.msh";
"""

fC = """\
// circle structure
h0 = {0};
r = {1};
Point(1) = {{ 0, 0, 0, h0}};
Point(2) = {{ r, 0, 0, h0}};
Point(3) = {{-r, 0, 0, h0}};
Circle(1) = {{2, 1, 3}};
Circle(2) = {{3, 1, 2}};
Line Loop(3) = {{1,2}};
Plane Surface(4) = {{3}};
"""

fG ='''
// rectangular structure
h0 = {0};
a= {1};
Point(1) = {{-a, -a, 0, h0}};
Point(2) = {{-a, 0, 0, h0}};
Point(3) = {{-a, a, 0, h0}};
Point(4) = {{a, a, 0, h0}};
Point(5) = {{a, 0, 0, h0}};
Point(6) = {{a, -a, 0, h0}};
Line(1) = {{1, 2}};
Line(2) = {{2, 3}};
Line(3) = {{3, 4}};
Line(4) = {{4, 5}};
Line(5) = {{5, 6}};
Line(6) = {{6, 1}};
// middle line
Line(7) = {{2, 5}};
// upper plane
Line Loop(8) = {{1, 7, 5, 6}};
// lower plane
Line Loop(9) = {{7, -4, -3, -2}};
Plane Surface(10) = {{8}};
Plane Surface(11) = {{9}};
'''

def write_and_execute_gmsh(str, *args):
	F = open("tmp.geo", "w")
	F.write(str)
	F.close()

	os.system("gmsh -0 -o tmp.msh tmp.geo " + " ".join([arg for arg in args]))
	return read_gmsh("tmp.msh")

# create a structured grid
def grid_square(a, h0):
	return write_and_execute_gmsh(f.format(h0, a) + fS)

# create an unstructured grid
def grid_random(a, h0):
	return write_and_execute_gmsh(f.format(h0, a) + fU)

# create a structured grid
def grid_square_circle(r, h0):
	return write_and_execute_gmsh(fC.format(h0, r) + fS)

# create a structured grid
def grid_random_circle(r, h0):
	return write_and_execute_gmsh(fC.format(h0, r) + fU)

def grid_graded(r, h0, bgmesh):
	return write_and_execute_gmsh(fG.format(h0, r), "-2 -bgm", bgmesh)

# calculate the maximum mesh distance
# TODO: translate map to python3
def max_mesh_width(tetras, points):
	perms = [[0, 1], [1, 2], [2, 0]]
	return max(max(list(map(
		lambda tetra: list(map(  # ∀ tetras
			lambda point: math.sqrt(sum((point[1] - point[0]) * (point[1] - point[0]))),
			# 2. get the euclidean-norm of them
			map(lambda perm: points[tetra][[perm]], perms)),
		# 1. get all 2-permutations of the triangles verticies (=edges)
		tetras)))))

# generate meshlines for plotting
def make_lines(tetras, points):
	perms = [[0, 1], [1, 2], [2, 0]]
	NaN = np.array([[np.nan, np.nan]])

	def unfold(L):  # L=[l,l]=[[e,e],[e,e]]->[e,e,e,e]
		return [e for l in L for e in l]

	return functools.reduce(  # 3. somehow the correct formatting was not achieved in step 2
		lambda lines1, lines2: np.append(lines1, lines2, axis=0),  # we correct that in here
		unfold(map(
			lambda tetra: list(map(lambda finperm: np.append(np.array(finperm), NaN, axis=0),
							  # 2. append each 2-permutation with NaN
							  list(map(lambda perm: list(points[tetra][[perm]]), perms)))),
			# 1. get all 2-permutations of the triangles verticies (=edges)
			tetras)))

#########
## 9/2
#########

# edgeIndex(p,t)
#
# Creates a  sparse N x N–matrix, where N is the number of nodes, that contains the indices of the edges
# that connect the nodes. For an edge with index e connecting the nodes n1 and n2, the entry in row n1 and column n2 is e.
# Since for quadratic finite elements the orientation of the edges does not matter, we
# can simply create a lower triangular matrix.
#
# input:
#  p - 3x2-matrix of the coordinates of the triangle nodes
#  t - Mx3 matrix with indices of nodes of the triangles
# output:
#  T_edges - matrix containing info about connectivity of edges
#
# TODO: the matrix gets initialised with zeros. This means the info "no edge" is coded by a zero entry.
# Therefore we need to start counting edges with 1. Maybe its better to change the "no edge info" to NA or -1?
#  a convenient solution would be the application of a "sum-type" or even a so called "Maybe" where one could express this problem within the language's type system
#  unfortunately Python does not provide such a type system, so we have no better option as sticking with this "well documented" approach
def edgeIndex(p,t):
	# Its crucial for the if statements below that the initialised value here is zero
	T_edges = scipy.sparse.tril(sp.lil_matrix((len(p),len(p))),format = "lil") # create "empty" (=zero-filled) lower triangular sparse-matrix in format lil

	#variable used to keep track of the indices of the edges, edge indices start with 1 because 0 is reserved for the case where theres no edge
	edge_index = 1
	#for each edge of each triangle check if the edge has already been added (entry in T_edges is non zero), otherwise add the entry and increment edge_index by one.
	for triangle in t:
		
		for i in range(0,3):
			# j is the next node index in the triangle (0 being the next index for 2)
			j = (i+1) % 3
			# make sure Node1 contains the bigger index -> get lower triangular matrix
			if triangle[i] > triangle[j]:
				Node1 = triangle[i]
				Node2 = triangle[j]
			else:
				Node1 = triangle[j]
				Node2 = triangle[i]
			
			if T_edges[Node1, Node2] == 0:			# make sure edges are not added (and counted!) twice
				T_edges[Node1, Node2] = edge_index
				edge_index +=1

	return T_edges
	
#(p,t,be) = grid_square(1,0.52)
#eI = edgeIndex(p,t)
#print(eI.toarray())
