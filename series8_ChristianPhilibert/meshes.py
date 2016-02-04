#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# sudo pip install meshpy
import meshpy.gmsh_reader as reader, numpy as np, os, math
from meshpy.gmsh_reader import GmshMeshReceiverNumPy
import matplotlib
import matplotlib.pyplot as plt

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
f = '''
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

def write_and_execute_gmsh(str):
	F = open("tmp.geo", "w")
	F.write(str)
	F.close()

	os.system("gmsh -0 -o tmp.msh tmp.geo")
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

# calculate the maximum mesh distance
# TODO: translate map to python3
def max_mesh_width(tetras, points):
	perms = [[0, 1], [1, 2], [2, 0]]
	return max(max(map(
		lambda tetra: map(  # ∀ tetras
			lambda point: math.sqrt(sum((point[1] - point[0]) * (point[1] - point[0]))),
			# 2. get the euclidean-norm of them
			map(lambda perm: points[tetra][[perm]], perms)),
		# 1. get all 2-permutations of the triangles verticies (=edges)
		tetras)))

# generate meshlines for plotting
# TODO: translate map to python3
def make_lines(tetras, points):
	perms = [[0, 1], [1, 2], [2, 0]]
	NaN = np.array([[np.nan, np.nan]])

	def unfold(L):  # L=[l,l]=[[e,e],[e,e]]->[e,e,e,e]
		return [e for l in L for e in l]

	return reduce(  # 3. somehow the correct formatting was not achieved in step 2
		lambda lines1, lines2: np.append(lines1, lines2, axis=0),  # we correct that in here
		unfold(map(
			lambda tetra: map(lambda finperm: np.append(np.array(finperm), NaN, axis=0),
							  # 2. append each 2-permutation with NaN
							  map(lambda perm: list(points[tetra][[perm]]), perms)),
			# 1. get all 2-permutations of the triangles verticies (=edges)
			tetras)))

#(p, t, be)=grid_random(1,0.1)
