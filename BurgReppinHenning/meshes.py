# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 15:56:02 2015

@author: khenning
"""

import os
import subprocess

import numpy
import matplotlib.pyplot as plt

h = 2.0

def showMesh(verts, tris, c='g'):
    plt.xlabel('x')
    plt.ylabel('y')
    plt.triplot(verts[:,0], verts[:,1], tris, c+'-')

def show(verts1, verts2, tris1, tris2):
    plt.subplot(211)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.triplot(verts1[:,0], verts1[:,1], tris1, 'g-')
    Tv = numpy.ones([len(verts1[:,0]),1])
    plt.triplot(verts2[:,0]+Tv[0,:]*1.25, verts2[:,1], tris2, 'r-')

def showTriangle(T, c):    
    plt.plot([T[0,0],T[1,0]],[T[0,1],T[1,1]], str(c)+'-')
    plt.plot([T[1,0],T[2,0]],[T[1,1],T[2,1]], str(c)+'-')    
    plt.plot([T[2,0],T[0,0]],[T[2,1],T[0,1]], str(c)+'-') 
    
    plt.plot(T[0,0],T[0,1], str(c)+'o')
    plt.plot(T[1,0],T[1,1], str(c)+'o')
    plt.plot(T[2,0],T[2,1], str(c)+'o')

def showReferenceCell(xi0, xi1, xi2):    
    plt.plot([xi0[0,0],xi1[0,0]], [xi0[1,0],xi1[1,0]], 'k-')
    plt.plot([xi1[0,0],xi2[0,0]], [xi1[1,0],xi2[1,0]], 'k-')
    plt.plot([xi2[0,0],xi0[0,0]], [xi2[1,0],xi0[1,0]], 'k-')
    
    plt.plot(xi1[0], xi1[1], 'ro')
    plt.plot(xi0[0], xi0[1], 'ro')
    plt.plot(xi2[0], xi2[1], 'ro')
    

def showMappedCell(xi0, xi1, xi2):
    plt.plot([xi0[0],xi1[0]], [xi0[1],xi1[1]], 'k-')
    plt.plot([xi1[0],xi2[0]], [xi1[1],xi2[1]], 'k-')
    plt.plot([xi2[0],xi0[0]], [xi2[1],xi0[1]], 'k-')

    plt.plot(xi2[0], xi2[1], 'ro')
    plt.plot(xi1[0], xi1[1], 'ro')
    plt.plot(xi0[0], xi0[1], 'ro')
    

###############################################################################
###############################################################################

def generate_quad_triangulation(cellsize, boxdim=1, structedGrid=False):
    global h
    h = cellsize
    # a string holding the gmsh script commands for a rectangular domain
    # passing the box sidelength as boxdim and h as maximum cellwith
    description = "\n\
boxdim = "+str(boxdim)+";\n\
gridsize = "+str(h)+";\n\
Point(1) = {0.0,0.0,0.0,gridsize};\n\
Point(2) = {boxdim,0.0,0.0,gridsize};\n\
Point(3) = {boxdim,boxdim,0.0,gridsize};\n\
Point(4) = {0.0,boxdim,0.0,gridsize};\n\
Line(7) = {1,2};\n\
Line(8) = {2,3};\n\
Line(9) = {3,4};\n\
Line(10) = {4,1};\n\
Line Loop(14) = {7,8,9,10};\n\
Plane Surface(16) = 14;\n\
                    \n\
                    \n\
                    \n"
    if(structedGrid==True):
        description += "\n\
Transfinite Line{7,8,9,10} = boxdim/gridsize;\n\
Transfinite Surface{16};\n"
    # write the created geo script to a file
    geofile = open("grid2d.geo", "w")
    geofile.write(description)
    geofile.close()
    # call gmsh with that string
    subprocess.call(["gmsh", "-2", "grid2d.geo"], stdout = open(os.devnull, "w"))
    return
###############################################################################
###############################################################################
def generate_circle_triangulation(cellsize, boundaryCount=15, radius=1):
    r = radius
    cnt = boundaryCount
    h = cellsize
    
    pointStr = ""
    lineStr = ""
    lineLoop = "Line Loop("+str(2*cnt-1) + ")={"
    surfaceStr = "Plane Surface("+str(2*cnt)+") = "+str(2*cnt-1)+";"
    for i in range(1,cnt+1):
        phi = i/cnt*2*numpy.pi
        print(phi)
        pointStr += "Point("+str(i)+") = {" + str(r*numpy.cos(phi)) +"," + str(r*numpy.sin(phi)) + ", 0.0," + str(h) +"};\n"
        lineStr += "Line(" + str(cnt+(i-1)) + ") = {" + str(i) + "," + str(i+1 if i < cnt else 1) + "};\n"
        lineLoop += str(cnt+i-1) + ","
       
    tmp = list(lineLoop)
    tmp[len(tmp)-1] = '}'
    lineLoop = "".join(tmp) + str(";")
        
    description = pointStr + "\n" + lineStr + "\n" + lineLoop + "\n" + surfaceStr + "\n"
    print(description)
    geofile = open("grid2d.geo", "w")
    geofile.write(description)
    geofile.close()
    # call gmsh with that string
    subprocess.call(["gmsh", "-2", "grid2d.geo"])
    return
###############################################################################
###############################################################################

def read_gmsh():
    path = "grid2d.msh"
    np = 0
    dim = 2
    vertices = 0
    triangles = []
    edges = []
    points = []
    lines = []
    if(os.path.exists(path)):
        fileContent = open(path)
        # dummy init of line
        line = 'start'
        while line:
            line = fileContent.readline()
# read vertices            
###############################################################################            
            if line.find('$Nodes') == 0:
                line = fileContent.readline()
                np = int(line.split()[0])
                vertices = numpy.zeros((np, dim), dtype=float)
                for i in range(0, np):
                    line = fileContent.readline()
                    data = line.split()
                    idx = int(data[0])-1
                    if i != idx:
                        raise ValueError('problem with vertex ids')
                    vertices[i,:] = list(map(float, data[1:dim+1]))                 
# read element indices            
###############################################################################            
            if line.find('$Elements') == 0:
                line = fileContent.readline()
                ne = int(line.split()[0])
                # read in each line seperatly
                for i in range(0, ne):
                    line = fileContent.readline()
                    data = line.split()
                    idx = int(data[0])-1
                    if i != idx:
                        raise ValueError('problem with elements ids')
                    # read out wich element type we read in
                    etype = int(data[1])
                    # compute how the data field starts
                    ntags = int(data[2])
                    k = 3 + ntags
                    # etype 15, point
                    if(etype == 15):
                        points.append(list(map(int, data[k:])))
                    # etype 2, trinangle
                    elif(etype == 2):
                        tri = list(map(int, data[k:]))
                        # python indices starts by 0, gmsh starts by 1
                        # fix this here by adding -1 to every index
                        tri[0] -= 1
                        tri[1] -= 1
                        tri[2] -= 1
                        triangles.append(tri)
                         # etype 9, trinangle second order vertices, edges
                         # Triangle:               Triangle6:
                         # v                                                              
                         # ^                       
                         # |                        
                         # 2                       2
                         # |`\                     |`\
                         # |  `\                   |  `\
                         # |    `\                 5    `4
                         # |      `\               |      `\
                         # |        `\             |        `\
                         # 0----------1 --> u      0-----3----1 
                    elif(etype == 9):
                        triedg = list(map(int, data[k:]))
                        # python indices starts by 0, gmsh starts by 1
                        # fix this here by adding -1 to every index
                        triedg[0] -= 1
                        triedg[1] -= 1
                        triedg[2] -= 1
                        triangles.append([triedg[0],triedg[1],triedg[2]])
                        edges.append([triedg[3]-1,triedg[4]-1,triedg[5]-1])
                    # etype 1, line
                    elif(etype == 1):
                        line = list(map(int, data[k:]))
                        line[0] -= 1
                        line[1] -= 1
                        lines.append(line)
    return vertices, numpy.mat(triangles), numpy.mat(edges), numpy.mat(points), numpy.mat(lines)
    
###############################################################################
###############################################################################

def generate_gmsh(cellsize, boxdim=1, structedGrid=False):
    generate_quad_triangulation(cellsize, boxdim, structedGrid)
    return read_gmsh()

###############################################################################
###############################################################################

def generate_disc_gmsh(cellsize, boundaryCount=15, radius=1):
    generate_circle_triangulation(cellsize, boundaryCount, 1)
    return read_gmsh()

###############################################################################
###############################################################################
def max_mesh_width():
    return h
###############################################################################
###############################################################################
#
# interiorNodes(p, t, be)
#
# returns the interior nodes as indices into p.
#
# input:
# p - Nx2 array with coordinates of the nodes
# t - Mx3 array with indices of nodes of the triangles
# be - Bx2 array with indices of nodes on boundary edges
#
# output:
# IN - Ix1 array of nodes as indices into p that do not lie on
# the boundary
#    
def interiorNodes(V, t, be):
    Ti = t.copy()
    IN = numpy.array(Ti.flatten())
    IN = numpy.unique(IN)
    IN = numpy.delete(IN, be[:,0])
    return IN
    
def interiorNodes2(V, t, be):
    Ti = t.copy()
    IN = numpy.array(Ti.flatten())
    IN = numpy.unique(IN)
    
    beUnique = be.copy()
    beUnique = numpy.array(beUnique.flatten())
    beUnique = numpy.unique(beUnique)
    
    IN = numpy.delete(IN, beUnique)
    return IN    
###############################################################################
###############################################################################
#
# edgeIndex(t)
#
# returns the edge index matrix.
#
# input:
# t - Mx3 array with indices of nodes of the triangles
#
# output:
# E - nxn matrix of edges that connect node n1 and n2
# the boundary
#    
def edgeIndex(p,t):
    n =  len(p)
    E =  numpy. zeros([n,n])
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
    
        
        
   # for k in range(0, n):
    #    tri = t[k,:]
     #   for i in range(1,tri.size+1):
      #      index = i % 3
       #     pi1 = tri[0,index-1]
        #    pi2 = tri[0,index]
         #   E[pi1, pi2] = edges[0,index-1]
    return E
