#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import FEM, matplotlib.pyplot as plt, solver, math, numpy as np, graded_mesh

try:
    import meshes
except ImportError as e:
    #Da meshpy bei mir nicht geht, benutze ich hier meine alte Version, bei der ich das grid "manuell" erstellt habe
    import meshes_meineVersion as meshes
    pass

#Blatt10/Aufgabe4
h = 0.1 # 1.0/3.0	# maxi
# the number of mesh nodes and elements are approximately doubled when the mesh width is divided by √2
# this meansmal mesh width
n = 3		# Integrationsordnung der Gauss Quadratur für den Elementloadvektor
steps = 6

#  H=H0 * 1/sqrt(2)**n
#  N=N0 * 2**n
# which is equally
#  log(H) - log(H0) = n * log(1/sqrt(2)
#  log(N) - log(N0) = n * log(2)
# or
#  (log(H)-log(H0))/(log(Np)-log(Np0)) = log(1/sqrt(2)/log(2) = const = -0.5
def compute(N, title):
    H = np.array([(1.0/math.sqrt(2))**(i+1) for i in range(steps)])
    Np = np.zeros(len(H))
    for nh in range(len(H)):
        graded_mesh.createbgMesh("bgmesh.pos", [-1,1], [-1,1], H[nh], N)
        (p, t, be) = meshes.grid_graded(1, H[nh], "bgmesh.pos")
        Np[nh]=len(p)

    # plot meshes
    L=meshes.make_lines(t,p)
    plt.figure()
    plt.plot(L[:,0], L[:,1])
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.figure()
    H0=H[0]; Np0=Np[0];
    k = (np.log(H[1:])-np.log(H0))/(np.log(Np[1:])-np.log(Np0));
    plt.plot(k)
    plt.xticks(range(len(H)), [str(t) for t in H])
    plt.title("k konverges to -0.5 for N=" + str(N))
    plt.ylabel("k=(log(H)-log(H0))/(log(Np)-log(Np0))")
    plt.xlabel('mesh width h')

compute(1, "unstructured mesh")
compute(25, "graded mesh")

plt.show()
