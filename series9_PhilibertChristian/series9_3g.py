#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import FEM, matplotlib.pyplot as plt, solver, math, meshes

#Blatt9/Aufgabe3g
h0 = 0.1 # 1.0/3.0	# maximal mesh width
n = 3		# Integrationsordnung der Gauss Quadratur für den Elementloadvektor

# −∆u + u = f in Ω, (1a)
# grad u · n = 0 on ∂Ω Neumann b.c.

# definition of the source term f as Python function or Python's lambda function
# Setze Lösung u aus der Aufgabenstellung in die DGL ein und erhalte f, RB auch erfüllt
#  x[0,0] = x; x[0,1] = y
def f(x): return (8 * math.pi ** 2 + 1) * math.cos(2 * math.pi * x[0, 0]) * math.cos(2 * math.pi * x[0, 1])

# mesh generation for the square ]0,1[² in dependence of the maximal mesh width h0
(p, t, be)=meshes.grid_square(1, h0)

# solve -∆u + u = f on Ω system with Neumann boundary conditions, where grad u · n = 0 on ∂Ω with quadratic FEM
# u_n contains both: nodes and edges solutions
u_n = solver.solve_n0_P2(p,t,n,f)
u_n = u_n[0:len(p)] # * 114.10088913054449112705

# graphical representation with Matplotlib's plot_trisurf (for this you may use the function plot in the file FEM.py).
FEM.plot(p, t, u_n, "numerical solution for n = " + str(n) + " und h0 = " + str(h0))

# plot discretization error, evtl hätte man hier auch eine Norm für ganz Omega berechnen sollen und dann den error gegen "n" plotten
# man muss hier bissl aufpassen mit []-klammern
u = list(map(lambda x: math.cos(2 * math.pi * x[0]) * math.cos(2 * math.pi * x[1]), p))
e_n = u - u_n
FEM.plot(p, t, e_n, "discretization error for n = " + str(n) + " und h0 = " + str(h0))

plt.show()