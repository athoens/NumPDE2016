#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import FEM, matplotlib.pyplot as plt, solver, math, meshes

#Blatt8/Aufgabe4b
h0 = 0.1	# maximal mesh width
n = 3		# Integrationsordnung der Gauss Quadratur für den Elementloadvektor

# Choose f (x, y) such that u(x, y) = cos(πx) cos(πy) is the analytical solution of −∆u = f
# −∆ (cos(πx) cos(πy)) = -(∂x²+∂y²)(cos(πx) cos(πy)) = 2π² cos(πx) cos(πy)
#  x[0,0] = x; x[0,1] = y
def f(x): return (2 * math.pi ** 2) * math.cos(math.pi * x[0, 0]) * math.cos(math.pi * x[0, 1])







# mesh generation for the square ]0,1[² in dependence of the maximal mesh width h0
(p, t, be)=meshes.grid_random(1, h0)

# solve -∆u = f on Ω system with mixed boundary conditions, where grad·u(x₁,x₂)·n = 0 on ∂Ω
u_n, lambda_param = solver.solve_mixed_g0(p,t,be,f,n,lambda x1,x2: 0)

# graphical representation with Matplotlib's plot_trisurf (for this you may use the function plot in the file FEM.py).
FEM.plot(p, t, u_n, "numerical solution for n = " + str(n) + "und h0 = " + str(h0))

# plot discretization error, evtl hätte man hier auch eine Norm für ganz Omega berechnen sollen und dann den error gegen "n" plotten
# man muss hier bissl aufpassen mit []-klammern
u = list(map(lambda x: math.cos(math.pi * x[0]) * math.cos(math.pi * x[1]), p))
e_n = u - u_n
FEM.plot(p, t, e_n, "discretization error for n = " + str(n) + "und h0 = " + str(h0) + " mit λ_n = " + str(lambda_param))

plt.show()
