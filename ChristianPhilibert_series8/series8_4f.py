#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import FEM, matplotlib.pyplot as plt, solver, math, meshes

#Blatt8/Aufgabe4b
h0 = 0.333  # maximal mesh width
n = 3		# Integrationsordnung der Gauss Quadratur für den Elementloadvektor

# Choose f(x, y) such that u(x) = x sin(πr) is the analytical solution of (1)
# −∆ (x sin(πr)) = -(∂x²+∂y²)(x sin(πr)) =
# Wolframalpha: (-d/dx(d/dx(x sin(pi * sqrt(x^2+y^2)))))+(-d/dy(d/dy(x sin(pi * sqrt(x^2+y^2)))))
# = π²xy² sin(πr) / r² - 4πx cos(πr) / r + πxy² cos(πr) / r³ + π²x³ sin(πr)/r² + πx³ cos(πr)/r³
# = π²xy²/r² sin(πr)  - 4πx/r cos(πr) + πxy²/r³ cos(πr)  + π²x³/r² sin(πr) + πx³/r³ cos(πr)
# = π²xy²/r² sin(πr)  + π²x³/r² sin(πr) + πx³/r³ cos(πr) - 4πx/r cos(πr) + πxy²/r³ cos(πr)
# = (π²xy²/r² + π²x³/r²) sin(πr) + (πx³/r³ - 4πx/r + πxy²/r³) cos(πr)
#  x[0,0] = x; x[0,1] = y
def f(x):
	r, phi,x, y = math.sqrt(x[0, 0]**2+x[0, 1]**2), math.atan2(x[0, 1], x[0, 0]), x[0, 0], x[0, 1]
	#return math.pi * x[0, 0] * ( math.pi*math.sin(math.pi*r) - 3*math.cos(math.pi*r)/r )
	# sed -e "s/π/math.pi/g;s/²/**2/g;s/³/**3/g" <<< '(π²*x*y²/r² + π²*x³/r²)*math.sin(π*r) + (π*x³/r³ - 4*π*x/r + π*x*y²/r³)*math.cos(π*r)'
	return (math.pi**2*x*y**2/r**2 + math.pi**2*x**3/r**2)*math.sin(math.pi*r) + (math.pi*x**3/r**3 - 4*math.pi*x/r + math.pi*x*y**2/r**3)*math.cos(math.pi*r)
# grad u · n = ∂r u
## grad u · n = g, on ∂Ω  ⇔  g = ∂r u = ∂r x sin(πr) = (∂r x) sin(πr) + πx cos(πr) = cos(φ) sin(πr) + πx cos(πr)
# grad u · n = g, on ∂Ω  ⇔  g = ∂r u = ∂r r cos(φ) sin(πr) = ∂r(r cos(φ)) sin(πr) + r cos(φ) ∂r(sin(πr)) = cos(φ) sin(πr) + πr cos(φ) cos(πr)
def g(x,y):
	r, phi = math.sqrt(x**2+y**2), math.atan2(y, x)
	#return math.cos(phi) * math.sin(math.pi * r) + math.pi * x                 * math.cos(math.pi * r)
	return math.cos(phi) * math.sin(math.pi * r) + math.pi * r * math.cos(phi) * math.cos(math.pi * r)

# mesh generation for the square ]0,1[² in dependence of the maximal mesh width h0
(p, t, be)=meshes.grid_square_circle(1, h0)

# TODO: dirty fix for missing circle implementation of read_gmsh for circles
p=p[1:len(p),:]
t=t-1
be=be-1

# solve -∆u = f on Ω system with mixed boundary conditions, where grad·u(x₁,x₂)·n = 0 on ∂Ω
u_n, lambda_param = solver.solve_mixed_g0(p,t,be,f,n,g)

# graphical representation with Matplotlib's plot_trisurf (for this you may use the function plot in the file FEM.py).
FEM.plot(p, t, u_n, "numerical solution for n = " + str(n) + "und h0 = " + str(h0))

# plot discretization error, evtl hätte man hier auch eine Norm für ganz Omega berechnen sollen und dann den error gegen "n" plotten
# man muss hier bissl aufpassen mit []-klammern
u = list(map(lambda x: x[0]*math.sin(math.pi * math.sqrt(x[0]**2+x[1]**2)), p))
e_n = u - u_n
#FEM.plot(p, t, u, "discretization error for n = " + str(n) + "und h0 = " + str(h0) + " mit λ_n = " + str(lambda_param))

plt.show()
