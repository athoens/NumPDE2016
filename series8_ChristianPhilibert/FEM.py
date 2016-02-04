#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np, scipy.sparse as sp, matplotlib.pyplot as plt

# gaussTriangle(n)
# returns abscissas and weights for "Gauss integration" in the triangle with vertices (-1,-1), (1,-1), (-1,1)
# input:
#   n - order of the numerical integration (1 <= n <= 5)
#
# output:
#   x - 1xp-array of abscissas, that are 1x2-arrays (p denotes the number of 
#       abscissas/weights)
#   w - 1xp-array of weights (p denotes the number of abscissas/weights)
def gaussTriangle(n):
	if n == 1:
		x = [[-1 / 3., -1 / 3.]];
		w = [2.];
	elif n == 2:
		x = [[-2 / 3., -2 / 3.],
			 [-2 / 3., 1 / 3.],
			 [1 / 3., -2 / 3.]];
		w = [2 / 3.,
			 2 / 3.,
			 2 / 3.];
	elif n == 3:
		x = [[-1 / 3., -1 / 3.],
			 [-0.6, -0.6],
			 [-0.6, 0.2],
			 [0.2, -0.6]];
		w = [-1.125,
			 1.041666666666667,
			 1.041666666666667,
			 1.041666666666667];
	elif n == 4:
		x = [[-0.108103018168070, -0.108103018168070],
			 [-0.108103018168070, -0.783793963663860],
			 [-0.783793963663860, -0.108103018168070],
			 [-0.816847572980458, -0.816847572980458],
			 [-0.816847572980458, 0.633695145960918],
			 [0.633695145960918, -0.816847572980458]];
		w = [0.446763179356022,
			 0.446763179356022,
			 0.446763179356022,
			 0.219903487310644,
			 0.219903487310644,
			 0.219903487310644];
	elif n == 5:
		x = [[-0.333333333333333, -0.333333333333333],
			 [-0.059715871789770, -0.059715871789770],
			 [-0.059715871789770, -0.880568256420460],
			 [-0.880568256420460, -0.059715871789770],
			 [-0.797426985353088, -0.797426985353088],
			 [-0.797426985353088, 0.594853970706174],
			 [0.594853970706174, -0.797426985353088]];
		w = [0.450000000000000,
			 0.264788305577012,
			 0.264788305577012,
			 0.264788305577012,
			 0.251878361089654,
			 0.251878361089654,
			 0.251878361089654];
	else:
		print('numerical integration of order ' + str(n) + 'not available');

	return x, w


# plot(p,t,u)
#   plots the linear FE function u on the triangulation t with nodes p
# input:
# p  - Nx2 matrix with coordinates of the nodes
# t  - Mx3 matrix with indices of nodes of the triangles
# u  - Nx1 coefficient vector
def plot(p, t, u, title):
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_trisurf(p[:, 0], p[:, 1], t, u, cmap=plt.cm.jet)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('u')
	plt.title(title)
	#plt.show()

# elemStiffness(p)
#   computes the element stiffness matrix related to the bilinear form a_K(u,v) = ∫(K, grad u . grad v dx) for linear FEM on triangles.
# input:
#   p - 3x2-matrix of the coordinates of the triangle nodes
# output:
#   AK - element stiffness matrix
def elemStiffness(points):
	x = points[:, 0]
	y = points[:, 1]
	Dk = np.array([
		y[[1, 2, 0]] - y[[2, 0, 1]],
		x[[2, 0, 1]] - x[[1, 2, 0]]
	])
	detFK = np.linalg.det(np.array([
		points[1, :] - points[0, :],
		points[2, :] - points[0, :]
	]).T)
	Ak = 1.0 / (4.0 * 0.5 * np.abs(detFK)) * Dk.T.dot(Dk)  # abs() better safe then sorry
	return Ak

# elemMass(p)
#   computes the element mass matrix related to the bilinear form m_K(u,v) = ∫(K u v dx) for linear FEM on triangles.
# input:
#   p - 3x2-matrix of the coordinates of the triangle nodes
# output:
#   MK - element mass matrix
def elemMass(points):
	detFK = np.linalg.det(np.array([
		points[1, :] - points[0, :],
		points[2, :] - points[0, :]
	]).T)
	mK = detFK * np.ones([3, 3]) * 1.0 / 24.0
	mK[range(3), range(3)] = detFK * 1.0 / 12.0
	return mK

# transformiert (f·v dx)(x) für x ∈ K nach (g3 dξ)(ξ) für ξ ∈ K̂ für Dreiecke in 2D, Φ: K̂ → K, ξ ↦ Fₖ·ξ + τ and ∫(Ω,α)=∫(Φ⁻¹(Ω),Φ(α))
#   mit g3 = f(Φₖ(ξ))·N̂ₗ(ξ)·|det Fₖ|
# input:
#   f  - Lastfunktion/rechte Seite
#   p  - 3x2 array mit Koordinaten der Dreiecksknoten
#   xi - 1x2? array mit den ξ-Werten (sollten bereits transformiert sein)
#   l  - index für N_hut_l, also die reference element shape function
#
# with Φ : K̂ → K,  and 
#   ∫(    K,    f(  x)    ·   v                 dx )
# = ∫(Φ⁻¹(K), Φ(f(  x))   ·   v                 dx))
# = ∫(Φ⁻¹(K), Φ(f(  x))   · Φ(v)              Φ(dx))
# = ∫(    K̂ ,   f(Φ(ξ))   ·   v(Φ(ξ))         Φ(dx))
# = ∫(    K̂ ,   f(Jᵩ·ξ+τ) ·     N̂(ξ)   |det(Jᵩ)|dξ )
#  where Jᵩ = Fₖ
def g3(f, p, xi, l):
	F_k = np.array([p[1] - p[0], p[2] - p[0]]).T # Fₖ = [p₁-p₀,p₂-p₀]ᵀ = Jᵩ = dΦⁱ/dξ
	tau_K = np.array([p[0]])                     # τₖ = p₀
	detFk = abs(np.linalg.det(F_k))

	# Φₖ(ξ) = Fₖ·ξ + τₖ
	phi_k = lambda xi: np.dot(F_k, xi) + tau_K;

	# reference element shape functions N̂ₗ(ξ) with ξ ∈ triange{(-1,1),(-1,-1),(1,-1)} 2D
	N_dach = [
		lambda xi: 1 - xi[0] - xi[1],
		lambda xi: xi[0],
		lambda xi: xi[1]
	];

	#          f(Φₖ( ξ)) ·        N̂ₗ( ξ) ·|det Fₖ|
	return (f(phi_k(xi)) * N_dach[l](xi) * detFk)

# transformiert (g · v dx)(x) für x ∈ Γ ⊂ Ω nach (g2 dt)(t) für t ∈ I für Kanten in 1D, Φ: I → Γ, t ↦ Fₖ·[t, t]ᵀ + τ
# t ↦ ((p₁.x-p₀.x)·t + p₀.x, (p₁.y-p₀.y)+p₀.y)
#   mit g2 = g(Jᵩ·[t,t]ᵀ+τ)·N̂(t)·|det(Jᵩ)|
# input:
#   f  - Lastfunktion/rechte Seite
#   p  - 2x2 array mit Koordinaten der Kantenknoten
#   xi - Skalar (ξ-Wert, sollte bereits transformiert sein)
#   l  - index für N_hut_l, also die reference element shape function
# es sei γ ⊂ K̂
#   ∫(    γ ,   g              ·   v                 dx )
# = ∫(Φ⁻¹(γ), Φ(g              ·   v                 dx))
# = ∫(Φ⁻¹(γ), Φ(g)             · Φ(v)              Φ(dx))
# = ∫(Φ⁻¹(γ),   g(Φ(x))        ·   v(Φ(x))         Φ(dx))
# = ∫(    I ,   g(Jᵩ·[t,t]ᵀ+τ) ·     N̂(t)   |det(Jᵩ)|dt )
#  where Jᵩ  = Fₖ
def g2(g, p, xi, l):
	F_k =  np.diag(p[1] - p[0])  # Fₖ = Jᵩ = dΦⁱ/dt
	tau_K = np.array([p[0]])       # τₖ = p₀
	detFk = abs(np.linalg.det(F_k))

	# Φₖ(ξ) = Fₖ·ξ + τₖ
	phi_k = lambda t: np.dot(F_k, np.array([t, t])) + tau_K;

	# reference element shape functions N̂ₗ(t) with t ∈ [0,1] 1D
	N_dach = [
		lambda t: 1 - t,
		lambda t: t,
	];

	#            g(Φ(ξ)) ·         N̂( ξ) ·|det Fₖ|
	return (g(*phi_k(xi)[0,:].tolist()) * N_dach[l](xi) * detFk)

# elemLoad(p, n, f)
#   returns the element load vector related to linear form l_K(v) = ∫(K, f·v dx) for linear FEM on triangles.
# input:
#   p - 3x2 matrix of the coordinates of the triangle nodes
#   n - order of the numerical quadrature (1 <= n <= 5)
#   f - source term function
# output:
#   fK - element load vector (3x1 array)
#
def elemLoad(p, n, f):
	# get nodes and weights for the gauss quadrature
	x, w = gaussTriangle(n) # x ∈ triange{(-1,1),(-1,-1),(1,-1)} 2D
	x = np.asarray(x) # leggauss Punkte × 2D Koodinate
	w = np.asarray(w) # leggauss Punkte × 1D Gewicht

	# transformiere die x (xi_schlange)
	x_tf = (x + 1) / 2

	# Werte g (Bezeichnung wie im Tutorium) an den transformierten Stützstellen aus und speichere die Ergebnisse in einem Vektor. g hängt von der reference element shape funktion ab!
	# entspricht g aus dem Tutorium für die erste reference element shape function
	g_vector = [[g3(f, p, x_tf[i], j) for i in range(0, len(x))] for j in range(3)]

	# Bilde das Skalarprodukt (letzte Zeile im Tutorium)
	# das ist f_unterstrich_K. Ein Eintrag für jedes reference shape element
	# K = ¼ wᵀ·g
	load_K = np.asmatrix([0.25 * np.dot(w, g_elem) for g_elem in g_vector]).T

	return load_K

# elemLoadNeumann(p, n, g)
#   returns the element vector related to the Neumann boundary data ∫(I, g·v ds) for linear FEM on the straight boundary edge I.
# input:
#   p - 2x1 matrix of the coordinates of the nodes on the boundary edge
#   n - order of the numerical quadrature
#   g - Neumann data as standard Python function or Python’s lambda function
# output:
#   gK - element vector (2x1 array)
def elemLoadNeumann(p, n, g):
	# get nodes and weights for the gauss quadrature
	x, w = np.polynomial.legendre.leggauss(n) # x ∈ [-1,1] 1D
	x = np.asarray(x) # leggauss Punkte × 1D Koodinate
	w = np.asarray(w) # leggauss Punkte × 1D Gewicht

	# transformiere die x (xi_schlange)
	x_tf = (x + 1) / 2 # x_tf ∈ [0,1]
	
	# 2D g
	# p    : ℝ²
	# x_tf : ℝ
	# j    : welches N̂
	# g    : ℝ × ℝ → ℝ
	# g2   : (ℝ × ℝ → ℝ) × ℝ² × ℝ × ℕ → ℝ
	g_vector = [[g2(g, p, x_tf[i], j) for i in range(0, len(x))] for j in range(2)]

	# K = ½ wᵀ·g
	g_K = np.asmatrix([0.5 * np.dot(w, g_elem) for g_elem in g_vector]).T

	return g_K

def localToGlobal(elemFnc, p, t):
	A = sp.csr_matrix((len(p),len(p))) # create "empty" (=zero-filled) sparse-matrix

	for triangle in t:
		Ak = sp.lil_matrix(elemFnc(p[triangle]))

		T = sp.lil_matrix((3,len(p)))
		T[[0,1,2],[triangle]] = 1
		
		T = T.tocsr()
		#Ak = Ak.toscr() # Der Befehl wäre ws sinnvoll,funktioniert bei mir (P) aber nicht
		At = T.T.dot(Ak).dot(T) # A += Tkᵀ·Ak·Tk
		A = A + At # At genauso groß wie A
	return A.tolil()

# stiffness(p, t)
#   returns the stiffness matrix related to the bilinear form ∫(Ω, grad u · grad v dx) for linear FEM on triangles.
# input:
#   p - Nx2 matrix with coordinates of the nodes
#   t - Mx3 matrix with indices of nodes of the triangles
# output:
#   Stiff - NxN stiffness matrix in scipy’s sparse lil format
def stiffness(p, t):
	# jeder Knoten hat eine base-function: len(p)
	#  jede base-function hat shape functions:
	#   jede shape function hat 3 Knotenintegrale
	return localToGlobal(elemStiffness,p,t)

# mass(p, t)
#   returns the mass matrix related to the bilinear form ∫(Ω, u·v dx) for linear FEM on triangles.
# input:
#   p - Nx2 matrix with coordinates of the nodes
#   t - Mx3 matrix with indices of nodes of the triangles
# output:
#   Mass - NxN mass matrix in scipy’s sparse lil format
def mass(p, t):
	return localToGlobal(elemMass,p,t)

# load(p, t, n, f)
#   returns the load vector related to the linear form ∫(Ω, f·v dx) for linear FEM on triangles.
# input:
#   p - Nx2 matrix with coordinates of the nodes
#   t - Mx3 matrix with indices of nodes of the triangles
#   n - order of the numerical quadrature (1 <= n <= 5)
#   f - source term function
# output:
#   Load - Nx1 load vector as numpy-array
def load(p, t, n, f):
	L = np.zeros((len(p),1)) # create array

	for triangle in t:
		Lk = elemLoad(p[triangle],n,f) #3x1 array

		#initialisieren und indexing als lil format
		T = sp.lil_matrix((3,len(p)))
		# Remark: 'fancy indexing' feature needs at least scipy.__version__ of 13.2
		T[[0,1,2],[triangle]] = 1
		#convert to csr for product with Ak
		T = T.tocsr()

		Lt = T.T.dot(Lk) # L += Tkᵀ·Lk·Tk
		L = L + Lt # Lt genauso groß wie L
	return L

# loadNeumann(p, be, n, g)
#   returns the vector related to the Neumann boundary data ∫(∂Ω, g·v ds) for linear FEM on straight boundary edges.
# input:
#   p - Nx2 matrix with coordinates of the nodes
#   be - Bx2 matrix with the indices of the nodes of boundary edges
#   n - order of the numerical quadrature
#   g - Neumann data as standard Python function or Python’s lambda function
# output:
#   LoadNeumann - Nx1 vector as numpy-array
def loadNeumann(p, be, n, g):
	L = np.zeros((len(p),1)) # create array

	for edge in be:
		Lk = elemLoadNeumann(p[edge],n,g) #2x1 array

		#initialisieren und indexing als lil format
		T = sp.lil_matrix((2,len(p)))
		# Remark: 'fancy indexing' feature needs at least scipy.__version__ of 13.2
		T[[0,1],[edge]] = 1
		#convert to csr for product with Ak
		T = T.tocsr()

		Lt = T.T.dot(Lk) # L += Tkᵀ·Lk·Tk
		L = L + Lt # Lt genauso groß wie L
	return L

