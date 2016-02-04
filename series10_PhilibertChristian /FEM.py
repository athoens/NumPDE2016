#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np, scipy.sparse as sp, matplotlib.pyplot as plt
try:
    import meshes
except ImportError as e:
    #Da meshpy bei mir nicht geht, benutze ich hier meine alte Version, bei der ich das grid "manuell" erstellt habe
    import meshes_meineVersion as meshes
    pass

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
#   computes the element stiffness matrix related to the bilinear form aₖ(u,v) = ∫(K, grad u . grad v dx) for linear FEM on triangles.
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
	
	
### TODO: Diese Funktion wurde noch nicht getestet
def g3P2(f, p, xi, l):
	F_k = np.array([p[1] - p[0], p[2] - p[0]]).T # Fₖ = [p₁-p₀,p₂-p₀]ᵀ = Jᵩ = dΦⁱ/dξ
	tau_K = np.array([p[0]])                     # τₖ = p₀
	detFk = abs(np.linalg.det(F_k))

	# Φₖ(ξ) = Fₖ·ξ + τₖ
	phi_k = lambda xi: np.dot(F_k, xi) + tau_K;

	# reference element shape functions N̂ₗ(ξ) with ξ ∈ triange{(-1,1),(-1,-1),(1,-1)} 2D
	N_dach = [
		lambda xi: 1 - xi[0] - xi[1],
		lambda xi: xi[0],
		lambda xi: xi[1],
		lambda xi: xi[0]*xi[1],					#lambda2*lambda3
		lambda xi: (1 - xi[0] - xi[1])*xi[1],	#lambda1*lambda3
		lambda xi: (1 - xi[0] - xi[1])*xi[0]	#lambda1*lambda2
	];
	# das geht nicht:
#	N_dach = [
#		lambda xi: 1 - xi[0] - xi[1],
#		lambda xi: xi[0],
#		lambda xi: xi[1],
#		lambda xi: N_dach[2](xi)*N_dach[3](xi),
#		lambda xi: N_dach[1](xi)*N_dach[3](xi),
#		lambda xi: N_dach[1](xi)*N_dach[2](xi)
#	];

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

def localToGlobal(elemFnc, p, t): # erzeugt Massenmatrix M aus Mk und Stiffnessmatrix A aus Ak
	A = sp.csr_matrix((len(p),len(p))) # create "empty" (=zero-filled) sparse-matrix

	for triangle in t:  # for each K
		Ak = sp.lil_matrix(elemFnc(p[triangle])) # TODO

		T = sp.lil_matrix((3,len(p)))
		T[[0,1,2],[triangle]] = 1

		T = T.tocsr()
		#Ak = Ak.toscr() # Der Befehl wäre ws sinnvoll,funktioniert bei mir (P) aber nicht
		At = T.T.dot(Ak).dot(T) # A += Tkᵀ·Ak·Tk  # TODO
		A = A + At # At genauso groß wie A
	return A.tolil()

# load(p, t, n, f)
#   returns the load vector related to the linear form ∫(Ω, f·v dx) for linear FEM on triangles.
# input:
#   p - Nx2 matrix with coordinates of the nodes
#   t - Mx3 matrix with indices of nodes of the triangles
#   n - order of the numerical quadrature (1 <= n <= 5)
#   f - source term function
# output:
#   Load - Nx1 load vector as numpy-array
def load(p, t, n, f): # erzeugt Lastvektor F aus Fk # TODO: reduce this to local2Global variant
	F = np.zeros((len(p),1)) # create array

	for triangle in t:
		Fk = elemLoad(p[triangle],n,f) #3x1 array  # TODO

		#initialisieren und indexing als lil format
		T = sp.lil_matrix((3,len(p)))
		# Remark: 'fancy indexing' feature needs at least scipy.__version__ of 13.2
		T[[0,1,2],[triangle]] = 1
		#convert to csr for product with Ak
		T = T.tocsr()

		Ft = T.T.dot(Fk) # L += Tkᵀ·Lk # TODO
		F = F + Ft # Lt genauso groß wie L
	return F

# loadNeumann(p, be, n, g)
#   returns the vector related to the Neumann boundary data ∫(∂Ω, g·v ds) for linear FEM on straight boundary edges.
# input:
#   p - Nx2 matrix with coordinates of the nodes
#   be - Bx2 matrix with the indices of the nodes of boundary edges
#   n - order of the numerical quadrature
#   g - Neumann data as standard Python function or Python’s lambda function
# output:
#   LoadNeumann - Nx1 vector as numpy-array
def loadNeumann(p, be, n, g):  # TODO: reduce this to local2Global variant
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

# Quadratic finite (triangle) elements discrete variational problem (DVP)
#
# shape functions Nᵢ(λ)
#  N₀(λ)=   λ₁  linear
#  N₁(λ)=   λ₂  linear
#  N₂(λ)=   λ₃  linear
#  N₃(λ)=λ₂·λ₃  quadratic
#  N₄(λ)=λ₁·λ₃  quadratic
#  N₅(λ)=λ₁·λ₂  quadratic
# where
#  λⱼ are the barycentric coordinates
#  pₖ are the nodes in counter-clockwise order
#      λⱼ(x) = (x - pⱼ₊₁) · ⊥(pⱼ₋₁ - pⱼ₊₁) / 2|K|
#  ⇒ d λⱼ(x) =              ⊥(pⱼ₋₁ - pⱼ₊₁) / 2|K|
#  (Gₖ)ᵢⱼ  = dλⱼ · dλᵢ
#          = Dₖᵀ·Dₖ / 4|K|
#
# for β₁,β₂,β₃ ∈ ℕ it holds, that
#  ∫(K, λ₁ᵝ¹·λ₂ᵝ²·λ₃ᵝ³ dx) = 2|K| · β₁!β₂!β₃! / (β₁+β₂+β₃+2)! ①
# ⇒   β₁ β₂ β₃    ∫
#  N₀ 1         2|K|/3!
#  N₁    1      2|K|/3!
#  N₂       1   2|K|/3!
#  N₃    1  1   2|K|/4!
#  N₄ 1     1   2|K|/4!
#  N₅ 1  1      2|K|/4!
# 
# so
#  aₖ(u,v) = ∫(K, grad u · grad v dx)
#          = ∫(K,     du ·     dv dx)
# where
#  u(x) = uᵢ·vⁱ(x)
#  v(x) = vᵢ·vⁱ(x)
# with the linear element functions vⁱ
#  vⁱ(pᵢ) = 1 on point pᵢ                         Ⓛ
#  
# and the quadratic element functions vⁱ⁺ʲ
#  vⁱ⁺ʲ(pⱼ₀) = vⁱ⁺ʲ(pⱼ₁) = 0                      Ⓠ
#  on the edge eⱼ
#   eⱼ = {t·(pⱼ₁ - pⱼ₀) + pⱼ₀, for t ∈ [0,1]} ⊆ Ω ⓔ
#  for the neighbouring nodes pⱼ₀, pⱼ₁
#

# elemStiffnessP2(p)
#
# computes the element stiffness matrix related to the bilinear form
#  aₖ(u,v) = ∫(K, grad u · grad v dx) 
# for quadratic FEM on triangles.
#
# input:
#  p - 3x2-matrix of the coordinates of the triangle nodes
# output:
#  AK - element stiffness matrix
def elemStiffnessP2(p):
	Ak = np.zeros((6,6))
	
	x = p[:, 0]
	y = p[:, 1]
	Dk = np.array([
		y[[1, 2, 0]] - y[[2, 0, 1]],
		x[[2, 0, 1]] - x[[1, 2, 0]]
	])
	detFK = np.linalg.det(np.array([
		p[1, :] - p[0, :],
		p[2, :] - p[0, :]
	]).T)
	
	#print(Dk)
	#print(detFK)
	
	Gk = 1.0 / (4.0 * 0.5 * detFK) * Dk.T.dot(Dk) # hier war vorher detFK**2! Das war ein Fehler in der Angabe
	#print(Gk)
	#Ab hier wird die Matrix Ak mit Werten besetzt:
	#Erläuterungen siehe Schmierzettel bzw Fotos die ich in den Ordner hochgeladen habe
	
	#BLOCK OBEN LINKS
	Ak[0:3,0:3] = Gk

	#BLOCK OBEN RECHTS
	#Für die 3x3 Blockmatrix oben rechts von Ak gilt (siehe Zettel):
	#Spalte 0 ergibt sich aus Spalte 1 + Spalte 2 von Gk,
	#Spalte 1 ergibt sich aus Spalte 0 + Spalte 2 von Gk und
	#Spalte 2 ergibt sich aus Spalte 1 + Spalte 2 von Gk
	#jeweils mit Vorfaktor (1/3) * |K|, wobei |K| = (1/2) * detFK
	for i in range(3,6):
		j = (i+1)%3
		k = (i-1)%3
		#print(i)
		#print(j)
		#print(k)
		Ak[0:3,i] = (detFK/6)*(Gk[:,j]+Gk[:,k])
		
	#BLOCK UNTEN LINKS ergibt sich aus Symmetrie
	Ak[3:6,0:3] = Ak[0:3,3:6].T
	
	#BOCK UNTEN RECHTS
	#Vorfaktor ist (1/12)* |K|, wobei |K| = (1/2) * detFK
	#off-diagonal:
	for i in range(3,6):
		for j in range(3,6):
			#falls j!=i, also nur für off-diagonal
			if(j != i):
				#i modulo3, j modulo3, k sollen 0,1,2 sein (beliebige Reihenfolge) (Permutation), wähle also k, so dass k nicht i und nicht j ist
				# wenn j = (i + 1) modulo 3, dann soll k = (i + 2) modulo 3
				if ((i+1)%3 == (j%3)):
					k = (i+2)%3
				# wenn j = (i + 2) modulo 3, dann soll k = (i + 1) modulo 3
				if ((i+2)%3 == (j%3)):
					k = (i+1)%3
				#print(i)
				#print(j)
				#print(k)
				Ak[i,j] = (detFK/24) * (Gk[k,k] + Gk[k,i-3] + Gk[j-3,k] + 2*Gk[i-3,j-3])

	#Diagonale
	for k in range(3,6):
		i = (k+1)%3
		j = (k+2)%3
		Ak[k,k] = (detFK/24) * 2 *(Gk[i,i] + Gk[j,j] + Gk[i,j])
	
	return Ak

#import meshes_meineVersion as meshes
#(p,t,be) = meshes.grid_square(1,0.52)
#print(p)
#print(t)
#Ak = elemStiffnessP2(p[t[0,:]])
#print(Ak)

#
# elemMassP2(p)
#
# computes the element mass matrix related to the bilinear form m_K(u,v) = ∫(K u v dx) for quadratic FEM on triangles.
#  vᵢ ∈ B{Vₙ}
#  uₙ = Σuᵢvᵢ
#  ∫(K uᵢvᵢ vⱼ dx)
#  uᵢ ∫(K vᵢ vⱼ  dx)
#  
#  
# input:
# p - 3x2-matrix of the coordinates of the triangle nodes
#
# output:
# MK - element mass matrix
#
def elemMassP2(p):
	Mk = np.zeros((6,6))
	
	x = p[:, 0]
	y = p[:, 1]
	Dk = np.array([
		y[[1, 2, 0]] - y[[2, 0, 1]],
		x[[2, 0, 1]] - x[[1, 2, 0]]
	])
	detFK = np.linalg.det(np.array([
		p[1, :] - p[0, :],
		p[2, :] - p[0, :]
	]).T)
	
	#Benutze ganz viel Gleichung 1 vom Aufgabenblatt um die Einträge zu berechnen
	#BLOCK OBEN LINKS
	#Permutationen von 2 Lambas. Auf der Diagonalen stehen nicht-gemischten, also die (lambda_i)^2
	#off-diagonal: 1/4!, diagonal: 2/4!
	Mk[0:3,0:3] = (1/24) * (np.ones((3,3)) + np.identity(3))

	#BLOCK OBEN RECHTS
	#Permutationen von 3 Lambas. Auf der Diagonalen stehen die komplett gemischten: lambda1*lambda2*lambda3, sonst sind 2 lambdas doppelt
	#off-diagonal: 2/5!, diagonal: 1/5!	
	Mk[0:3,3:6] = (1/120) * (np.full((3,3),2) - np.identity(3))
	
	#BLOCK UNTEN LINKS ergibt sich aus Symmetrie
	Mk[3:6,0:3] = Mk[0:3,3:6].T
	
	#BLOCK UNTEN RECHTS
	#Permutationen von 4 Lambas. Auf der Diagonalen stehen (lambda_i)^2*(lambda_j)^2, sonst lambdai*lambdaj*lambdak^2
	#off-diagonal: 2/6!, diagonal: 4/6!
	Mk[3:6,3:6] = (1/360) * (np.ones((3,3)) + np.identity(3))
	
	#alle Einträge werden noch mit 2*|K| = detFK multipliziert
	Mk = detFK * Mk
	return Mk
	
#import meshes_meineVersion as meshes
#(p,t,be) = meshes.grid_square(1,0.52)
#Mk = elemMassP2(p[t[0,:]])
#print(Mk)

	
# elemLoadP2(p, n, f)
#
# returns the element load vector related to linear form
#  lₖ(v) = ∫(K, f·v dx)
# for quadratic FEM on triangles.
#
# input:
#  p - 3x2 matrix of the coordinates of the triangle nodes
#  n - order of the numerical quadrature (1 <= n <= 5)
#  f - source term function
# output:
#  fK - element load vector (3x1 array)
def elemLoadP2(p,n,f):
	# get nodes and weights for the gauss quadrature
	x, w = gaussTriangle(n) # x ∈ triange{(-1,1),(-1,-1),(1,-1)} 2D
	x = np.asarray(x) # leggauss Punkte × 2D Koodinate
	w = np.asarray(w) # leggauss Punkte × 1D Gewicht

	# transformiere die x (xi_schlange)
	x_tf = (x + 1) / 2

	# Werte g (Bezeichnung wie im Tutorium) an den transformierten Stützstellen aus und speichere die Ergebnisse in einem Vektor. g hängt von der reference element shape funktion ab!
	# entspricht g aus dem Tutorium für die erste reference element shape function
	g_vector = [[g3P2(f, p, x_tf[i], l) for i in range(0, len(x))] for l in range(6)]

	# Bilde das Skalarprodukt (letzte Zeile im Tutorium)
	# das ist f_unterstrich_K. Ein Eintrag für jedes reference shape element
	# K = ¼ wᵀ·g
	load_K = np.asmatrix([0.25 * np.dot(w, g_elem) for g_elem in g_vector]).T

	return load_K

	
# local to global
#  the matrix view of Tkᵀ·Ak·Tk is missleading here (see https://www.youtube.com/watch?v=4l-qzZOZt50&t=1h57m21s)
# for a map Φ, whe transform it with M via
#   M⁻¹ Φ M if Φ ∈ T¹₁V ≅ V  ⊗ V* ≅ {T: V* × V → K} ≅ endomorphism on V*
#   Mᵀ  Φ M if Φ ∈ T⁰₂V ≅ V* ⊗ V* ≅ {T: V  × V → K} ≅ bilinear form
# Aₖ is the matrix representation of a bilinear form of the shape functions aₖ(uᵢ,vⱼ) on each K
# A  is the transformed bilinear form of the element functions on Ω
#   A = Σₖ Aₖ
# for T[[0,1,2],[triangle]] = 1
#      A   += Tᵀ·Aₖ·T translates 
#   to Aᵢʲ += Tᵢˢ·Aₖʳₛ·Tʲᵣ
#   to Aᵢʲ += Tᵢˢ·Tʲᵣ·Aₖʳₛ
#   to Aᵢʲ += Tᵢˢ·Tʲᵣ·Aₖ(ϵʳ,εₛ)
#   to Aᵢʲ += Aₖ(Tʲᵣϵʳ,Tᵢˢεₛ)
#   to Aᵢʲ += Aₖ(Tʲ,Tᵢ)
#   to A(?) = Aₖ(?)
# …
# die Idee ist, dass bei den node-functions jedes K genau 3 shape functions erzeugt hat
# aus der Zuordnung der 3K-shape functions zu den K element functions, wird dann die inverse zuordnung bestimmt
#  T: ∀ edges ∀ local shapes. weighten that

# wir wollen         ∫(Ω, f·vᵢ dx) 
# wir haben ∀ K, Nᵢ. ∫(K, f·Nᵢ dx)
# DoF: p, ei
def localToGlobalP2(elemFnc, p, t, ei, Ttrafo, elemParams):  # ⚡ add ei

	N = len(p)		#Anzahl Knoten
	M = ei.getnnz() #ei.getnnz() liefert die Anzahl an nicht-0 Elementen in ei; sollte der Anzahl der Kanten entsprechen
	A = None
	
	for triangle in t: # for each K
		Ak = sp.lil_matrix(elemFnc(p[triangle],*elemParams))
		
		#erstelle T_nodes wie bei linearen FE
		T_nodes = sp.lil_matrix((3,N))
		T_nodes[[0,1,2],[triangle]] = 1

		#T ist die T-Matrix für die quadratischen FE (N+M) x (N+M), wobei N = Anzahl Knoten und M = Anzahl Kanten
		# T = [ node-DoF      0
		#           0     edge-DoF ]
		#T enthält oben links die T-Matrix der linearen FE, rechts oben und links unten 0-Blöcke
		#und rechts unten eine 1 für jede Kante: Dabei entspricht der Zeilenindex (-3) dem lokalen Index und der Spaltenindex (-N) dem globalen Index der Kante
		
		T = sp.lil_matrix((6,N + M))
		T[0:3,0:N] = T_nodes

		#erhalte lokale Kanten des lokalen Dreiecks
		edges = [	[triangle[0], triangle[1]],
					[triangle[1], triangle[2]],
					[triangle[2], triangle[0]]
				] # 3 edges
		
		#print(edges)
		
		local_edge_index = 1 #1,2,3 (nicht 0,1,2)
		for e in edges:
			#für jede lokale Kante (2 globale Knotenindices) liefert ei den globalen Kantenindex
			#ei is lower triangular Matrix, enthält Kantenindices: 1-M (nicht 0)
			
			# Der index der von ei geliefert wird ist double und nicht int. macht das probleme? falls  ja in der funktion edgeindex beheben
			global_edge_index = ei[max(e[0],e[1]),min(e[0],e[1])] #damit die Einträge links unten indiziert werden
			#print(global_edge_index)
			if(global_edge_index == 0): print("global_edge_index = 0 in FEM.py, but it should start with 1!")
			
			#schreibe in T [0,1,2... x 0,1,2...] eine 1 in die Zeile 2 + lokalenKantenIndex und die Spalte (N -1) + gloablenKantenIndex
			T[2 + local_edge_index, int((N - 1) + global_edge_index)] = 1
			local_edge_index += 1

		#Für alle Dreiecke K: Multipliziere Ak von beiden Seiten mit Tk
		#und summiere über alle Beiträge/Dreiecke
		T = T.tocsr()
		#At = T.T.dot(Ak).dot(T) # A += Tkᵀ·Ak·Tk
		At = Ttrafo(Ak,T)

		if A is None: # 1st loop
			A = At
		else: # all other loops
			A = A + At # At genauso groß wie A
	return A.tolil()

#import meshes_meineVersion as meshes
#(p,t,be) = meshes.grid_square(1,0.52)
#print(p)
#print(t)
#ei = meshes.edgeIndex(p,t)
#A = localToGlobalP2(elemStiffnessP2,p,t,ei)
#print(A.todense())
	
# old an busted: gDoF = #(nodes) = len(p)
# new hotness:   gDoF = #(nodes) + #(edged) = len(p) + len(ei)
# A  ∈ gDoF × gDoF
# Ak ∈ lDoF × lDoF

# ei - edge Index
def stiffnessP2(p, t, ei):
	trafoMatrix = lambda Ak, T: T.T.dot(Ak).dot(T)
	return localToGlobalP2(elemStiffnessP2,p,t,ei,trafoMatrix,())  # erzeugt (globale) Stiffnessmatrix A aus (lokaler) Ak

# ei - edge Index
def massP2(p, t, ei):
	trafoMatrix = lambda Mk, T: T.T.dot(Mk).dot(T)
	return localToGlobalP2(elemMassP2,p,t,ei,trafoMatrix,()) # erzeugt (globale) Massenmatix M aus (lokaler) Mk

# ei - edge Index
def loadP2(p, t, n, f, ei):
	trafoVektor = lambda Fk, T: T.T.dot(Fk)
	return localToGlobalP2(elemLoadP2,p,t,ei,trafoVektor,(n,f)) # erzeugt (globale) Massenmatix M aus (lokaler) Mk

