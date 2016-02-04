#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# checkout pipe notation: http://dev-tricks.net/pipe-infix-syntax-for-python
#  sudo pip3 install pipe
#  from pipe import *
#  u_g=map(g,t[I]) | list() | np.array()
#  ..this does not work, but nice intent!

import FEM, scipy.sparse.linalg as spla, math, matplotlib.pyplot as plt, numpy as np, copy, scipy.sparse as sp
try:
    import meshes
except ImportError as e:
    #Da meshpy bei mir nicht geht, benutze ich hier meine alte Version, bei der ich das grid "manuell" erstellt habe
    import meshes_meineVersion as meshes
    pass

# create system matrices for triangular FEM
def create_system(p, t, n, f):
    # computation of the stiffness and mass matrix
    # conversion of the sparse matrices from Scipy's lil_matrix format to some other format that is more appropriate for solving the system, e.g. use Scipy's csr format
    MASS  = FEM.mass(p,t).tocsr()       # b(u,v) = ∫( Ω, u·v dx)
    STIFF = FEM.stiffness(p,t).tocsr()  # b(u,v) = ∫( Ω, grad u · grad v dx)

    # computation of the load vector in dependence of the source term f and the order n of the numerical integration
    LOAD = FEM.load(p,t,n,f)            #   l(v) = ∫( Ω, f·v dx)

    return MASS, STIFF, LOAD

# create system matrices for triangular FEM for mixed variational formulation
def create_system_mixed(p, t, n, f, g, be):
    # computation of the stiffness, mass and load matrices as before
    MASS, STIFF, LOAD = create_system(p, t, n, f)

    # computation of the load vector in dependence of the source term f and the order n of the numerical integration
    LOAD_BE = FEM.loadNeumann(p,be,n,g) #          ∫(∂Ω, g·v ds)  Neumann boundary data

    return MASS, STIFF, LOAD, LOAD_BE

# the indicies within this section could be determined also easily by len(be) which seperates the boundary elements from the interior ones but we don't assume this here and implemented a more general approach
#  we do not assume a full range of indices
#  we do not assume ...
#  works always, but may be slow! O(n · log n)
def interiorNodes(p, t, be):
    T=copy.copy(t)                                # O(n)
    T=T.reshape((T.shape[0]*T.shape[1],1)).T      # O(1)
    T.sort()                                      # O(n · log n)
    T=np.unique(T)                                # O(n)

    BE=copy.copy(be)                              # O(n)
    BE=BE.reshape((BE.shape[0]*BE.shape[1],1)).T  # O(1)
    BE.sort()                                     # O(n · log n)
    BE=np.unique(BE)                              # O(n)

    # could be realised in O(n)
    return np.array(list(filter(   # O(n)
        lambda x: x not in BE, T)  #      · O(log n)
    )), BE


#like the function above but specifically for problem 10/3.	return nodes that are not on the dirichlet boundary (i.e. interior nodes or nodes on the neumann boundary)
def non_Dirichlet_Nodes(p, t, be):
    T=copy.copy(t)                                # O(n)
    T=T.reshape((T.shape[0]*T.shape[1],1)).T      # O(1)
    T.sort()                                      # O(n · log n)
    T=np.unique(T)                                # O(n)

    BE=copy.copy(be)                              # O(n)
    BE=BE.reshape((BE.shape[0]*BE.shape[1],1)).T  # O(1)
    BE.sort()                                     # O(n · log n)
    BE=np.unique(BE)                              # O(n)
    
    # could be realised in O(n)
    # use BE and the coordinate of the points to determine whether it is in the interior or on the neumann boundary
    return np.array(list(filter(   # O(n)
        lambda x: x not in BE or p[x][1] > 0 , T)  #      · O(log n)
    ))


	
# solve
#     -∆u + u = f  on  Ω
#  grad u · n = 0  on ∂Ω (Neumann boundary conditions)
def solve_n0(p, t, be, f, n):
    MASS, STIFF, LOAD = create_system(p,t,n,f)

    # solution of the linear system using Scipy�s spsolve
    return spla.spsolve(MASS + STIFF,LOAD) # works

# solve
#     -∆u + u = f  on  Ω
#           u = 0  on ∂Ω (Dirichtlet boundary conditions)
def solve_d0(p, t, be, f, n):
    MASS, STIFF, LOAD = create_system(p,t,n,f)

    # get interior nodes
    I, notI = interiorNodes(p, t, be)

    # u_0 auf dem Rand ist 0, im Inneren die Lösung des reduzierten LGS
    u_n=np.zeros(len(p))
    u_n[I]= spla.spsolve((MASS+STIFF)[I,:][:,I],LOAD[I]) # streiche Zeilen und Spalten nacheinander A[I,:][:,I] ≠ A[I,I]

    # rekonstruiere u_n
    return u_n

# solve
#     -∆u + u = f  on  Ω
#           u = g  on ∂Ω (Dirichtlet boundary conditions)
def solve_d(p, t, be, f, n, g):
    MASS, STIFF, LOAD = create_system(p,t,n,f)

    # get interior nodes
    I, notI = interiorNodes(p, t, be)

    # u_g ist im Inneren 0, auf dem Rand ≡ g
    u_g=np.zeros((len(p),1))
    u_g[notI,0]=np.array(list(map(lambda pt: g(*tuple(pt)),p[notI]))) # tuple : np.array → tuple, * : tuple → parameter list

    # Systemmatrix aufstellen: SYSTEM · u_g = LOAD
    SYSTEM=MASS+STIFF
    LOAD=LOAD-SYSTEM.dot(u_g)

    # u_0 auf dem Rand ist 0, im Inneren die Lösung des reduzierten LGS
    u_0=np.zeros(len(p))
    u_0[I]= spla.spsolve(SYSTEM[I,:][:,I],LOAD[I]) # streiche Zeilen und Spalten nacheinander A[I,:][:,I] ≠ A[I,I]

    # rekonstruiere u_n
    return u_0+u_g.reshape(u_g.shape[0])

# solve
#            -∆u = f  on  Ω ①(u)
#     grad u · n = g  on ∂Ω ② (Neumann boundary conditions)
# where
#    ∫(Ω, f·dx) = -∫(∂Ω, g·dS) ③ compatibility condition
#    ∫(Ω, u·dx) =  0           ④
# bei der normalen variationellen Formulierung fordern wir das auch über den Raum. Der Raum ist da H1_sternchen, also H1 mit integral über u = 0
# die DGL nur mit der NBC ist nicht eindeutig lösbar, (nur bis auf eine Konstante eindeutig)
# und genau deswegen fordern wir integral über u = 0. Damit wir die Konstante fixieren und eine eindeutige Lösung bekommen
#  ③: [Skript 20151110]: For pure Neumann b.c. AND c=0 we are in H¹(Ω)
#     we can add constants to u s.t. 0 = ∫(Ω, f·1 dx) + ∫(∂Ω, g·1 dS) → the solution is not present in this eq., only sources are influenced
#     ⇒ "compatibility condition" to the sources has to be satisfied
#       a) either we fix the constant in the solution by demanding a vanishing mean i.e. we switch to H¹_*(Ω) = {v∈H¹(Ω). ∫(Ω, v dx) = 0} (LAST HOMEWORK)
#       b) or we demand ④
# 
# mixed variational formulation
#  b(u,v) + λ ∫(Ω, v·dx) = l(v), ∀ v∈H¹(Ω) ⑤
#             ∫(Ω, u·dx) = 0               ⑥
# with u=uᵢvⁱ leads to
#    b(uᵢvⁱ,vʲ)    = b(vⁱ,vʲ)·uᵢ    ≅ A·u
#    λₙ∫(Ω, vⁱ·dx)                  ≅ λₙ·m
#    ∫(Ω, uᵢvⁱ·dx) = ∫(Ω, vⁱ·dx)·uᵢ ≅ mᵀ·u # = <m, u> = l(u)
#  A ·u+λₙ·m = f
#  mᵀ·u      = 0
#
# stokes
#   ∫(Ω,df) = ∫(∂Ω,f)
# integration by parts Ⓟ
#           d(ab)  =      da b  +      a db    on Ω
#   ⇒ ∫( Ω, d(ab)) = ∫(Ω, da b  +      a db)
#   ⇔ ∫(∂Ω,   ab)  = ∫(Ω, da b) + ∫(Ω, a db)
#   ⇔ ∫( Ω, da b)  = ∫(∂Ω,  ab) - ∫(Ω, a db)
#
# convention
#   MASS    = b(u,v) = ∫( Ω, u·v dx)
#   STIFF   = b(u,v) = ∫( Ω, grad u · grad v dx)
#   LOAD    =   l(v) = ∫( Ω, f·v dx)
#   LOAD_BE =          ∫(∂Ω, g·v ds)  Neumann boundary data
#    BV = {p¹ … pN} of Vn
#    BW = {q¹ … qN} of Wn
#
# variational formulation of ①, ∀ v∈H¹(Ω), s.t. ①(u) ⇒ Ⓘ(u)
#  -∫(Ω, ∆u·v dx) = ∫(Ω,f·v dx) Ⓘ
#     da = ∆u ⇒  a = grad u
#     b  = v  ⇒ db = grad v
# ⇔ ∫(Ω, grad u grad v dx) = ∫(Ω,f·v dx) + ∫(∂Ω,   grad u v n·dS), with Ⓟ (this is the variational formulation Ⓥ)
# ⇔ ∫(Ω, grad u grad v dx) = ∫(Ω,f·v dx) + ∫(∂Ω,        g v n·dS), with ②
#     u = uᵢvⁱ = Σᵢ uᵢ·vⁱ with vⁱ ∈ V
# ⇔ ∫(Ω, grad uᵢvⁱ grad vʲ dx)  = ∫(Ω,f·vⁱ dx) + ∫(∂Ω, g vⁱ n·dS), with ∀ v∈H¹(Ω) ⇔ ∀ vⁱ ∈ BV, S = span(H¹(Ω))
# ⇔ ∫(Ω, grad vⁱ grad vʲ dx)·uᵢ = ∫(Ω,f·vⁱ dx) + ∫(∂Ω, g vⁱ n·dS)
# ⇔           STIFF         ·u  =    LOAD      +     LOAD_BE
def solve_mixed_g0(p, t, be, f, n, g):
    _, STIFF, LOAD, LOAD_BE = create_system_mixed(p,t,n,f,g,be)

    #m = sp.csr_matrix((len(p),1))
    m = FEM.load(p, t, n, lambda x: 1.0)
    Z = sp.csr_matrix((1,1))

    # solution of the linear system using Scipy's spsolve
    # [STIFF  m;
    #  m.T    0]
    SYSTEM = sp.vstack([sp.hstack([STIFF, m]), sp.hstack([m.T,Z])])
    RHS = sp.vstack([LOAD+LOAD_BE, Z])
    plt.spy(SYSTEM)
    #plt.show()
    u = spla.spsolve(SYSTEM,RHS)
    return u[0:len(u)-1], u[len(u)-1] # return only u, lambda

# create system matrices for triangular FEM
def create_systemP2(p, t, n, f, ei): # TODO: same as create_system but with ei
    # computation of the stiffness and mass matrix
    # conversion of the sparse matrices from Scipy's lil_matrix format to some other format that is more appropriate for solving the system, e.g. use Scipy's csr format
    MASS  = FEM.massP2(p,t,ei).tocsr()       # b(u,v) = ∫( Ω, u·v dx)
    STIFF = FEM.stiffnessP2(p,t,ei).tocsr()  # b(u,v) = ∫( Ω, grad u · grad v dx)

    # computation of the load vector in dependence of the source term f and the order n of the numerical integration
    LOAD = FEM.loadP2(p,t,n,f,ei).tocsr()         #   l(v) = ∫( Ω, f·v dx)

    return MASS, STIFF, LOAD

# solve
#     -∆u + u = f  on  Ω
#  grad u · n = 0  on ∂Ω (Neumann boundary conditions)
def solve_n0_P2(p, t, n, f):
    ei = meshes.edgeIndex(p,t)
    MASS, STIFF, LOAD = create_systemP2(p, t, n, f, ei)

    # solution of the linear system using Scipy's spsolve
    return spla.spsolve((MASS + STIFF),LOAD) # works


# solve
#     -∆u = f  on  Ω
#  grad u · n = 0  on ∂ΩN (Neumann boundary conditions)
#           u = 0  on ∂ΩD (Dirichtlet boundary conditions)
def solve_mixed_bc(p, t, be, n, f):
    MASS, STIFF, LOAD = create_system(p, t, n, f)

    # get interior nodes and boundary nodes on the Neumann boundary part = non dirichlet nodes!
    non_D_Nodes = non_Dirichlet_Nodes(p, t, be)
    
    # u_0 auf dem Rand ist 0, im Inneren die Lösung des reduzierten LGS
    u_n = np.zeros(len(p))
    u_n[non_D_Nodes] = spla.spsolve((STIFF)[non_D_Nodes,:][:,non_D_Nodes],LOAD[non_D_Nodes]) # streiche Zeilen und Spalten nacheinander A[I,:][:,I] ≠ A[I,I]

    # rekonstruiere u_n
    return u_n