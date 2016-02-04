#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import FEM, matplotlib.pyplot as plt, solver, math, numpy as np, graded_mesh

try:
    import meshes
except ImportError as e:
    #Da meshpy bei mir nicht geht, benutze ich hier meine alte Version, bei der ich das grid "manuell" erstellt habe
    import meshes_meineVersion as meshes
    pass

#Blatt10/Aufgabe3
h0 = 0.5 # 1.0/3.0	# maximal mesh width
n = 3		# Integrationsordnung der Gauss Quadratur für den Elementloadvektor
N = 25
steps = 5 # 5
simtype = ['square','random','graded']
energy_error = np.zeros((steps,len(simtype)))
energy_norm_of_u_n = np.zeros((steps,len(simtype)))
energy_norm_of_u = 1.548888
np.full
max_width = np.zeros((steps,len(simtype)))

# −∆u = 1 in (-1,1)^2, (1a)
#  grad u · n = 0  on ∂ΩN (Neumann boundary conditions)
#           u = 0  on ∂ΩD (Dirichtlet boundary conditions)

# definition of the source term f as Python function or Python's lambda function
def f(x): return 1

# graded ∈ {0,1}
def compute(simid):
	for i in range (0,steps):
		h = h0*2**(-i/2.0)
		print(h)

		# auf diese weise erzeuge ich ein mesh für (-1,1)^2
		if simid == 1:
			graded_mesh.createbgMesh("bgmesh.pos", [-1,1], [-1,1], h, N)
			(p, t, be) = meshes.grid_graded(1, h, "bgmesh.pos")
		elif simid == 2:
			(p, t, be) = meshes.grid_random(2, h)
			p = p - [1,1]
		else:
			(p, t, be) = meshes.grid_square(2, h)
			p = p - [1,1]

		# solve −∆u = 1 in (-1,1)^2 with mixed boundary conditions
		u_n = solver.solve_mixed_bc(p, t, be, n, f)

		#discretization error in the energy norm
		f_vector = FEM.load(p,t,n,f)

		# compute energy norm
		energy_norm_of_u_n[i,simid] = u_n.dot(f_vector)
		energy_error[i,simid] = abs(energy_norm_of_u - math.sqrt(abs(energy_norm_of_u_n[i,simid]))) #falls f nur pi (nicht pi^2) enthält ist das integral über f*u = 2 * math.pi + 0.25

		#TODO 4: für unstructured meshes: Ich kann ja nur ohne gmesh arbeiten und hab nur eine selbstgeschriebene funktion für structured meshes.
		#Diese Zeile funzt hoffentlich bei dir. evtl musst du hier noch was ändern "# create random grid":
		#max_width[i] = meshes.max_mesh_width(t, p)
		#bei mir mach ich das mal so:
		max_width[i,simid] = h
	FEM.plot(p, t, u_n, "numerical solution for n = " + str(n) + "und h_" + str(i) + " = " + str(h))

for simid in range(len(simtype)):
	print("Simulation: " + simtype[simid])
	compute(simid)

print("max_width")
print(max_width)
print("energy_error")
print(energy_error)

# plot discretization error in the energy norm for different mesh sizes
#TODO 5: ich hab jetzt so einen plot mit double log axes hinbekommen, musste dafür xlim und ylim extra setzen und da verliert der plot dann die Zahlen auf den Achsen. Das sollte noch schöner werden
plt.figure()
p=plt.loglog(max_width, energy_error)
plt.title('energy norm of u-u_n')
plt.xlim([np.min(np.min(max_width)), np.max(np.max(max_width))])
plt.ylim([np.min(np.min(energy_error)), np.max(np.max(energy_error))])
plt.legend(p,simtype)
plt.xlabel('max_width')
plt.ylabel('energy_of_u_n')
plt.show()

