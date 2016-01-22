#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import FEM, matplotlib.pyplot as plt, solver, math, numpy as np
try:
    import meshes
except ImportError as e:
    #Da meshpy bei mir nicht geht, benutze ich hier meine alte Version, bei der ich das grid "manuell" erstellt habe
    import meshes_meineVersion as meshes
    pass

#Blatt7/Aufgabe3
h0 = 0.5	# maximal mesh width
n = 3		# Integrationsordnung der Gauss Quadratur für den Elementloadvektor
steps = 7
energy_error = np.zeros(steps)
max_width = np.zeros(steps)

# definition of the source term f as Python function or Python's lambda function
# Setze Lösung u aus der Aufgabenstellung in die DGL ein und erhalte f, RB auch erfüllt
#  x[0,0] = x; x[0,1] = y
def f(x): return (8 * math.pi**2 + 1) * math.cos(2 * math.pi * x[0,0]) * math.cos(2 * math.pi * x[0,1])

#TODO 1: f2 exisitiert nur weil ich DAS SCHEISS FORMAT NICHT HINGEKRIEGT HAB-.-
#bei f_vector = list(map(f2,p)) hat p "eine Klammer zu wenig", so dass man hier x[0] statt x[0,0] schreiben muss
#def f2(x): return (8 * math.pi**2 + 1) * math.cos(2 * math.pi * x[0]) * math.cos(2 * math.pi * x[1])

for i in range (0,steps):
	h = h0*2**(-i/2.0)
	print(h)
	# create random grid
	#(p, t, be)=meshes.grid_random(1,h)
	(p, t, be) = meshes.grid_square(1, h)

	# solve -∆u + u = f on Ω system with Neumann boundary conditions, where grad u · n = 0 on ∂Ω
	u_n = solver.solve_n0(p, t, be, f, n)
	#FEM.plot(p, t, u_n, "numerical solution for n = " + str(n) + "und h_" + str(i) + " = " + str(h))

	#discretization error in the energy norm

	###TODO 2: Stimmt das f oben? auf dem übungsblatt steht nur pi, nicht pi^2..aber in series7 war es auch pi^2. dementsprechend ändert sich uU auch die Lösung u und damit integral über f*u
	#print(p)
	#p_new = [[[p1[0],p1[1]]] for p1 in p]
	f_vector = FEM.load(p,t,n,f)
	#f_vector = list(map(f2,p))
	#print(f_vector)
	# TODO 3: ist das richtig, dass das b(..,..) negativ ist. Eigentlich sollte es positiv sein, evtl sollte man die Reihenfolge der Differenz umdrehen..
	energy_error[i] = math.sqrt(abs(0.25 + 2 * math.pi**2 - u_n.dot(f_vector))) #falls f nur pi (nicht pi^2) enthält ist das integral über f*u = 2 * math.pi + 0.25
	#TODO 4: für unstructured meshes: Ich kann ja nur ohne gmesh arbeiten und hab nur eine selbstgeschriebene funktion für structured meshes.
	#Diese Zeile funzt hoffentlich bei dir. evtl musst du hier noch was ändern "# create random grid":
	#max_width[i] = meshes.max_mesh_width(t, p)
	#bei mir mach ich das mal so:
	max_width[i] = h
#print(p)

print(max_width)
print(energy_error)

# plot discretization error in the energy norm for different mesh sizes
#TODO 5: ich hab jetzt so einen plot mit double log axes hinbekommen, musste dafür xlim und ylim extra setzen und da verliert der plot dann die Zahlen auf den Achsen. Das sollte noch schöner werden

plt.loglog(max_width, energy_error, label='energy error')
plt.xlim([min(max_width), max(max_width)])
plt.ylim([min(energy_error), max(energy_error)])

plt.legend()
plt.xlabel('max_width')
plt.ylabel('energy_error')
plt.show()