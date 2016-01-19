import meshes
import FEM
import time
import scipy.sparse.linalg as spla

meshwidth = 0.05
processes = 8

p,t=meshes.grid_square(1,meshwidth)

timer=time.time()
seqM=FEM.mass(p,t)
seqTime=time.time()-timer

timer=time.time()
parM,processes=FEM.massParallel(p,t,processes)
parTime=time.time()-timer

print "difference of matrices:        ", spla.onenormest(seqM-parM)
print "time for sequential assembling:", seqTime
print "time for parallel assembling:  ", parTime
print "speed-up:  ", seqTime/parTime
print "efficiency:", (seqTime/parTime)/processes
