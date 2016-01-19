import multiprocessing as mp
import numpy as np

sharedArray = mp.Array('i',10)
sharedValue = mp.Value('i')

def incrA(a,i):
  for j in range(10):
    a[j]+=i

def incrV(v,i):
  with v.get_lock():
    v.value+=i
    
def incrVs(v,i):
  with v[0].get_lock():
    v[0].value+=i
  with v[1].get_lock():
    v[1].value+=2*i
    
process = []
for i in range(10):
  process.append(mp.Process(target=incrA,args=(sharedArray,i)))
  process[i].start()
for i in range(10):
  process[i].join()

print sharedArray[0], sharedArray[1], sharedArray[2], sharedArray[3]

process = []
for i in range(1000):
  process.append(mp.Process(target=incrV,args=(sharedValue,i)))
  process[i].start()
for i in range(1000):
  process[i].join()
  
print sharedValue.value

v=0
for i in range(1000):
  v+=i

print int(0.5*999*1000)

sharedValue0 = mp.Value('i')
sharedValue1 = mp.Value('i')
sharedValues = [sharedValue0,sharedValue1]
process = []
for i in range(1000):
  process.append(mp.Process(target=incrVs,args=(sharedValues,i)))
  process[i].start()
for i in range(1000):
  process[i].join()

print sharedValues[0].value
print sharedValues[1].value


