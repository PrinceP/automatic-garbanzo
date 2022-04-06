import numpy as np
import sys


from numpy import dot
from numpy.linalg import norm



v1 = np.load(sys.argv[1])
v2 = np.load(sys.argv[2])


print(v1.shape)
print(v2.shape)
v1 = v1.reshape(1,int(sys.argv[3]))
v2 = v2.reshape(1,int(sys.argv[3]))



cos_sim = dot(v1, v2.T)/(norm(v1)*norm(v2))
print(cos_sim)
