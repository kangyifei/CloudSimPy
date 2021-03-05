from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
import math
x=[2,
2,
2,
4,
4,
5,
6,
6,
6,
7,
8,
]
y=[
8,
6,
16,
16,
13,
16,
16,
15,
14,
16,
16,
]
z=[79.07,
52.61,
61.18,
247.92,
3.04,
5.28,
0.25,
0.11,
0.11,
0.25,
0.25
]
z=[math.log10(ele) for ele in z]
x=np.array(x)
y=np.array(y)
z=np.array(z)
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(x,y,z)
plt.show()