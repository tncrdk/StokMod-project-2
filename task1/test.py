import numpy as np
import matplotlib.pyplot as plt


a = np.array([1, 1, np.nan, 4, 4, 4, 4, np.nan, 2, 2])
x = np.array([1, 2, 2, 2, 3, 4, 5, 5, 5, 6])

plt.plot(x, a)
plt.show()
