import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# data = np.array(np.random.rand(1000))
# y, binEdges = np.histogram(data, bins=100)
# bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
# plt.hist(data, 100)
# plt.plot(bincenters, y, '-', c='r')
# plt.show()

f, ax = plt.subplots()
d = np.random.normal(size=100)
sns.distplot(d, hist=False, ax=ax)
plt.show()
