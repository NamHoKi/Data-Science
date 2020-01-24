import scipy as sp
import matplotlib.pyplot as plt

data = sp.genfromtxt('web_traffic.tsv', delimiter='\t')
print(data[:10])
print(data.shape) # -> (743, 2)
x = data[:,0] #1,2,3,4,5, ...
y = data[:,1] #2272, nan, ...
sp.isnan(y) # -> array of True(1) / False(0)'s
sp.sum(sp.isnan(y))
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

plt.scatter(x, y)
plt.title('Web traffic over a month')
plt.xlabel('Time')
plt.ylabel('Hits/Time')
plt.show()
