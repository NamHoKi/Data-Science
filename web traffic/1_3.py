import scipy as sp
import matplotlib.pyplot as plt

def error(f, x, y):
   return sp.sum((f(x)-y)**2) # Seems to be MSE error.

data = sp.genfromtxt('web_traffic.tsv', delimiter='\t')

print(data[:10])
print(data.shape) # -> (743, 2)

x = data[:,0] #1,2,3,4,5, ...
y = data[:,1] #2272, nan, ...

sp.isnan(y) # -> array of True(1) / False(0)'s
sp.sum(sp.isnan(y))

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

f1p, residue, rank, sv, rcond = sp.polyfit(x, y, 1,full=True)
print("Model coefs: %s" % f1p) # f(x) = 2.5961 * x + 989.0249
print(residue) #3.1739e+08

xgrid = sp.linspace(0, x[-1], 100)

import numpy as np
#fa, fb
inflection = int(3.5*7*24) # 기준점
xa = x[:inflection] #before
ya = y[:inflection]
xb = x[inflection:] #after
yb = y[inflection:]

#before
#d=1
fa = sp.poly1d(sp.polyfit(xa, ya, 1)) # before
#d=2
fa2 = sp.poly1d(sp.polyfit(xa, ya, 2)) # before
#d=3
fa3 = sp.poly1d(sp.polyfit(xa, ya, 3)) # before
#d=10
fa10 = sp.poly1d(sp.polyfit(xa, ya, 10)) # before
#d=100
fa100 = sp.poly1d(sp.polyfit(xa, ya, 100)) # before

#after
#d=1
fb = sp.poly1d(sp.polyfit(xb, yb, 1)) # after
#d=2
fb2 = sp.poly1d(sp.polyfit(xb, yb, 2)) # after
#d=3
fb3 = sp.poly1d(sp.polyfit(xb, yb, 3)) # after
#d=10
fb10 = sp.poly1d(sp.polyfit(xb, yb, 10)) # after
#d=100
fb100 = sp.poly1d(sp.polyfit(xb, yb, 100)) # after

#d=1
error_a = error(fa, xa, ya)
error_b = error(fb, xb, yb)

#d=2
error_a2 = error(fa2, xa, ya)
error_b2 = error(fb2, xb, yb)

#d=3
error_a3 = error(fa3, xa, ya)
error_b3 = error(fb3, xb, yb)

#d=10
error_a10 = error(fa10, xa, ya)
error_b10 = error(fb10, xb, yb)

#d=100
error_a100 = error(fa100, xa, ya)
error_b100 = error(fb100, xb, yb)

result = [error_a + error_b,error_a2 + error_b2,error_a3 + error_b3,error_a10 + error_b10,error_a100 + error_b100]
print(error_a + error_b) #1.5664e+08 / d=1
print(error_a2 + error_b2) #122584668.14611217 / d=2
print(error_a3 + error_b3) #122584668.14611217 / d=3
print(error_a10 + error_b10) #122584668.14611217 / d=10
print(error_a100 + error_b100) #122584668.14611217 / d=100

#before
#d=1
plt.plot(xgrid, fa(xgrid), color='c', linewidth=4)
#d=2
plt.plot(xgrid, fa2(xgrid), color='b', linewidth=4)
#d=3
plt.plot(xgrid, fa3(xgrid), color='g', linewidth=4)
#d=10
plt.plot(xgrid, fa10(xgrid), color='y', linewidth=4)
#d=100
plt.plot(xgrid, fa100(xgrid), color='w', linewidth=4)

plt.legend(['fa=%i' % fa.order, 'fa2=%i' % fa2.order, 'fa3=%i' % fa3.order, 'fa10=%i' % fa10.order, 'fa100=%i' % fa100.order], loc='upper left')

#범위지정
plt.ylim(np.min(y)-100,np.max(y)+100)

plt.scatter(x, y)
plt.title('Web traffic over a month')
plt.xlabel("Time")
plt.ylabel("Hits/Time")
plt.show()

#after
#d=1
plt.plot(xgrid, fb(xgrid), color='m', linewidth=4)
#d=2
plt.plot(xgrid, fb2(xgrid), color='r', linewidth=4)
#d=3
plt.plot(xgrid, fb3(xgrid), color='orange', linewidth=4)
#d=10
plt.plot(xgrid, fb10(xgrid), color='brown', linewidth=4)
#d=100
plt.plot(xgrid, fb100(xgrid), color='k', linewidth=4)

plt.legend(['fb=%i' % fb.order, 'fb2=%i' % fb2.order, 'fb3=%i' % fb3.order, 'fb10=%i' % fb10.order, 'fb100=%i' % fb100.order], loc='upper left')

#범위지정
plt.ylim(np.min(y)-100,np.max(y)+100)

plt.scatter(x, y)
plt.title('Web traffic over a month')
plt.xlabel("Time")
plt.ylabel("Hits/Time")
plt.show()

print(fb2(800))
