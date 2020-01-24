import scipy as sp
import matplotlib.pyplot as plt
import numpy as np

def error(f,x,y):
    return sp.sum((f(x)-y)**2)

data = sp.genfromtxt('web_traffic.tsv', delimiter='\t')
print(data[:10])
print(data.shape) # (743, 2)
x = data[:,0]
y = data[:,1]
sp.isnan(y)
sp.sum(sp.isnan(y))
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]


#fa, fb
inflection = int(3.5*7*24) # 기준점
xa = x[:inflection] #before
ya = y[:inflection]
xb = x[inflection:] #after
yb = y[inflection:]

fa = sp.poly1d(sp.polyfit(xa, ya, 1)) # before
fb = sp.poly1d(sp.polyfit(xb, yb, 1)) # after

error_a = error(fa, xa, ya)
error_b = error(fb, xb, yb)

print(error_a + error_b) #1.5664e+08


f1p, residue, rank, sv, rcond = sp.polyfit(x, y, 1,full=True)
print('Model coefs: %s' % f1p) # f(x) = 2.5961 * x + 989.0249
print(residue) #3.1739e+08

f1 = sp.poly1d(f1p)

xgrid = sp.linspace(1, x[-1],100)

plt.plot(xgrid, f1(xgrid),color='g', linewidth=4) # 초록직선

f2p = sp.polyfit(x, y, 2)
print('f2p : ',f2p) # array([ 1.0532e-02, -5.2655e+00, 1.9748e+03])
f2 = sp.poly1d(f2p)
plt.plot(xgrid, f2(xgrid),color='r', linewidth=4)

f3p = sp.polyfit(x, y, 3)
f3 = sp.poly1d(f3p)
plt.plot(xgrid, f3(xgrid),ls='dashed',color='orange', linewidth=4)

f10p = sp.polyfit(x, y, 10)
f10 = sp.poly1d(f10p)
plt.plot(xgrid, f10(xgrid),color='k', linewidth=4)

f100p = sp.polyfit(x, y, 100)
f100 = sp.poly1d(f100p)
plt.plot(xgrid, f100(xgrid),ls='dashdot',color='b', linewidth=4)
plt.legend(['d=%i' % f1.order,'d=%i' % f2.order,'d=%i' % f3.order,'d=%i' % f10.order,'d=%i' % f100.order], loc='upper left')

plt.plot(xgrid, fa(xgrid), color='c', linewidth=4)
plt.plot(xgrid, fb(xgrid), color='m', linewidth=4)

plt.ylim(np.min(y)-100,np.max(y)+100)
plt.xlabel('Time')
plt.ylabel('Hits/Time')
plt.scatter(x, y)
plt.show()
