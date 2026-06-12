import numpy as np
import matplotlib.pylab as plt
import sys

from scipy.interpolate import interp1d

A = 1.0
H = 0.679
G = 4.49*10**-6
SOFTENING = 0.4

diskVCirc = []
diskR = []

haloVCirc = []
haloR = []
m = -1

"""

Get disk rotation curve


"""

i = 0
vc = 0

f = open("disk")
for line in f:
    line = line.split()
    if (len(line) < 7):
        pass
    else:
        if (i == 0):
            m = float(line[0])/4.302
        r2 = float(line[1])**2 + float(line[2])**2 + float(line[3])**2
        diskR.append(np.sqrt(r2))
        i += 1

print i
#bins = np.linspace(0.1,40,200)
#dr = bins[1] - bins[0]
#hist, counts = np.histogram(haloR, bins=bins)
#plt.clf()

#plt.plot(bins[1:] - dr, bins[1:]**-1 * bins[1:]**-1 * hist)
#ax = plt.gca()
#ax.set_xscale('log')
#ax.set_yscale('log')
#plt.show()

diskR = np.sort(diskR)
i = 1
for r in diskR:
    vc =  10**5 * np.sqrt(G * m * i / r)
    #print force                
    diskVCirc.append(vc)
    i += 1



#f = open("vcirc1.out")
#for line in f:
#    line = line.split()
#    diskR.append(float(line[0]))
#    diskVCirc.append(float(line[1]) / H)
#
#f.close()

"""

Get halo rotation curve

"""

f = open("halo")
i = 0
#innerSums = np.zeros(len(haloR))
#outerSums = np.zeros(len(haloR))
#s1 = 0
#s2 = 0
vc = 0.

for line in f:
    line = line.split()
    if (len(line) < 7):
        pass
    else:
        if (i == 0):
            m = float(line[0]) / 4.302
        r2 = float(line[1])**2 + float(line[2])**2 + float(line[3])**2
        haloR.append(np.sqrt(r2))
        i += 1

bins = np.linspace(0.1,200,400)
dr = bins[1] - bins[0]
hist, counts = np.histogram(haloR, bins=bins)
plt.clf()

plt.plot(bins[1:] - dr, bins[1:]**-1 * bins[1:]**-1 * hist)
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
plt.show()
haloR = np.sort(haloR)

#print(haloR)

i = 1
for r in haloR:
    vc =  10**5 * np.sqrt(G * m * i / r)
    #print force
    haloVCirc.append(vc)
    i += 1

#haloVCirc = np.sqrt(-np.array(force) * np.array(haloR))
#print(haloVCirc)


haloVCircInterp = interp1d(haloR, haloVCirc, bounds_error=False, fill_value=0.)
diskVCircInterp = interp1d(diskR, diskVCirc, bounds_error=False, fill_value=0.)
rSpace = np.linspace(0,100,500)

total = np.sqrt(haloVCircInterp(rSpace)**2 + diskVCircInterp(rSpace)**2)

halo, = plt.plot(rSpace,haloVCircInterp(rSpace),  linewidth=2)
disk, = plt.plot(rSpace, diskVCircInterp(rSpace), linewidth=2)
tot, = plt.plot(rSpace, total, linewidth=2)
plt.xlim(0.,40.)
plt.xlabel(r"$R$ (kpc/$h$)", fontsize=18)
plt.ylabel(r"$v_c$ (km / s)", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend([halo,disk,tot], ['Halo', 'Disk', 'Total'])
plt.show()

f = open("rotation_curve.dat", "w")

for r,vt in zip(rSpace,total):
    f.write(str(r) + " " + str(vt) + " " + str(haloVCircInterp(r)) + " " + \
        str(diskVCircInterp(r)) + " " + str(0.) + "\n") 


