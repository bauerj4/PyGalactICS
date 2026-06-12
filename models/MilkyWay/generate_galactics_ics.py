import numpy as np
import matplotlib.pylab as plt
import os
import pynbody
import sys

X0 = 0
Y0 = 0
Z0 = 0
VX0 = 0
VY0 = 0
VZ0 = 0

X0 = float(sys.argv[1])
Y0 = float(sys.argv[2])
Z0 = float(sys.argv[3])
VX0 = float(sys.argv[4])
VY0 = float(sys.argv[5])
VZ0 = float(sys.argv[6])

SQRT_FRIEDMANN_A = np.sqrt(1.0)
SQRT3_FRIEDMANN_A = SQRT_FRIEDMANN_A**3

out_name = "disk.dat"
# Get the particles

disk_m = []
disk_x = []
disk_y = []
disk_z = []
disk_r = []
disk_vx = []
disk_vy = []
disk_vz = []
disk_v = []

nparts = 0
npart_array = np.zeros(6)
stream_npart_array = []

galaxy = open("galaxy")
i = 0
for line in galaxy:
    line=line.split()
    disk_m.append(float(line[0]))
    disk_x.append(float(line[1]))
    disk_y.append(float(line[2]))
    disk_z.append(float(line[3]))
    disk_vx.append(float(line[4]))
    disk_vy.append(float(line[5]))
    disk_vz.append(float(line[6]))
    i= i + 1

nparts = i
print "There are " + str(nparts) + " disk particles."
disk_m = np.array(disk_m)
disk_x = np.array(disk_x) + X0
disk_y = np.array(disk_y) + Y0
disk_z = np.array(disk_z) + Z0
disk_r = np.array([disk_x, disk_y, disk_z]).T
disk_vx = 100. * np.array(disk_vx) / SQRT3_FRIEDMANN_A + VX0 * 1./SQRT3_FRIEDMANN_A
disk_vy = 100. * np.array(disk_vy) / SQRT3_FRIEDMANN_A + VY0 * 1./SQRT3_FRIEDMANN_A
disk_vz = 100. * np.array(disk_vz) / SQRT3_FRIEDMANN_A + VZ0 * 1./SQRT3_FRIEDMANN_A
disk_v = np.array([disk_vx, disk_vy, disk_vz]).T
print np.mean(disk_vx), np.mean(disk_vy), np.mean(disk_vz)


disk_m = disk_m *(10.0**10/(2.325 * 10**9))**-1
npart_array[2] = nparts

#print stream_m[0]
#print stream_halo_m[0]
print "Outputting Binary file -", out_name
print

try:
    with open( out_name ) as f:
        pass;
    print "Output file exists - deleting it.";
    os.remove( out_name );
except IOError:
    pass;

file = open(out_name,'wb');

buffer_size = np.array([256],np.uint32);
buffer_size.tofile(file,"",format="%i");

#npart_array = np.array([0,npart_array[2],npart_array[0],npart_array[1],0,0],np.uint32);
npart_array = np.array([0,nparts,0,0,\
                        0,0],np.uint32);

npart_array.tofile(file,"",format="%i");
print "The particle array is " + str(npart_array)
mass_array = np.array([0,disk_m[0],0,0,0,0],np.float64); # masses for each type 
mass_array.tofile(file,"",format="%d");

time = np.array([0],np.float64);
time.tofile(file,"",format="%d");

redshift = np.array([0],np.float64)
redshift.tofile(file,"",format="%d")

flagsfr = np.array([0],np.uint32)
flagsfr.tofile(file,"",format="%i")

flagFB = np.array([0],np.uint32)
flagFB.tofile(file,"",format="%i")

npart_array.tofile(file,"",format="%i")

flagcooling = np.array([0],np.uint32)
flagcooling.tofile(file,"",format="%i")

numfiles = np.array([0],np.uint32)
numfiles.tofile(file,"",format="%i")

boxsize = np.array([0],np.float64)
boxsize.tofile(file,"",format="%d")

Omega0 = np.array([0],np.float64)
Omega0.tofile(file,"",format="%d")

OmegaL0 = np.array([0],np.float64)
OmegaL0.tofile(file,"",format="%d")

H0 = np.array([0],np.float64)
H0.tofile(file,"",format="%d")

unused_buffer = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],np.uint32)
unused_buffer.tofile(file,"",format="%d")

buffer_size.tofile(file,"",format="%i")

# Write Positions
pos = np.array([disk_x,disk_y,disk_z]).T
value = sum(npart_array)*3*4
buffer_size = np.array([value],np.uint32)

buffer_size.tofile(file,"",format="%i")
coords = np.array(pos, np.float32)
coords.tofile(file,"",format="%f")
buffer_size.tofile(file,"",format="%i")

#Write Velocities                                                                                                    

vels = np.array([disk_vx, disk_vy, disk_vz]).T
buffer_size.tofile(file,"",format="%i")
vels   = np.array(vels, np.float32 )
vels.tofile(file,"",format="%f")
buffer_size.tofile(file,"",format="%i")

#Write IDs                                                                                                           

value = sum(npart_array)*4

buffer_size = np.array([value],np.uint32)

buffer_size.tofile(file,"",format="%i")
ids    = np.arange( 1, sum(npart_array) + 1, dtype=np.uint32 )
ids.tofile(file,"",format="%i")
buffer_size.tofile(file,"",format="%i")

file.close()

print "Validating..."
data = pynbody.load(out_name)

check_pos = np.transpose(data['pos'])

r = []
for p in data['pos']:
    r.append(np.linalg.norm(p))


v = []
for p in data['vel']:
    v.append(np.linalg.norm(p))



check_x = np.mean(check_pos[0])
check_y = np.mean(check_pos[1])
check_z = np.mean(check_pos[2])

check_vel = np.transpose(data['vel'])
check_vx = np.mean(check_vel[0])
check_vy = np.mean(check_vel[1])
check_vz = np.mean(check_vel[2])

print "Mean pos = "+ str([check_x,check_y,check_z])
print "Mean vel = "+ str([check_vx,check_vy,check_vz])


plt.hist(disk_vx, histtype='step',bins=50)
plt.hist(disk_vy, histtype='step',bins=50)
plt.hist(disk_vz, histtype='step',bins=50)
plt.show()


print disk_m
