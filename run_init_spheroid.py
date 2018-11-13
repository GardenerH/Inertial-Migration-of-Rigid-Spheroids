# /usr/bin/env hoomd
import hoomd
from hoomd import md
from hoomd import deprecated
hoomd.context.initialize()

import os
import shutil
import re
import sys
import xml.etree.ElementTree as ET
import math as m
import numpy as np


######################
##### Init params #####
######################

# free parameters, units
# of particle diameter(1.0)
N_drop = 1 #number of droplets, only 0 or 1
#r_sphere = 4 #radius of droplet
L = 4 #major axis of the spheroid
B = 4 #minor axis of the spheroid

lx = 60.0
ly = 40.0
lz = 40.0
mass = 1.0
mass_f = 1.0
mass_p = 1/8

# dumps
dump_period = 1e2
rsz_steps = 1e3
eql_steps = 1e3

# geometry
wall_dens = 61.35 #from Millan et al., JCP (2007)
solv_dens = 4.0
sphere_dens = 4.0

#fcc number density
v_box = lx*ly*lz
v_sphere = 4./3.*m.pi*L*B**2
N_solv = int(solv_dens*v_box-sphere_dens*v_sphere) #solv number density

a = (4/wall_dens)**(1./3.) #wall fcc number density
b = 0.5 #sphere fcc number density
K = 125.0 #bond force constant

# get the job index from PBS_ARRAYID, or return 0 if it is not specified (for test jobs)
def get_array_id():
    pbs_arrayid = os.getenv('SLURM_ARRAY_TASK_ID');
    if pbs_arrayid is None:
        return 0
    else:
        return int(pbs_arrayid) - 1;

id = get_array_id();
fx_arr = [0.0005,0.001,0.002,0.003,0.004,0.005]
fx_fnl = fx_arr[id]
fName = '-eql-' + str(fx_fnl)

# random solvent **placer**, no system init via hoomd
def random_solv_plcr():
    # from lx,ly,lz randomly pluck positions
    for i in range(int(N_solv)):
        x = lx*np.random.rand()-lx/2.
        y = ly*np.random.rand()-ly/2.
        z = lz*np.random.rand()-lz/2.
        grid_solv_pos.append([x,y,z])
    return

# an alternative wall builder
def fcc_wall_builder():
    # create an array of bead positions
    xs = np.linspace(-lx/2.,lx/2.,m.ceil(lx/a))
    ys = np.linspace(-ly/2.,ly/2.,m.ceil(ly/a))
    zs = np.array([-1/4*lz-a/2.,-1/4*lz,-1/4*lz])#don't want overlaps
    for z in zs:
       for y in ys:
            for x in xs:
                # place the solvent particles
                if z == -1/4*lz:
                    x_new = x+a/2.
                    y_new = y+a/2.
                    if x_new > lx/2.:
                        x_new = x_new - lx
                    if y_new > ly/2.:
                        y_new = y_new - ly
                    grid_wall_pos.append([x_new,y_new,z])
                else:
                    grid_wall_pos.append([x,y,z])
    return

#rigid sphere builder
def fcc_rigid_sphere_builder():
    # create an array of bead positions
    xs = np.linspace(-16,16,round(4*16/b)+1)
    ys = np.linspace(-8,8,round(4*8/b)+1)
    zs = np.linspace(lz/4-8,lz/4+8,round(4*8/b)+1)
    for z in zs:
        if (2*(z-lz/4)/b)%2 == 0:
            for y in ys:
                for x in xs:
                    if ((x+y)/b)%1 == 0:
                        grid_cube_pos.append([x,y,z])
        if (2*(z-lz/4)/b)%2 == 1:
            for y in ys:
                for x in xs:
                    if ((x+y)/b)%1 == 1/2:
                        grid_cube_pos.append([x,y,z])
    for p in grid_cube_pos:
        r_diff = (p[1]**2 + (p[2]-lz/4)**2)/B**2
        if r_diff <= 1-(p[0]**2)/L**2:
                x = p[0]
                y = p[1]
                z = p[2]
                grid_sphere_pos.append([x,y,z])
    return

# declare pos array(s)
star_pos = []
grid_solv_pos = []
grid_cube_pos = []
grid_sphere_pos = []
grid_wall_pos = []
masses = []
diameters = []
bodies = []
types = []
bonds = []

# create the particle grid
solv_build = random_solv_plcr() #solvent
wall_build = fcc_wall_builder() #wall
sphere_build = fcc_rigid_sphere_builder() #rigid sphere


# ***TESTING***
print('Length of solv_pos: ' + str(len(grid_solv_pos)))
print('Length of wall_pos: ' + str(len(grid_wall_pos)))
print('Length of sphere_pos: ' + str(len(grid_sphere_pos)))
                  
# aggreate all arrays
full_pos = grid_solv_pos + grid_wall_pos + grid_sphere_pos

# build a single list for each particle field
masses.extend(int(N_solv)*[str(mass_f)]) #tack on the solvent
masses.extend(len(grid_wall_pos)*[str(mass)]) #tack on the wall
masses.extend(len(grid_sphere_pos)*[str(mass_p)])#tack on the sphere

                  
diameters.extend(int(N_solv)*['1.0']) #tack on the solvent
diameters.extend(len(grid_wall_pos)*['1.0']) #tack on the wall
diameters.extend(len(grid_sphere_pos)*['1.0']) #tack on the sphere

bodies.extend(int(N_solv)*['-1']) #tack on the solvent
bodies.extend(len(grid_wall_pos)*['-1']) #tack on the wall
bodies.extend(len(grid_sphere_pos)*['-1']) #tack on the sphere

types_solv = int(N_solv)*['S1']
types_wall = len(grid_wall_pos)*['wall']
types_sphere = len(grid_sphere_pos)*['S2']

types = types_solv + types_wall + types_sphere

#print('Print for testing')
print(len(masses))
print(len(diameters))
print(len(bodies))
print(len(types))
print(len(bonds))
print(len(full_pos))

# catch improper configurations
if len(full_pos) != len(masses):
    print('The position and mass arrays are not the same length!')
    sys.exit(-1)

# write out the file
with open('Init' + fName + '.xml', 'w') as inpFile:
    inpFile.write('\n'.join(['<?xml version="1.1" encoding="UTF-8"?>',
                             '<hoomd_xml version="1.5">',
                             '<configuration time_step = "0" dimensions = "3">']))
    inpFile.write('<box lx="{size1}" ly="{size2}" lz="{size3}" />\n'.format(size1=lx,size2=ly,size3=lz))
    inpFile.write('<position>\n' + '\n'.join('{} {} {}'.format(x, y, z) for (x, y, z) in full_pos)
        + '</position>\n')
    inpFile.write('<mass>\n' + '\n'.join(str(mass) for mass in masses) + '</mass>\n')
    inpFile.write('<diameter>\n' + '\n'.join(str(diameter) for diameter in diameters) + '</diameter>\n')
    inpFile.write('<body>\n' + '\n'.join(str(body) for body in bodies) + '</body>\n')
    inpFile.write('<type>\n' + '\n'.join(str(type) for type in types) + '</type>\n')
    inpFile.write('\n</configuration>\n</hoomd_xml>\n')



######################
###### Main Run ######
######################

# read in the Init file
system = deprecated.init.read_xml(filename='Init' + fName + '.xml', wrap_coordinates = True)

# add bond types via snapshot
snap = system.take_snapshot(bonds=True)
snap.bonds.types = ['tether1','tether2','tether3']
system.restore_snapshot(snap)

groupSphere =hoomd.group.type(name='groupSphere', type='S2')


tags_list1 = []
tags_list2 = []
tags_list3 = []
tags_spls = []
for p in groupSphere:
    tags1 = []
    tags2 = []
    tags3 = []
    tags_spls.append(p.tag)
    p_i = p.position
    for p in groupSphere:
        p_j = p.position
        r = ((p_j[0]-p_i[0])**2+(p_j[1]-p_i[1])**2+(p_j[2]-p_i[2])**2)**(1./2.)
        # VMD has issues with the num of bonds, test individually...
        if r >= m.sqrt(2)/2*b-0.01 and r <= m.sqrt(2)/2*b+0.01:
            tags1.append(p.tag)
        if r >= 1.0*b-0.01 and r <= 1.0*b+0.01:
            tags2.append(p.tag)
        if r >= m.sqrt(3)*b-0.01 and r <= m.sqrt(3)*b+0.01:
            tags3.append(p.tag)
    tags_list1.append(tags1)
    tags_list2.append(tags2)
    tags_list3.append(tags3)

cnt = 0
for t_i in tags_spls:
    for t_j in tags_list1[cnt]:
        if t_i == t_j:
            pass
        else:
            system.bonds.add("tether1", t_i, t_j)
    for t_j in tags_list2[cnt]:
        if t_i == t_j:
            pass
        else:
            system.bonds.add("tether2", t_i, t_j)
    for t_j in tags_list3[cnt]:
        if t_i == t_j:
            pass
        else:
            system.bonds.add("tether3", t_i, t_j)
    cnt += 1

# NOTE: in these simulations epsilon has ben rescaled so that the ODT for the given volume fraction is closer to that of the cores.
nl = md.nlist.cell()

harmonic = hoomd.md.bond.harmonic()
harmonic.bond_coeff.set('tether1',k=K,r0=m.sqrt(2)/2*b)
harmonic.bond_coeff.set('tether2',k=K,r0=1.0*b)
harmonic.bond_coeff.set('tether3',k=K,r0=m.sqrt(3)*b)

dpd = md.pair.dpd(r_cut=1.0, nlist=nl, kT=1.0, seed=id)

dpd.pair_coeff.set('wall', 'S1', A=3.0, gamma=4.5)
dpd.pair_coeff.set('wall', 'wall', A=0.0, gamma=0.0)
dpd.pair_coeff.set('S2','S1',A=3.0,gamma=4.5)
dpd.pair_coeff.set('S1','S1',A=25.0,gamma=4.5)
dpd.pair_coeff.set('S2','S2',A=0.0,gamma=0.0)
dpd.pair_coeff.set('S2','wall',A=10.0,gamma=4.5)

#make sure that the neighbor lists do not include particles that are bonded or are in the same body do not interact with one another
nl.reset_exclusions(exclusions = ['bond', 'body'])

md.integrate.mode_standard(dt=0.01)

# set up groups
all = hoomd.group.all()
groupWALL = hoomd.group.type(name='groupWALL', type='wall')
notWALL=hoomd.group.difference(name="particles-not-typeWALL", a=all, b=groupWALL)
md.integrate.nve(group=notWALL)

hoomd.run(rsz_steps)

# dump the final xml
deprecated.dump.xml(filename='Init' +  fName  + '.xml', all=True, group=all, image=False)