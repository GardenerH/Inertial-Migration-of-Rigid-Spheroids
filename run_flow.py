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

# dumps
dump_period = 1e4
flw_steps = 1e7

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

# force params
fx_arr = [0.0005,0.001,0.002,0.003,0.004,0.005]
fx_fnl = fx_arr[id]
fName = '-eql-' + str(fx_fnl)

# read in the Init file
system = deprecated.init.read_xml(filename='Init' + fName + '.xml')

harmonic = hoomd.md.bond.harmonic()
harmonic.bond_coeff.set('tether1',k=K,r0=m.sqrt(2)/2*b)
harmonic.bond_coeff.set('tether2',k=K,r0=1.0*b)
harmonic.bond_coeff.set('tether3',k=K,r0=m.sqrt(3)*b)

# pair forces
nl = md.nlist.cell()
dpd = md.pair.dpd(r_cut=1.0, nlist=nl, kT=1., seed=id)

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
groupFLUID = hoomd.group.type(name='groupFLUID', type='S1')
groupPAR = hoomd.group.type(name='groupPAR', type='S2')
notWALL=hoomd.group.difference(name="particles-not-typeWALL", a=all, b=groupWALL)

md.integrate.nve(group=notWALL)

#start the logs, including restart file


# dump the system data - position, velocity
# NOTE: this is compressed system data that
# will be post processed later with gtar
zipfile = hoomd.dump.getar('dump' + fName + '.zip',
                static=['type'],
                dynamic={'position': dump_period, 'velocity': dump_period})

# logs
#hoomd.analyze.log('energies-flow' + fName + '.txt', quantities=['temperature', 'potential_energy', 'kinetic_energy'], header_prefix='#', period = dump_period, overwrite = True)

# start up the pos writer
pos = deprecated.dump.pos(filename="dump-flow" + fName + ".pos", period=dump_period)
pos.set_def('S1', 'sphere 1 CC0000')
pos.set_def('wall', 'sphere 1 336600')
pos.set_def('S2', 'sphere 1 0000FF')

# apply the constant force
const = md.force.constant(fx=fx_fnl, fy=0.0, fz=0.0,group=groupFLUID)
const = md.force.constant(fx=fx_fnl, fy=0.0, fz=0.0,group=groupPAR)
hoomd.run(flw_steps)

#hoomd.dump.dcd('traj-flow' + fName + '.dcd', period = 600, overwrite = True)
#hoomd.run(120000)

# dump the final xml
deprecated.dump.xml(filename='Final-flow' + str(fx_fnl) + '.xml', all=True, group=all)

