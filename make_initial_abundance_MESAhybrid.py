#!/bin/python


# This script creates the initial abundances for a particle trajectory
# it takes two arguments:
#   the file contaning the particle track (from which the initial radius is obtained)
#   the file containing the initial WD profile from which the initial abundance is obtained

# material has several components:

# 1. metallicity (in solar units)
#      this material consists of the solar "metals" (elements heavier than he4) 
#      with everything lighter than o18 converted to Ne22, all scaled by metallicity
# 2. simmering ashes
#      material produced from carbon burning during simmering phase
#      some mixture of c13, ne22, ne23, na23 etc,
#      but currently we will just make it all ne22
# 3. carbon
# 4. oxygen (1-everything else)

import sys
import math
import h5py
import numpy as np

#radius calc
def func_rad():
        # -----------------------------
        # get initial radial position
        # -----------------------------
        # Convert particle location to radius
        #rad = sqrt(r^2 + z^2)
        f = h5py.File('../alltracks_3D.hdf5','r')


        ids = f['trackids']
        ids = list(ids)
        initpos = f['initialpositions']
        initpos = np.array(initpos)
        inpt = int(sys.argv[2])
        whichid = ids.index(inpt)
        x = initpos[whichid,0]
        y = initpos[whichid,1]
        z = initpos[whichid,2]
        r = np.sqrt(x**2 + y**2 + z**2)

        return r


        
#core stuff
def func_core(r):
        # read nuclide set to fill
        nuctab_ref = open("nuclides200.txt")
        todoZ = []
        todoA = []
        todoname = []
        for line in nuctab_ref :
                sl=line.split()
                todoZ.append(int(sl[0]))
                todoA.append(int(sl[1]))
                todoname.append(sl[2])

        nuctab_ref.close()


        
        # now construct initial abundances
        # initialize our abundance dictionary to zeros
        initabund = dict()
        nucnames = dict()
        for i in range(len(todoZ)) :
                initabund[(todoZ[i],todoA[i])] = 0.0
                nucnames[(todoZ[i],todoA[i])] = todoname[i]


        # -----------------------------
        # open abundances file
        # radius density temperature c12 ne20 ne22
        # -----------------------------
        progenitor_file = "400k_flash_21new.dat"
        prog_ref = open(progenitor_file)

        # Need to look at 2 lines to see if the particle is in a radius range
        prevLine = None
        for line in prog_ref :
                if prevLine is not None:
                        sl  = prevLine.split()
                        sl2 = line.split()
                        lower_r_lim = float(sl[0])
                        upper_r_lim = float(sl2[0])

                        #If the radius of the particle is between that of the two lines
                        if lower_r_lim <= r and upper_r_lim > r:
                                #print(sl)
                                #print(sl2)
                                newl = [0]*(len(sl))
                                newl[0] = r
                                for i in range(1,6):
                                        #m = np.polyfit([sl[0],sl2[0]],[sl[i],sl2[i]],1)
                                        m = (float(sl2[i])-float(sl[i]))/(float(sl2[0])-float(sl[0]))
                                        newl[i] = m*(r-float(sl[0])) + float(sl[i])
                    
                                initabund[(6,12)]  = float(newl[3]) #c12
                                initabund[(10,20)] = float(newl[4]) #ne20
                                initabund[(10,22)] = float(newl[5]) #ne22
                        if r > 2.462000000000000e+08:
                                #FLUFF need to do something with this
                                initabund[(6,12)]  = 0.5E0
                                initabund[(10,20)] = 8.235124106228266E-3
                                initabund[(10,22)] = 6.913104313531712E-3
                                break
                prevLine = line  

        #print('{0} {1} {2} {3} {4} {5}'.format(*newl))
        #print('{:12.8e} {:12.8e} {:12.8e} {:12.8e} {:12.8e} {:12.8e}'.format(newl[0],newl[1],newl[2],newl[3],newl[4],newl[5]))

        prog_ref.close()
        
        # everything else is oxygen
        notO = 0.0
        #for abund in initabund.itervalues() :
        for abund in initabund.values() :
                notO = notO + abund

        initabund[(8,16)] = 1.0-notO


        #--------------------------
        # now write out initial abundances
        #print(initabund)
        
        #k = sorted(initabund)
        #k = initabund.keys()
        #sorted(k)
        #k.sort()

        
       # for nuc in k:
       #     Z= nuc[0]
       #     A= nuc[1]
       #     abund = initabund[nuc]
            
       #     print('{:3d} {:3d} {:12.8e} {}'.format(Z,A,abund,nucnames[nuc]))
        #inlist = open("inlist_test","w")
        
        c12 = initabund[(6,12)]
        ne20 = initabund[(10,20)]
        ne22 = initabund[(10,22)]
        o16 = initabund[(8,16)]

        return c12, ne20, ne22, o16

#shell stuff
def func_shell(r):
        # read nuclide set to fill
        nuctab_ref = open("nuclides200.txt")
        todoZ = []
        todoA = []
        todoname = []
        for line in nuctab_ref :
                sl=line.split()
                todoZ.append(int(sl[0]))
                todoA.append(int(sl[1]))
                todoname.append(sl[2])

        nuctab_ref.close()


        
        # now construct initial abundances
        # initialize our abundance dictionary to zeros
        initabund = dict()
        nucnames = dict()
        for i in range(len(todoZ)) :
                initabund[(todoZ[i],todoA[i])] = 0.0
                nucnames[(todoZ[i],todoA[i])] = todoname[i]


        # -----------------------------
        # open abundances file
        # DIFFERENT FILE NOT SURE IF FORMAT IS SAME AS LINE 78
        # -----------------------------
        progenitor_file = "1p1M_hybrid_to_Mch_bignet.mod"

        with open(progenitor_file) as f:
                prog_ref = f.readlines()[15:4018]
                
        #print(prog_ref)
        #column numbers of elements we care about
        abundcol = [17,20,22,26,28]

        # Need to look at 2 lines to see if the particle is in a radius range
        prevLine = None
        for line in prog_ref:
                
                if prevLine is not None:
                        sl  = prevLine.split()
                        sl2 = line.split()
                        #print("sl2 length:",len(sl2))
                        
                        sl = [sub.replace('D', 'E') for sub in sl]
                        sl2 = [sub.replace('D', 'E') for sub in sl2]
                        
                        #print("sl:", sl[3], type(sl[3]))
                        #print("sl2:", sl2[3], type(sl2[3]))
                        
                        lower_r_lim = np.exp(float(sl2[3]))
                        upper_r_lim = np.exp(float(sl[3]))

                        
                        #print("lower lim:", lower_r_lim)
                        #print("upper lim:", upper_r_lim)
                        
                        #If the radius of the particle is between that of the two lines
                        if lower_r_lim <= r and upper_r_lim > r:
                                
                                
                                newl = [0]*(len(sl))
                                newl[0] = r
                                for i in range(1,29):
                                        #if i in abundcol:       
                                        #m = np.polyfit([sl[0],sl2[0]],[sl[i],sl2[i]],1)
                                        m = (float(sl2[i])-float(sl[i]))/(float(sl2[0])-float(sl[0]))
                                        newl[i] = m*(r-float(sl[0])) + float(sl[i])
                                        #else:
                                                #continue
                    
                                initabund[(6,12)]  = float(newl[18]) #c12
                                initabund[(7,14)] = float(newl[21]) #n14
                                initabund[(8,16)] = float(newl[23]) #o16
                                initabund[(10,20)] = float(newl[27]) #ne20
                                c12,ne20,ne22,o16  = func_core(r)
                                initabund[(10,22)] = float(ne22) #ne22
                                
                                
                        if r > 2.462000000000000e+08:
                                #FLUFF need to do something with this
                                initabund[(6,12)]  = 0.5E0
                                initabund[(10,20)] = 8.235124106228266E-3
                                initabund[(10,22)] = 6.913104313531712E-3
                                break
                prevLine = line  

        #print('{0} {1} {2} {3} {4} {5}'.format(*newl))
        #print('{:12.8e} {:12.8e} {:12.8e} {:12.8e} {:12.8e} {:12.8e}'.format(newl[0],newl[1],newl[2],newl[3],newl[4],newl[5]))

        f.close()
        
        # everything else is oxygen
        #notO = 0.0
        #for abund in initabund.itervalues() :
        #for abund in initabund.values() :
        #        notO = notO + abund

        #initabund[(8,16)] = 1.0-notO


        #--------------------------
        # now write out initial abundances

        k = sorted(initabund)
        #k = initabund.keys()
        #k.sort()

        
        #for nuc in k:
        #    Z= nuc[0]
        #    A= nuc[1]
        #    abund = initabund[nuc]
        #    print('{:3d} {:3d} {:12.8e} {}'.format(Z,A,abund,nucnames[nuc]))
        #inlist = open("inlist_test","w")
        
        c12 = initabund[(6,12)]
        ne20 = initabund[(10,20)]
        ne22 = initabund[(10,22)]
        o16 = initabund[(8,16)]
        n14 = initabund[(7,14)]
        return c12, ne20, ne22, o16, n14

        nuctab_ref = open()
        

