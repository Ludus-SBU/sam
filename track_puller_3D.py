import sys
import h5py
import numpy as np
import subprocess as sp
import make_initial_abundance_MESAhybrid_3D

f = h5py.File(sys.argv[1],'r')
wanttrackid = int(sys.argv[2])
#print("Track id: ", wanttrackid,"File: ",f)

trackids = f["trackids"][:]

# find the index of track we're looking for
# start in the obvious place
ti = wanttrackid -1
if ( trackids[ti] != wanttrackid ) :
	print("didn't find desired track id where it was expected to be")
	print("need to implement search")
	exit(2)

#Find where in tracks particle is
#Particles can have different track lengths (allows for reconstructed particles)
trackstart = f["trackstarts"][ti]
tracklength = f["tracklengths"][ti]
trackdata = f["trackdata"]
#d contains the TIME density and temperature
#forget d it isnt helpful just pull r calculated in script below
d = trackdata[ trackstart:trackstart+tracklength , 0:3]
#print(d)
#move this into core section of if statement and have similar call in shell section
r  = make_initial_abundance_MESAhybrid_3D.func_rad()

#r = 2.15E8
#print("r:",r)
#Edit inlist to have new initial abundance values
#if d[0,1] < 2e5:
#determine if in the core or shell then edit inlist accordingly 
if r > 2.140619329540536E8:

        c12,ne20,ne22,o16,n14  = make_initial_abundance_MESAhybrid_3D.func_shell(r)
        with open("inlist_one_zone_burn_shell","r") as file:
                shelldata = file.readlines()

        shelldata[9] = "     init_shell_c12 =  " + str(c12) + "\n"
        shelldata[10] = "     init_shell_n14 =  " + str(n14) + "\n"
        shelldata[11] = "     init_shell_o16 =  " + str(o16) + "\n"
        shelldata[13] = "     init_shell_ne20 =  " + str(ne20) + "\n"
        shelldata[14] = "     init_shell_ne22 =  " + str(ne22) + "\n"

        with open("inlist_one_zone_burn_shell","w") as file:
                file.writelines(shelldata)
                #Shell=true here means let python use bash/default shell

        sp.check_call('cp inlist_one_zone_burn_shell inlist_one_zone_burn',shell=True)
else:
        c12,ne20,ne22,o16  = make_initial_abundance_MESAhybrid_3D.func_core(r)
        with open("inlist_one_zone_burn_core","r") as file:
                coredata = file.readlines()
        #print("Inlist data: ", coredata)
        coredata[5] = "     init_c12 =  " + str(c12) + "\n"
        coredata[6] = "     init_ne20 =  " + str(ne20) + "\n"
        coredata[7] = "     init_ne22 =  " + str(ne22) + "\n"
        coredata[9] = "     init_o16 =  " + str(o16) + "\n"

        with open("inlist_one_zone_burn_core","w") as file:
                file.writelines(coredata)
        
        sp.check_call('cp inlist_one_zone_burn_core inlist_one_zone_burn',shell=True)

actuallength = tracklength
for i in range(tracklength):
	if i >= 1 and d[i-1,0] == d[i,0]:
		actuallength -= 1
#the history in question
print(actuallength)
for i in range(tracklength):
	if i == 0 or d[i-1,0] != d[i,0]:
		print("%20.14e\t\t%20.14e\t\t%20.14e\t\t%20.14e\n" % (d[i,0], np.log10(d[i,2]), np.log10(d[i,1]), 1.0 ))


