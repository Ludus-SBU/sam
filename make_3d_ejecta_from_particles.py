# Josh Martin 2026 based off work by Sam Boos

#needs abundance_runs, final_checkpoint, and particle_track in directory
# script to make 2D ejecta from post-processed particle data
#import matplotlib.pyplot as plt
import numpy
import h5py
import yt
import sys

particlenum=100001
dirsize=300

deltav = 500e5 #JM May need to change this
maxv=4.5e9 #JM May need to change this
lastparticle=particlenum

print(lastparticle)
print(dirsize)
# get final positions and velocities for all tracks

pf = h5py.File("../tracks.hdf5")

fpos = pf['finalpositions'][:]
fvel = pf['finalvelocities'][:]
leftgrid = pf['leftgrid'][:]
weights = pf['weights'][:]
fgpot = pf['gpot'][:]

# normalize weights
sw = sum(weights)
weights = weights/sw

numtracks = len(pf['trackids'])
fvelr = numpy.zeros( numtracks ) # initializing final radial velocity array
fr = numpy.zeros( numtracks ) # initializing final position array

# determine average expansion time ( slope of r vs. v_r relation )
# said another way, texp is the average of r/v_rad over all particles.
# texp is the homologous expansion timescale. Essentially it is how long since the explosion.

texpinv = 0.0
for i in range(numtracks) :
	fr[i] = numpy.sqrt(fpos[i][0]**2 + fpos[i][1]**2 + fpos[i][2]**2)
	fvelr[i] = ( fpos[i][0]*fvel[i][0] + fpos[i][1]*fvel[i][1] + fpos[i][2]*fvel[i][2]) / fr[i]
	texpinv = texpinv + fvelr[i]/fr[i]

texpinv = texpinv / numtracks
texp = 1.0/texpinv

print('t_exp = ', texp)


#  allocate grid 

vgridsize = int(maxv/deltav)

ejectamassdens = numpy.zeros( ( 2*vgridsize, 2*vgridsize, 2*vgridsize ) )
ejectatemp = numpy.zeros( (2*vgridsize, 2*vgridsize, 2*vgridsize ) )
vx = numpy.zeros( (2*vgridsize, 2*vgridsize, 2*vgridsize ) )
vy = numpy.zeros( (2*vgridsize, 2*vgridsize, 2*vgridsize ) )
vz = numpy.zeros( (2*vgridsize, 2*vgridsize, 2*vgridsize ) )
totmass = 0.0

ds = yt.load("../final_checkpoint")

#print( dir(ds.fields.gas) )

ad = ds.all_data()

# Now looping over every cell on the grid
# This for loop cuts out the bound remnant and the fluff material from our data
# and then sorts the remaining ejecta into velocity bins.

for i in range( ad['dens'].size ) :
	x = float(ad['x'[i]]) # For each iteration, we extract the position, cell-size, and velocity of the cell.
	y = float(ad['y'][i])
	z = float(ad['z'][i])
	dx = float(ad['dx'][i])
	dy = float(ad['dy'][i])
	dz = float(ad['dz'][i])
	velx = float(ad['velx'][i])
	vely = float(ad['vely'][i])
	velz = float(ad['velz'][i])

	r = numpy.sqrt( x**2 + y**2 + z**2 ) # Finding the distance from the origin
	v_rad = ( x*velx + y*vely + z*velz ) / r # Finding the radial velocity (this is a projection formula,
	# not sqrt of the sum of the squares because we want speed along the radial direction NOT overall speed)

	#if ( dr > deltav*texp ) :
	#	print( 'deltav is %d but dr/texp is %d'%( deltav, dr/texp) )
	#	print( '(vr,vz) =  ( %d, %d )'%(velr,velz) )

	# only include ejected material w/ sufficiently low fluff mass fraction.
	if ( float(ad['ener'][i]) - float (ad['eint'][i]) + float(ad['gpot'][i]) > 0 and float(ad['flff'][i]) < 0.01 ) :
		mass = float( ad['density'][i] ) * dx * dy * dz
		totmass += mass

		# Source grid cell may be bigger than destination grid cell, so, if so, need to use source
		# grid dx and allow for contribution more than just 4 cells
		# If cell is smaller than target grid size, use target grid size so that smoothing is consistent
		# with CIC (cell-in-cloud) used for particles
		# Using source grid dv is a little off since we are using the velocity in the cell to map to the new grid
		# instead of its spatial coordinates.  But we just use t_exp to convert the spatial extent
		# into a velocity extent.  This won't quite match up in regions where the expansion law is
		# not quite fit by r = v*t_exp, but it will be quite close.

		# Finding the velocity-width of a cell
		srcdvx = max(dx/texp, deltav)
		srcdvy = max(dy/texp, deltav)
		srcdvz = max(dz/texp, deltav)

		# Calculating the range of velocities of a cell
		xedgelo = velx - 0.5*srcdvx
		xedgehi = velx + 0.5*srcdvx
		yedgelo = vely - 0.5*srcdvy
		yedgehi = vely + 0.5*srcdvy
		zedgelo = velz - 0.5*srcdvz
		zedgehi = velz + 0.5*srcdvz

		# Finding the index for the velocity cell that the each of the velocity bounds belong too
		# Explaining each part
		# mingridi = min( 2*vgridsize,     ## The maximum grid index is at 2*vgridsize - 1, so any velocities larger than vmax will
										   ## have index 2*vgridsize and will be skipped over in the next loop.
		# max( 0,                          ## this is the lowest possible index which is the left edge of a given axis
		# int( numpy.floor( xedgelo/deltav ## using floor to get an integer value and xedgelo/deltav is the
										   ## index of the lowest velocity in a cell
		# +vgridsize ) ) ) )               ## vgridsize is the index of "halfway" so adding by vgridsize will
										   ## place a velocity of 0 in the central velocity bin.
		mingridi = min( 2*vgridsize, max( 0, int( numpy.floor( xedgelo/deltav+vgridsize ) ) ) )
		maxgridi = min( 2*vgridsize-1, max( -1, int( numpy.floor( xedgehi/deltav+vgridsize ) ) ) )
		mingridj = min( 2*vgridsize, max( 0, int( numpy.floor( yedgelo/deltav+vgridsize ) ) ) )
		maxgridj = min( 2*vgridsize-1, max( -1, int( numpy.floor( yedgehi/deltav+vgridsize ) ) ) )
		mingridk = min( 2*vgridsize, max( 0, int( numpy.floor( zedgelo/deltav+vgridsize ) ) ) )
		maxgridk = min( 2*vgridsize-1, max( -1, int( numpy.floor( zedgehi/deltav+vgridsize ) ) ) )
		
		for gridi in range( mingridi, maxgridi+1) :
			for gridj in range( mingridj, maxgridj+1) :
				for gridk in range(mingridk, maxgridk+1) :
					# algorithm for weights:
					# we find the overlap length along each axis and multiply them, then divide by the bin widths to normalize
					# basic idea: [min(redgehi, right edge of bin) - max(redgelo, left edge of bin)] / bin_width
					weight = ( min( xedgehi, (gridi-vgridsize+1)*deltav ) - max( xedgelo, (gridi-vgridsize)*deltav ) ) * ( min( yedgehi, (gridj-vgridsize+1)*deltav ) - max( yedgelo, (gridj-vgridsize)*deltav ) ) * ( min( zedgehi, (gridk-vgridsize+1)*deltav ) - max( zedgelo, (gridk-vgridsize)*deltav ) ) / srcdvx / srcdvy / srcdvz
					ejectamassdens[gridi,gridj,gridk] += weight * mass
					ejectatemp[gridi,gridj,gridk] += weight * mass * float(ad['temperature'][i])

del ad
del ds
avgdens = totmass / ( 4.0/3.0*numpy.pi*maxv**3*texp**3 )
print( 'avgdens = ', avgdens )
# now convert mass in each bin to density.  trimming to spherical
for i in range(2*vgridsize) :
	for j in range(2*vgridsize) :
		for k in range(2*vgridsize) :
			# Storing the velocity values that represent each velocity grid-cell.
			vx[i,j,k] = (i-vgridsize+1)*deltav # Doing -vgridsize+1 allows us to have our negative 
											   # velocity values since vgridsize is the "halfway" index.
			vy[i,j,k] = (j-vgridsize+1)*deltav
			vz[i,j,k] = (k-vgridsize+1)*deltav

			# JM - If code is bugging, may need to deal with this
			# trim to be spherical, since it is cut in r an z directions by grid
			# if ( numpy.sqrt( (i+0.5)**2 + (j-vgridsize+0.5)**2) > vgridsize  or  ejectamassdens[i,j] == 0.0 ) :
			
			# Filling empty bins with negligible density and temperature so downstream code never deals
			# with "truly zero" values.
			if ( ejectamassdens[i,j,k] == 0.0 ) :
				ejectamassdens[i,j,k] = avgdens*1e-20
				ejectatemp[i,j,k] = 100.0
			else :
				# If the bin has mass, we get the true temperature of the bin by dividing off the mass
				# since we initially found the temperature density of the bin by weighting each initial cell
				# according to the mass of a cell that had a specific velocity. Now we are just getting
				# dividing off the total mass.
				ejectatemp[i,j,k] = ejectatemp[i,j,k]/ejectamassdens[i,j,k]
				# ejectamassdens was also accumulated as total mass per bin so if we want density
				# we need to divide by the volume of each velocty bin
				ejectamassdens[i,j,k] = ejectamassdens[i,j,k] / ( deltav**3*texp**3 )


fout = h5py.File('ejecta.hdf5', 'w')
fout.create_dataset( 'rho', data=ejectamassdens, dtype='d' )
fout.create_dataset( 'temp', data=ejectatemp, dtype='d' )
fout.create_dataset( 'vx', data=vx, dtype='d' )
fout.create_dataset( 'vy', data=vy, dtype='d' )
fout.create_dataset( 'vz', data=vz, dtype='d' )
fout.create_dataset( 'erad', data=ejectatemp, dtype='d' )
fout.create_dataset( 'dr', data=[deltav*texp, deltav*texp, deltav*texp], dtype='d')
fout.create_dataset( 'time', data=[texp], dtype='d')

print('finished density and temperature')


# iterate through particles
dirs = numpy.arange( 1, lastparticle, dirsize ) # 1 , lastparticle, directory size


# nuclide set is same for all, so just take from one file


yfile = open('../abundance_runs/run_1/final_abundances_1.dat')

nnuc = int(  yfile.readline().split()[1] )
nucZ = numpy.zeros(nnuc)
nucA = numpy.zeros(nnuc)
i = 0
for line in yfile :
	sl = line.split()
	nucZ[i] = int(sl[1])
	nucA[i] = nucZ[i] + int(sl[2])
	i+=1
yfile.close()

ejectaabund = numpy.zeros( ( 2*vgridsize, 2*vgridsize, 2*vgridsize, nnuc ) )
weightaccum = numpy.zeros( ( 2*vgridsize, 2*vgridsize, 2*vgridsize ) )

for di in range(len(dirs)) :

	if (di == (len(dirs)-1)):
		pids = numpy.arange( dirs[di], lastparticle+1, 1)
	else :
		pids = numpy.arange( dirs[di], dirs[di]+dirsize, 1)
	for pindex in range( pids.size ):

		pid = int(pids[pindex])
		if ( leftgrid[pid-1] > 0 ) :
			print ('skipping particle that left grid ', pindex, ' at vel ', v_rad)
			continue
		# skip if E_kin + E_grav <= 0
		if ( 0.5*(fvel[pid-1][0]**2 + fvel[pid-1][1]**2 + fvel[pid-1][2]**2) + fgpot[pid-1] <= 0 ):
			continue
		
		# locate destination on grid
		# want to find cell for which particle is between center of this and next cells
		# index starts from zero for first cell
		# but may be -1 indicating particle is in lower half of cell
		velx = fvel[int(pids[pindex]-1)][0]
		vely = fvel[int(pids[pindex]-1)][1]
		velz = fvel[int(pids[pindex]-1)][2]
		#velx = fpos[int(pids[pindex])][0] / texp
		#vely = fpos[int(pids[pindex])][1] / texp
		#print 'velx=', velx, ' vely=',vely
		gridi = int( numpy.floor( ((velx+maxv)/deltav)- 0.5 ) )
		gridj = int( numpy.floor( ((vely+maxv)/deltav)- 0.5 ) )
		gridk = int( numpy.floor( ((velz+maxv)/deltav)- 0.5 ) )
		#print 'gridi gridj: ', gridi, gridj
		# weight factors for cloud in cell
		lowweighti = (velx+maxv)/deltav - 0.5 - gridi
		lowweightj = (vely+maxv)/deltav - 0.5 - gridj
		lowweightk = (velz+maxv)/deltav - 0.5 - gridk

		try: 
			# now read the track yield data
			yfile = open('../abundance_runs/run_%d/final_abundances_%d.dat' % (dirs[di],pids[pindex]), 'r' )
		except IOError:
			print ('particle ', pindex, ' not found,  pid ', int(pids[pindex]))
		else:
			py = dict()
			# first line just lists number of nuclides
			yfile.readline()
			for line in yfile :
				sl = line.split()
				py[ sl[0] ] = float(sl[3].replace('D','E'))
				py[ ( int(sl[1]), int(sl[1])+int(sl[2]) ) ] = float(sl[3].replace('D','E'))
			yfile.close()
			#print 'py[si28] = ', py['si28']
		
			# now add portion to grid cells : i.e. cloud-in-cell
	
			# for the starting cell - let's call it (0,0,0)
			if ( gridi >= 0 and gridi < 2*vgridsize and gridj >= 0 and gridj < 2*vgridsize and gridk >= 0 and gridk < 2*vgridsize):
				weightaccum[gridi,gridj,gridk] += weights[pid-1]*lowweighti*lowweightj*lowweightk
				for ni in range(nnuc) :
					ejectaabund[gridi,gridj,gridk,ni] += weights[pid-1]*lowweighti*lowweightj*lowweightk * py[ (nucZ[ni],nucA[ni]) ]
			# spillover into (1,0,0)
			if ( gridi+1 >= 0 and gridi+1 < 2*vgridsize and gridj >= 0 and gridj < 2*vgridsize and gridk >= 0 and gridk < 2*vgridsize):
				weightaccum[gridi+1,gridj,gridk] += weights[pid-1]*(1.0-lowweighti)*lowweightj*lowweightk
				for ni in range(nnuc) :
					ejectaabund[gridi+1,gridj,gridk,ni] += weights[pid-1]*(1.0-lowweighti)*lowweightj*lowweightk * py[ (nucZ[ni],nucA[ni]) ]
			# spillover into (0,1,0)
			if ( gridi >= 0 and gridi < 2*vgridsize and gridj+1 >= 0 and gridj+1 < 2*vgridsize and gridk >= 0 and gridk < 2*vgridsize):
				weightaccum[gridi,gridj+1,gridk] += weights[pid-1]*lowweighti*(1.0-lowweightj)*lowweightk
				for ni in range(nnuc) :
					ejectaabund[gridi,gridj+1,gridk,ni] += weights[pid-1]*lowweighti*(1.0-lowweightj)*lowweightk * py[ (nucZ[ni],nucA[ni]) ]
			# spillover into (0,0,1)
			if ( gridi >= 0 and gridi < 2*vgridsize and gridj >= 0 and gridj < 2*vgridsize and gridk+1 >= 0 and gridk+1 < 2*vgridsize):
				weightaccum[gridi,gridj,gridk+1] += weights[pid-1]*lowweighti*lowweightj*(1.0-lowweightk)
				for ni in range(nnuc) :
					ejectaabund[gridi,gridj,gridk+1,ni] += weights[pid-1]*lowweighti*lowweightj*(1.0-lowweightk) * py[ (nucZ[ni],nucA[ni]) ]
			# spillover into (1,1,0)
			if ( gridi+1 >= 0 and gridi+1 < 2*vgridsize and gridj+1 >= 0 and gridj+1 < 2*vgridsize and gridk >= 0 and gridk < 2*vgridsize):
				weightaccum[gridi+1,gridj+1,gridk] += weights[pid-1]*(1.0-lowweighti)*(1.0-lowweightj)*lowweightk
				for ni in range(nnuc) :
					ejectaabund[gridi+1,gridj+1,gridk,ni] += weights[pid-1]*(1.0-lowweighti)*(1.0-lowweightj)*lowweightk * py[ (nucZ[ni],nucA[ni]) ]
			# spillover into (0,1,1)
			if ( gridi >= 0 and gridi < 2*vgridsize and gridj+1 >= 0 and gridj+1 < 2*vgridsize and gridk+1 >= 0 and gridk+1 < 2*vgridsize):
				weightaccum[gridi,gridj+1,gridk+1] += weights[pid-1]*lowweighti*(1.0-lowweightj)*(1.0-lowweightk)
				for ni in range(nnuc) :
					ejectaabund[gridi,gridj+1,gridk+1,ni] += weights[pid-1]*lowweighti*(1.0-lowweightj)*(1.0-lowweightk) * py[ (nucZ[ni],nucA[ni]) ]
			# spillover into (1,0,1)
			if ( gridi+1 >= 0 and gridi+1 < 2*vgridsize and gridj >= 0 and gridj < 2*vgridsize and gridk+1 >= 0 and gridk+1 < 2*vgridsize):
				weightaccum[gridi+1,gridj,gridk+1] += weights[pid-1]*(1.0-lowweighti)*lowweightj*(1.0-lowweightk)
				for ni in range(nnuc) :
					ejectaabund[gridi+1,gridj,gridk+1,ni] += weights[pid-1]*(1.0-lowweighti)*lowweightj*(1.0-lowweightk) * py[ (nucZ[ni],nucA[ni]) ]
			# spillover into (1,1,1)
			if ( gridi+1 >= 0 and gridi+1 < 2*vgridsize and gridj+1 >= 0 and gridj+1 < 2*vgridsize and gridk+1 >= 0 and gridk+1 < 2*vgridsize):
				weightaccum[gridi+1,gridj+1,gridk+1] += weights[pid-1]*(1.0-lowweighti)*(1.0-lowweightj)*(1.0-lowweightk)
				for ni in range(nnuc) :
					ejectaabund[gridi+1,gridj+1,gridk+1,ni] += weights[pid-1]*(1.0-lowweighti)*(1.0-lowweightj)*(1.0-lowweightk) * py[ (nucZ[ni],nucA[ni]) ]
			
	print("finished dir ", dirs[di])


# now complete averaging
for i in range(vgridsize) :
	for j in range(2*vgridsize) :
		if ( weightaccum[i,j] > 0.0 ) :
			for ni in range(nnuc) :
				ejectaabund[i,j,ni] = ejectaabund[i,j,ni] / weightaccum[i,j]
		else :
			# pure he
			ejectaabund[i,j,4] = 1.0


# fill cells with missing abundances (ones that didn't have a particle in them)
# strategy: Find nearest cells with data and average them.
#           All cells in a 1-cell width band are included in average.
#           Done by incrementing out in 1-cell stages to find nearest cells.
#           Note if all 4 neighbor cells have data, this is just interpolation.
#           For cells outside main region, this leads effectively to "constant" extrapolation
#            based on furthest out data in that direction.

# extents arrays hold number of cells away from row with same j index as cell being filled
# max (cell-index) radius is corner-to-corner of 1x2*vgridsize grid
oldextents = numpy.zeros(3*vgridsize)
newextents = numpy.zeros(3*vgridsize)
for i in range(vgridsize) :
	for j in range(2*vgridsize) :
		if ( ( weightaccum[i,j] == 0.0 ) and ( ejectamassdens[i,j] > 1.01*avgdens*1e-20) ) :
			# need to fill

			# search for valid data
			oldextents[:] = -1
			newextents[:] = -1
			newextents[0] = 0
			foundcells = list()
			radius = 0
			# increment extents  and search cells at this radius
			while ( len(foundcells) < 1 ):
				# increment extents array
				oldextents[:] = newextents[:]
				radius = radius+1
				newextents[radius]=0
				for ri in range(radius):
					ei = oldextents[ri]
					while ( ri*ri + (ei+1)*(ei+1) < radius*radius + 1e-6 ):
						ei=ei+1
					newextents[ri] = ei
				# search at this radius
				for ri in range(radius+1):
					for ei in range(int(oldextents[ri]+1), int(newextents[ri]+1)):
						if (ri>0) :
							isignrange = [-1,+1]
						else :
							isignrange = [1]
						if (ei>0) :
							jsignrange = [-1,+1]
						else :
							jsignrange = [1]
						for isign in isignrange :
							for jsign in jsignrange :
								iref = int(i+isign*ri)
								jref = int(j+jsign*ei)
								# only check if inside domain
								if ( (iref>-1) and (iref<vgridsize) and (jref>-1) and (jref<2*vgridsize) ) :
									if ( weightaccum[iref,jref] > 0.0 ) :
										foundcells.append( (iref,jref) )
			# average cells found
			# had set everything without particle weight to He, undo that
			ejectaabund[i,j,4] = 0.0
			# now for every nuclide
			w = 1.0/len(foundcells)
			for ni in range(nnuc) :
				for cell in foundcells :
					ejectaabund[i,j,ni] += w*ejectaabund[cell[0],cell[1],ni]
			

#fout = h5py.File('trial_output.hdf5', 'w')
fout.create_dataset( 'Z', data=nucZ, dtype='i')
fout.create_dataset( 'A', data=nucA, dtype='i')
fout.create_dataset( 'comp', data=ejectaabund, shape=ejectaabund.shape, dtype='f' )

