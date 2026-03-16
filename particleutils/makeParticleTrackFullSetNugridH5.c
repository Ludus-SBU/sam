/* (c) Dean Townsley <Dean.M.Townsley@ua.edu> 2016.  GPL */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hdf5.h"
#include "hdf5_hl.h"
#include "ecodes.h"
#include "flash_structures.h"
#include "particleutil.h"

int main( int argc, char **argv)
{
    int lowfi, hifi, fileindex, len;
    char filename[4096];
    char *firstpartfilename, *outputfilename;

    hid_t pfile_id, outfile_id;
    hid_t input_dataset, input_dataspace;
    int nprop, numtracks, numtimes;
    unsigned int unumtracks;
    double time;
    char *prop_names;
    double *trackdata_buf, *trackdata_onetime_buf, *part_props, *positions, *velocities, *weights;
    int *trackids;
    unsigned int *tracklengths, *leftgrid;
    long long int *trackstarts;
    hsize_t input_dataspace_dims[10], input_max_dims[10];
    int npart_thisfile, fullset, geometry;
    int i, tagi, minci, tempi, densi, velxi, velyi, velzi, posxi, posyi, poszi;
    int ti, pi, timechunksize, timechunk, outputsize;

    herr_t herr;
    hsize_t part_props_mem_size[2];
    hid_t part_props_memspace;
    hsize_t part_props_slice_start[2], part_props_slice_count[2];

    hsize_t pertrackdata_size[1];
    hid_t pertrackdata_dspace_id, trackids_id, tracklengths_id, trackstarts_id, leftgrid_id, weights_id;
    hsize_t posveldata_size[2];
    hid_t posveldata_dspace_id, initialpositions_id, finalpositions_id, finalvelocities_id;
    hsize_t trackdata_size[2];
    hid_t trackdata_dspace_id, trackdata_id;
    hsize_t trackdata_mem_size[2], trackdata_start[2], trackdata_stride[2], trackdata_count[2], trackdata_block[2];
    hsize_t trackdata_mem_start[2], trackdata_mem_stride[2];
    hid_t trackdata_mspace_id; 

    if ( argc < 5 ) {
        fprintf(stderr,"Utility for creating full particle set from flash simulation for nugrid post-processing\n");
        fprintf(stderr," Usage: %s first_particle_filename_0000 startnumber endnumber outputfilename.hdf5\n", argv[0]);
        exit(1);
    }
    
    firstpartfilename = argv[1];
    lowfi = atoi(argv[2]);
    hifi = atoi(argv[3]);
    outputfilename = argv[4];

    /* get metadata (num props, prop names) from first file */
    pfile_id = H5Fopen(argv[1], H5F_ACC_RDONLY, H5P_DEFAULT);
    if (pfile_id < 0) {
        fprintf(stderr, "error opening first file, %s, for metadata", argv[1]);
        exit(1);
    }
    getParticleMetadata(pfile_id, &numtracks, &nprop, &tagi, &prop_names);
    H5Fclose(pfile_id);

    fprintf(stderr, "Processing %i particle tracks\n", numtracks );

    /* find index of temperature, density, velx, vely, velz, posx, posy, posz */
    minci=-1;
    tempi=-1; densi=-1;
    velxi=-1; velyi=-1; velzi=-1;
    posxi=-1; posyi=-1; poszi=-1;
    i=0;
    while ( i < nprop && ( tempi<0 || minci<0 || densi<0 || velxi<0 || velyi<0 || velzi<0 || posxi<0 || posyi<0 || poszi<0 ) ) {
        if ( minci<0 && strncmp( (prop_names+PROP_STRING_SIZE*i), "minc", 4) == 0 ) minci = i;
        if ( tempi<0 && strncmp( (prop_names+PROP_STRING_SIZE*i), "temp", 4) == 0 ) tempi = i;
        if ( densi<0 && strncmp( (prop_names+PROP_STRING_SIZE*i), "dens", 4) == 0 ) densi = i;
        if ( velxi<0 && strncmp( (prop_names+PROP_STRING_SIZE*i), "velx", 4) == 0 ) velxi = i;
        if ( velyi<0 && strncmp( (prop_names+PROP_STRING_SIZE*i), "vely", 4) == 0 ) velyi = i;
        if ( velzi<0 && strncmp( (prop_names+PROP_STRING_SIZE*i), "velz", 4) == 0 ) velzi = i;
        if ( posxi<0 && strncmp( (prop_names+PROP_STRING_SIZE*i), "posx", 4) == 0 ) posxi = i;
        if ( posyi<0 && strncmp( (prop_names+PROP_STRING_SIZE*i), "posy", 4) == 0 ) posyi = i;
        if ( poszi<0 && strncmp( (prop_names+PROP_STRING_SIZE*i), "posz", 4) == 0 ) poszi = i;
        i++;
    }
    /* we can continue without minc (weights are optional) but not without any others */
    if ( i==nprop && ( tempi<0 || densi<0 || velxi<0 || velyi<0 || velzi<0 || posxi<0 || posyi<0 || poszi<0 ) ) {
        fprintf(stderr, "Did not find all property indices in particle file\n");
        exit(2);
    }

    /* allocations and initializations */
    len = strlen(firstpartfilename);
    numtimes = hifi-lowfi+1;
    part_props = malloc( sizeof(double) * nprop * numtracks );
    part_props_mem_size[0] = numtracks;
    part_props_mem_size[1] = nprop;
    part_props_memspace = H5Screate_simple( 2, part_props_mem_size, NULL );

    /* use a 800MB  buffer */
    timechunksize = 800000000/sizeof(double)/numtracks/3;
    fprintf(stderr,"Will process in timestep chunks of %i\n", timechunksize );
    trackdata_onetime_buf = malloc( sizeof(double) * numtracks * 3 );
    trackdata_buf = malloc( sizeof(double) * timechunksize * numtracks * 3 );
    positions = malloc( sizeof(double) * numtracks * 3 );
    velocities = malloc( sizeof(double) * numtracks * 3 );

    if ( minci >= 0 ) weights = malloc( sizeof(double) * numtracks );
    trackids = malloc( sizeof(int) * numtracks );
    tracklengths = malloc( sizeof(unsigned int) * numtracks );
    trackstarts = malloc( sizeof(long long int) * numtracks );
    leftgrid = malloc( sizeof(unsigned int) * numtracks );

    /* we are doing equal length tracks */
    for (i=0;i<numtracks;i++) {
        tracklengths[i] = numtimes;
        trackstarts[i] = ((long long unsigned int) numtimes )*i;
    }

    /* open output file */
    outfile_id = H5Fcreate(outputfilename, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
    if (outfile_id <0) {
        fprintf(stderr, "error opening output file %s, may already exist\n", outputfilename);
        exit(1);
    }

    /* set general dataset attributes */
    H5LTset_attribute_string( outfile_id, "/", "format", "flashdist-tc-nugrid-snia-trackset-1.0" );
    unumtracks = (unsigned int) numtracks; /* cast because getParticleMetaData used int */
    H5LTset_attribute_uint( outfile_id, "/", "numtracks", &unumtracks, 1);
    fullset = 1;
    H5LTset_attribute_int( outfile_id, "/", "fullset", &fullset, 1);
    /* geometry is currently hardcoded as 3D */
    geometry = 3;
    H5LTset_attribute_int( outfile_id, "/", "geometry", &geometry, 1);

    /* create various datastructures that will be written to in main loop */

    pertrackdata_size[0] = numtracks;
    pertrackdata_dspace_id = H5Screate_simple( 1, pertrackdata_size, NULL );
    trackids_id = H5Dcreate2( outfile_id, "trackids", H5T_NATIVE_INT32, pertrackdata_dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    tracklengths_id = H5Dcreate2( outfile_id, "tracklengths", H5T_NATIVE_UINT32, pertrackdata_dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    trackstarts_id = H5Dcreate2( outfile_id, "trackstarts", H5T_NATIVE_UINT64, pertrackdata_dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    leftgrid_id = H5Dcreate2( outfile_id, "leftgrid", H5T_NATIVE_UINT32, pertrackdata_dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    if ( minci >= 0 ) weights_id = H5Dcreate2( outfile_id, "weights", H5T_NATIVE_DOUBLE, pertrackdata_dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );

    /* actually can go ahead and write these */
    H5Dwrite( tracklengths_id, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, tracklengths );
    H5Dclose( tracklengths_id );
    H5Dwrite( trackstarts_id, H5T_NATIVE_ULLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, trackstarts );
    H5Dclose( trackstarts_id );

    posveldata_size[0] = numtracks;
    posveldata_size[1] = 3;  /* hard code 3d for now */
    posveldata_dspace_id = H5Screate_simple( 2, posveldata_size, NULL );
    initialpositions_id = H5Dcreate2( outfile_id, "initialpositions", H5T_NATIVE_DOUBLE, posveldata_dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    finalpositions_id = H5Dcreate2( outfile_id, "finalpositions", H5T_NATIVE_DOUBLE, posveldata_dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    finalvelocities_id = H5Dcreate2( outfile_id, "finalvelocities", H5T_NATIVE_DOUBLE, posveldata_dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );

    trackdata_size[0] = numtracks * numtimes;
    trackdata_size[1] = 3;
    trackdata_dspace_id = H5Screate_simple( 2, trackdata_size, NULL );
    trackdata_id = H5Dcreate2( outfile_id, "trackdata", H5T_NATIVE_DOUBLE, trackdata_dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );

    /* go ahead and set up trackdata memory buffer properties once */
    trackdata_mem_size[0] = numtracks*timechunksize;
    trackdata_mem_size[1] = 3;
    trackdata_mspace_id = H5Screate_simple( 3, trackdata_mem_size, NULL );

    timechunk = 0;
    /* now start main loop over timesteps/dumpfiles */
    for (fileindex = lowfi; fileindex <= hifi; fileindex++ ) {
         /* construct current particle filename */
         strncpy( filename, firstpartfilename, len-4);
         sprintf( filename+len-4, "%04i", fileindex );

         pfile_id = H5Fopen( filename, H5F_ACC_RDONLY, H5P_DEFAULT );
         if (pfile_id <0) {
             fprintf(stderr, "error opening file %s\n", filename);
             exit(1);
         }

         time = getTime(pfile_id);

         /* now open and read particle properties data for all particles in file */
         input_dataset = H5Dopen( pfile_id, "tracer particles" );
         if ( input_dataset < 0 ) {  fprintf(stderr, "Error opening particle dataset\n"); exit(1); }

         input_dataspace = H5Dget_space( input_dataset );
         H5Sget_simple_extent_dims( input_dataspace, input_dataspace_dims, input_max_dims );
         npart_thisfile = input_dataspace_dims[0];

         /* configure memory destination for read, only up to npart_thisfile */
         part_props_slice_start[0] = 0;
         part_props_slice_start[1] = 0;
         part_props_slice_count[0] = npart_thisfile;
         part_props_slice_count[1] = nprop;
         H5Sselect_hyperslab( part_props_memspace, H5S_SELECT_SET,
                               part_props_slice_start, NULL, part_props_slice_count, NULL );

         /* now read */
         fprintf(stderr, "reading data from file %s\n", filename );
         herr = H5Dread( input_dataset, H5T_NATIVE_DOUBLE, part_props_memspace, H5S_ALL, H5P_DEFAULT, part_props );
         if ( herr < 0 ) { fprintf(stderr,"Error reading particle data\n"); exit(1); }
         H5Dclose(input_dataset);
         H5Fclose(pfile_id);
         fprintf(stderr,"Read %i particles from file\n",npart_thisfile);

         
         /* 
          * now start stuffing data into output arrays
          * iterate through input, using particle ID as index
          * since this is a full set these should go from 1 to number of particles
          *
          * The partdata structure will be filled with all particle data the first timestep.
          * After that, any particles that are no longer there will just stay at their final values.
         */

         /* set time for all particles, even those not in current file */
         for (pi=0;pi<numtracks;pi++) {
             trackdata_onetime_buf[ pi*3 ] = time;
         }
         /* update single-time buffer data for partiles in current file */
         for ( i=0; i<npart_thisfile; i++ ) {
             int partid;

             partid = part_props[ i*nprop + tagi ];
             if ( partid > numtracks ) {
                 fprintf( stderr, "error: highest particle ID is higher than number of particles.\nNot starting at first file?");
                 exit(1);
             }
             pi = partid-1;
             trackids[pi] = partid;
             trackdata_onetime_buf[ pi*3 + 1 ] = part_props[ i*nprop + densi ];
             trackdata_onetime_buf[ pi*3 + 2 ] = part_props[ i*nprop + tempi ];
             /* for leftgrid, only gets set if particle is still there.
              * so for a particle dropping off the grid, the last update will be the correct leftgrid[] value.
              * will go back at the end and reset to zero (didn't leave) for particles still present at end */
             leftgrid[ pi ] = fileindex - lowfi + 1;
             /* need to similarly accumulate position and velocity, since we don't know when a particle might leave grid
              * (in which case it would just not appear in the rest of the particle dump file) */
             positions[ pi*2 ] = part_props[ i*nprop + posxi ];
             positions[ pi*2 + 1 ] = part_props[ i*nprop + posyi ];
             positions[ pi*2 + 1 ] = part_props[ i*nprop + poszi ];
             velocities[ pi*2 ] = part_props[ i*nprop + velxi ];
             velocities[ pi*2 + 1 ] = part_props[ i*nprop + velyi ];
             velocities[ pi*2 + 1 ] = part_props[ i*nprop + velzi ];

             if ( minci >= 0 ) weights[ pi ] = part_props[ i*nprop + minci ];
             
         }

         /* now transfer into timechunked buffer - don't go direct so that we can preserve values for particles leaving grid */
         ti = (fileindex-lowfi)%timechunksize;
         for (pi=0;pi<numtracks;pi++) {
             trackdata_buf[ pi*timechunksize*3 + ti*3 ] = trackdata_onetime_buf[pi*3];
             trackdata_buf[ pi*timechunksize*3 + ti*3 + 1 ] = trackdata_onetime_buf[pi*3+1];
             trackdata_buf[ pi*timechunksize*3 + ti*3 + 2 ] = trackdata_onetime_buf[pi*3+2];
         }

         /* now write out data */
         /* if first file write ids, weights, and initial positions */
         if (fileindex == lowfi) {
             H5Dwrite( trackids_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, trackids );
             H5Dclose( trackids_id );
             if ( minci >= 0 ) {
                 H5Dwrite( weights_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, weights );
                 H5Dclose( weights_id );
             }
             H5Dwrite( initialpositions_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, positions );
             H5Dclose( initialpositions_id );
         }

         /* only write every timechunksize */
         if ( ((fileindex-lowfi+1)%timechunksize)==0 || fileindex == hifi ) {
             if (fileindex == hifi) {
                 outputsize = numtimes - timechunk*timechunksize;
             } else {
                 outputsize = timechunksize;
             }
             fprintf(stderr,"writing %i timesteps for all tracks\n", outputsize );

             /* now write chunk of timesteps for each track */
             /* count and block size are same for memory and file */
             trackdata_count[0] = numtracks;
             trackdata_count[1] = 1;
             trackdata_block[0] = outputsize;
             trackdata_block[1] = 3;

             trackdata_mem_start[0] = 0;
             trackdata_mem_start[1] = 0;
             trackdata_mem_stride[0] = timechunksize;
             trackdata_mem_stride[1] = 3;
             H5Sselect_hyperslab( trackdata_mspace_id, H5S_SELECT_SET, trackdata_mem_start, trackdata_mem_stride, trackdata_count, trackdata_block );

             trackdata_start[0] = timechunk*timechunksize;
             trackdata_start[1] = 0;
             trackdata_stride[0] = numtimes;
             trackdata_stride[1] = 3;
             H5Sselect_hyperslab( trackdata_dspace_id, H5S_SELECT_SET, trackdata_start, trackdata_stride, trackdata_count, trackdata_block );

             herr = H5Dwrite( trackdata_id, H5T_NATIVE_DOUBLE, trackdata_mspace_id, trackdata_dspace_id, H5P_DEFAULT, trackdata_buf );

             timechunk++;
        }
         
    } /* finish loop over input particle dump files */

    H5Dclose( trackdata_id );

    /* clean up leftgrid data -- if particle was still there at the end, then set to zero to indicate it didn't leave grid */
    for (i=0;i<numtracks;i++) {
        if (leftgrid[i]==numtimes) leftgrid[i]=0;
    }
    H5Dwrite( leftgrid_id, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, leftgrid );
    H5Dclose( leftgrid_id );

    /* now write final data: positions and velocities*/
    H5Dwrite( finalpositions_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, positions );
    H5Dclose( finalpositions_id );
    H5Dwrite( finalvelocities_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, velocities );
    H5Dclose( finalvelocities_id );

    H5Fclose( outfile_id );

    return 0;

}
