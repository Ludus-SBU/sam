/* (c) Dean Townsley <Dean.M.Townsley@ua.edu> 2008, 2012.  GPL */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hdf5.h"
#include "ecodes.h"
#include "flash_structures.h"
#include "particleutil.h"

struct mem_file_index_pair {
    int mi; // index in memory array
    int fi; // index in file
};

int compare_file_index( const void * vp1, const void * vp2) {
    struct mem_file_index_pair * p1 = (struct mem_file_index_pair *)vp1;
    struct mem_file_index_pair * p2 = (struct mem_file_index_pair *)vp2;
    if ( p1->fi <  p2->fi ) return -1;
    if ( p1->fi == p2->fi ) return 0;
    return 1;
}

int getParticleProps(char *filename, int *indices, int npart, int nprop, double *part_props, double *time)
{

    hid_t   file_id, dataspace, dataset, memspace;
    herr_t  status;
    hsize_t  maximum_dims[10];
    hsize_t  dataspace_dims[10];

    int rank, i;

    hsize_t   data_slice_start[2], data_slice_count[2], data_block_size[2];
    hsize_t   mem_size[2];

    struct mem_file_index_pair * index_map;
    double * buf;
    int toi;

    /* open the file */
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "error opening file %s\n", filename);
        exit(1);
    }

    if (time!=NULL) *time = getTime(file_id);

    /* open dataset and find number of particles */
    dataset = H5Dopen(file_id, "tracer particles", H5P_DEFAULT );
    if(dataset < 0){
        fprintf(stderr, "Error Opening Dataset!\n");
        return DATASET_OPEN_FAIL;
    }

    /* check that nprop matches */
    dataspace = H5Dget_space(dataset);
    H5Sget_simple_extent_dims(dataspace,dataspace_dims, maximum_dims);
    if (nprop != dataspace_dims[1]) {
        fprintf(stderr,"nprops not consistent with file\n"); exit(1); }

    /* construct map between requested memory position and file indices */
    index_map = malloc(sizeof(struct mem_file_index_pair)*npart);
    for (i=0; i<npart; i++) {
        index_map[i].mi = i;
        index_map[i].fi = indices[i];
    }
    /* sort map to be in file-index order */
    qsort(index_map, npart, sizeof(struct mem_file_index_pair), compare_file_index);

    /* describe memory */
    rank = 2;
    mem_size[0] = npart;
    mem_size[1] = nprop;
    memspace = H5Screate_simple(rank, mem_size, NULL);

    /* clear selection */
    H5Sselect_none(dataspace);
    /* select regions which correspond to data for the requested particles */
    for ( i=0; i<npart; i++ ) {
        /* set up a block which only selects one particle */
        data_slice_start[0] = (hsize_t) indices[index_map[i].mi];
        data_slice_start[1] = (hsize_t) 0;

        data_slice_count[0] = 1;
        data_slice_count[1] = 1;

        data_block_size[0] = 1;
        data_block_size[1] = nprop;

        status = H5Sselect_hyperslab(dataspace, H5S_SELECT_OR,
                    data_slice_start, NULL, data_slice_count, data_block_size);
        if (status < 0){
            fprintf(stderr,"Error Selecting Hyperslab!\n");
            return HYPERSLAB_SELECT_FAIL;
        }
    }

    /* read in the data */
    status = H5Dread(dataset, H5T_NATIVE_DOUBLE, memspace, dataspace,
                     H5P_DEFAULT, part_props);
    if (status < 0){
        printf("Error Reading In Particle Data!");
        return DATA_READ_ERROR;
    }

    /* now re-order data to order as requested */
    buf = malloc(sizeof(double)*nprop);
    for (i=0; i<npart; i++) {
        while ( i != (toi=index_map[i].mi) ) {
            /* swap this entry to where it belongs */
            memcpy(buf, part_props+toi*nprop, sizeof(double)*nprop);
            memcpy(part_props+toi*nprop, part_props+i*nprop, sizeof(double)*nprop);
            memcpy(part_props+i*nprop, buf, sizeof(double)*nprop);
            /* update map to still be ordered same as memory buffer */
            /* !! note this destroys the file index info !! */
            index_map[i].mi = index_map[toi].mi;
            index_map[toi].mi = toi;
        }
    }
    free(buf);

    free(index_map);

    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);
    H5Fclose(file_id);

    return NORMAL_STATUS;
}

