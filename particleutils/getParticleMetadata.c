/* (c) Dean Townsley <Dean.M.Townsley@ua.edu> 2008.  GPL */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hdf5.h"
#include "ecodes.h"
#include "particleutil.h"

int getParticleMetadata(hid_t file_identifier, int *npart, int *nprops_ret,
                         int *tagIndex, char **prop_names ){

    hid_t dataspace, dataset, memspace, string_type;
    herr_t status;

    int rank, i;

    hsize_t dimens_2d[2];
    

    hsize_t  maximum_dims[10];
    hsize_t  dataspace_dims[10];

    int nprops;
    char *part_names;

    /*find the number of particles and number of properties */
    dataset = H5Dopen(file_identifier, "tracer particles", H5P_DEFAULT );
    if(dataset < 0) return DATASET_OPEN_FAIL;

    dataspace = H5Dget_space(dataset);
    if(dataspace < 0) return DATASPACE_SELECT_FAIL;

    /* first dimension is number of particles second is number of props */
    H5Sget_simple_extent_dims(dataspace,dataspace_dims, maximum_dims);
    if (npart!=NULL) *npart = dataspace_dims[0];
    nprops = dataspace_dims[1];
    if (nprops_ret!=NULL) *nprops_ret=nprops;
 
    H5Sclose(dataspace);
    H5Dclose(dataset);

    /* if we don't want property names or which property is tag,
       can just return now. */
    if ( tagIndex==NULL && prop_names==NULL) return NORMAL_STATUS;

    /* else continue.... */

    /* retrieve particle properties names */

    dataset = H5Dopen(file_identifier, "particle names", H5P_DEFAULT );
    if(dataset < 0){
        return DATASET_OPEN_FAIL;
    }
    
    /* allocate and configure our destination memory space */
    rank = 2;
    dimens_2d[0] = nprops;
    dimens_2d[1] = 1;
    
    part_names = (char *)malloc(sizeof(char)*PROP_STRING_SIZE* nprops);

    /*must manually set the string size for now*/
    string_type = H5Tcopy(H5T_C_S1);
    status = H5Tset_size(string_type, PROP_STRING_SIZE);

    memspace = H5Screate_simple(rank, dimens_2d, NULL);
    if(memspace < 0){
	fprintf(stderr, "Memspace Creation for particle names failed!\n");
	return MEMSPACE_SELECT_FAIL;
    }
    
    status = H5Dread(dataset, string_type, memspace, H5S_ALL, H5P_DEFAULT,
                     part_names);
    if(status < 0){
      fprintf(stderr, "Error Reading particle name data!");
      return DATA_READ_ERROR;
    }

    /* find which property is tag */
    if (tagIndex!=NULL) {
        for(i = 0; i < nprops; i++){
            if(strncmp((part_names + PROP_STRING_SIZE*i), "tag", 3)==0){
                *tagIndex = i;
                break;
            }
        }
    }

    if (prop_names != NULL) *prop_names = part_names;
    else free(part_names);
 
    H5Sclose(memspace);
    H5Dclose(dataset);

    return NORMAL_STATUS;
}
