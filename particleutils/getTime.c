/* (c) Dean Townsley <Dean.M.Townsley@ua.edu> 2008, 2012, 2016.  GPL */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hdf5.h"
#include "ecodes.h"
#include "flash_structures.h"

double getTime(hid_t file_id) {
    hid_t dataset, dataspace;
    hid_t real_list_type, string_type;
    hsize_t num, maxnum;
    herr_t status;

    real_list_t *real_scalars;

    int i;
    double time;
    

    /* open the scalars dataset which contains the time */
    dataset = H5Dopen(file_id, "real scalars", H5P_DEFAULT );

    /* read extent of dataset (i.e. # of name/value pairs)  */
    dataspace = H5Dget_space(dataset);
    H5Sget_simple_extent_dims(dataspace, &num, &maxnum);
    H5Sclose(dataspace);

    /* malloc a pointer to a list of real_list_t's */
    real_scalars = (real_list_t *) malloc( num * sizeof(real_list_t));

    /* describe the real_list_t structure for hdf5 */
    /* create an empty vessel sized to hold one real_list_t's worth of data */
    real_list_type = H5Tcreate(H5T_COMPOUND, sizeof(real_list_t));
    /* subdivide the empty vessel into its component sections (name and value) */
    string_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(string_type, LIST_STRING_SIZE);
    H5Tinsert(real_list_type, "name",  HOFFSET(real_list_t, name),  string_type);
    H5Tinsert(real_list_type, "value", HOFFSET(real_list_t, value), H5T_NATIVE_DOUBLE);
    /* wait to close string_type, since I'm unsure if it's still open */

    /* read the data into memory */
    status = H5Dread(dataset, real_list_type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     real_scalars);
    if (status < 0) {
      fprintf(stderr,"Error reading time from data file\n");
      exit(1);
    }

    H5Tclose(real_list_type);
    H5Tclose(string_type);
    H5Dclose(dataset);

    /* search through for the time element */
    i=0;
    while ( i<num && strncmp(real_scalars[i].name,"time", 4)!=0) i++;
    if (i >=num) {
        fprintf(stderr,"did not find time\n"); exit(1); }
    time = real_scalars[i].value;


    free(real_scalars);

    return time;

}

