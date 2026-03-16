/* (c) Dean Townsley <Dean.M.Townsley@ua.edu> 2008, 2016. GPL */
#ifndef PARTICLEUTIL_H
#define PARTICLEUTIL_H

#include "flash_structures.h"


int getParticleProps(char *filename, int *indices, int npart, int nprop, double *part_props, double *time);

int getParticleMetadata(hid_t file_identifier, int *npart, int *nprop, int *tagIndex, char **prop_names );

double getTime(hid_t file_id);


#endif
