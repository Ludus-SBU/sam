/* header containing structures used by flash in datafiles */

/* (c) Dean Townsley <Dean.M.Townsley@ua.edu> 2008.  GPL */

#ifndef FLASH_STRUCTURES_H
#define FLASH_STRUCTURES_H

#define LIST_STRING_SIZE 80
#define PROP_STRING_SIZE 25

typedef struct real_list_t {
  char name[LIST_STRING_SIZE];
  double value;
} real_list_t;

#endif /* FLASH_STRUCTURES_H */
