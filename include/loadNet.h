#ifndef __LOADNET_H__
#define __LOADNET_H__
#include "core.h"
#include <stdint.h>


struct _data_storage{
    Tstruct type;
    uint32_t parent;
    uint32_t sibling;
    uint32_t reserve;
    char name[16];
    Data_Len len;
    uint32_t dim[DIM_DEPTH];
    uint32_t data;
};
typedef struct _data_storage data_storage;

struct _cont_storage{
    Tstruct type;
    uint32_t parent;
    uint32_t sibling;
    uint32_t child;
    char name[16];
};
typedef struct _cont_storage cont_storage;

struct _net_storage{
    Tstruct type;
    uint32_t reserve;
    uint32_t sibling;
    uint32_t child;
    char name[32]; 
};
typedef struct _net_storage net_storage;

void load_ml_net(char *file_name);

#endif