#ifndef __CORE_H__
#define __CORE_H__

#include <stdint.h> 
#include "common.h"
#include "config.h"

#define BNN_MAGIC "eg.BNN"

typedef struct header {
    char magic[8];          
    uint32_t version;       
    uint32_t node_offset;   
    uint32_t data_offset; 
    char compatible[32];   
    char arguments[12];
} header_t;

typedef enum Type_Struct{
    TERROR=0,   /* error type. */

    NET_ROOT,   /* The root of net. */

    TLAYER,     /* container type. */
    TSHORTCUT,

    TCONV,      /* data type. */
    TBACHNORM,
    TLINER,
    TMAX,
} Tstruct;

struct data_info{
    Tstruct type;
    Tstruct *parent;
    Tstruct *reserve[2];
    char name[16];
    Data_Len len;
    uint32_t dim[DIM_DEPTH];
    void *data;
};
typedef struct data_info data_info_t;

struct container{
    Tstruct type;
    Tstruct *parent;
    Tstruct *sibling;
    Tstruct *child;
    char name[16];
};
typedef struct container container_t;

struct net{
    Tstruct type;
    Tstruct *reserve[2];
    Tstruct *child;
    char name[32]; 
    void (*process)(struct data_info*);
};

struct common{
    Tstruct type;
    Tstruct *parent;
    Tstruct *sibling;
    Tstruct *child;
    char name[16];
};
typedef struct common common_t;

#endif