#ifndef __CORE_H__
#define __CORE_H__

#include <stdint.h> 
#include "common.h"
#include "config.h"

#define BNN_MAGIC "eg.BNN"
#define WEIGHT "weight"
#define BIAS "bias"
#define ALPHA "alpha"
#define MEAN "running_mean"
#define VAR "running_var"

typedef enum data_order{
    OWEIGHT = 0,
    OA = 0,
    OBIAS,
    OB = 1,
    OMEAN,
    OVAR,
    OMAX,
}data_order_t;

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
    TBATCHNORM,
    TLINER,
    
    TMAX,
    BASICBLOCK,
    ACTIVATE,
} Tstruct;

struct data_info{
    Tstruct type;
    Tstruct *parent;
    Tstruct *sibling;
    Tstruct *reserve;
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
    Tstruct *reserve;
    Tstruct *sibling;
    Tstruct *child;
    char name[32]; 
    void (*process)(struct data_info*);
};
typedef struct net net_t;

struct common{
    Tstruct type;
    Tstruct *parent;
    Tstruct *sibling;
    Tstruct *child;
    char name[16];
};
typedef struct common common_t;

struct opts{
    data_info_t (*data_binary)(data_info_t *);
    data_info_t (*net_inference)(common_t *, data_info_t *);
    data_info_t (*BinarizeConv2d)(data_info_t *, data_info_t *, uint8_t, uint8_t, bool);
    data_info_t (*Conv2d)(data_info_t *, data_info_t *, uint8_t, uint8_t, bool);
};

typedef struct _global_data{
    net_t *net;
    struct opts *ops;
} global_data;

extern global_data gd;
#define Net (gd.net)

Tstruct *node_create(Tstruct *dad, Tstruct new, uint32_t dim[DIM_DEPTH], char *name, Data_Len len);
void reverse_nodes(common_t *parent);

#endif