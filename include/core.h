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

struct list_head {
	struct list_head *next, *prev;
};
static inline void INIT_LIST_HEAD(struct list_head *list)
{
	list->next = list;
	list->prev = list;
}

typedef enum datatype{
    DTUNKNOW = 0,
    DTCONV,
    DTBN,
    DTSHORTCUT,
    DTLINER,
} DATA_TYPE;

struct data_info{
    DATA_TYPE type;
    Data_Len len;
    char name[16];
    uint32_t dim[DIM_DEPTH];
    struct data_info *sibling;
    void *data;
};
typedef struct data_info data_info_t;

struct object{
    char name[16];
    struct object *parent;
    struct object *sibling;
    struct object *child;
    struct data_info *info;
};
typedef struct object object_t;

struct net{
    char name[32];
    struct object *start;
    void (*process)(struct data_info*);
};

#endif