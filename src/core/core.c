#include <stdlib.h>
#include <stdio.h>
#include <string.h>  
#include "core.h"

global_data gd = {
    .net = NULL,
    {
        NULL,
        NULL,
        NULL,
        NULL,
    },
};

int regesiter_func(){
    return 0;
}

Tstruct *node_create(Tstruct *dad, Tstruct new, uint32_t dim[DIM_DEPTH], char *name, Data_Len len){
    size_t malloc_size = 0;
    size_t arry_size;
    ARRAY_SIZE(dim, arry_size);
    dbg_print("name:%s ,arry_size:%d\n", name, arry_size);
    common_t *node = NULL;
    switch(new){
        case TLAYER:
        case TSHORTCUT:
            node = malloc(sizeof(container_t));
            break;
        case TBATCHNORM:
            malloc_size = arry_size * 2;
        case TCONV:
        case TLINER:
            malloc_size += arry_size;
            if(len == BINARY)
                malloc_size = arry_size/32;
            dbg_print("malloc_size:%ld\n", malloc_size+dim[0]);
            node = malloc(sizeof(data_info_t));
            malloc_size = (malloc_size+dim[0]) * sizeof(float);
            ((data_info_t *)node)->data = malloc(malloc_size);
            ((data_info_t *)node)->len = len?len:FLOAT_BYTE;
            memcpy(((data_info_t *)node)->dim, dim, sizeof(uint32_t)*DIM_DEPTH);
            break;
        default:printf("Tstruct error!\n");return NULL;
    }

    if(!node){
        dbg_print("node create malloc error!\n");
        return NULL;
    }
    dbg_print("malloc ok. size:%ld\n", malloc_size);

    /* new type have parent */
    if(dad){
        node->parent = dad;
        node->sibling = ((common_t *)dad)->child;
        ((common_t *)dad)->child = (Tstruct *)node;
        node->type = new;
    }
    /* new type don't have parent */
    else {
        node->type = new;
        node->parent = (Tstruct*)Net;
        node->sibling = Net->child;
        Net->child = (Tstruct*)node;
    }

    strcpy(node->name, name);
    return (Tstruct*)node;
}

/* Refer to the linux kernel device tree parsing */
void reverse_nodes(common_t *parent)
{
	common_t *child, *next;

	/* In-depth first */
	child = (common_t *)parent->child;
	while (child) {
		reverse_nodes(child);

		child = (common_t *)child->sibling;
	}

	/* Reverse the nodes in the child list */
	child = (common_t *)parent->child;
	parent->child = NULL;
	while (child) {
		next = (common_t *)child->sibling;

		child->sibling = parent->child;
		parent->child = (Tstruct *)child;
		child = next;
	}
}
