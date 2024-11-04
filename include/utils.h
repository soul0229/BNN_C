#ifndef _UTILS_H
#define _UTILS_H
#include "common.h"

enum data_len{
    UNKNOW = 0,
    BINARY,
    TWO_BYTE,
    FLOAT_BYTE,
    DFLOAT_BYTE,
};
typedef enum data_len Data_Len;

enum data_type{
    KERNEL = 0,
    BATCHNORM,
    LINEAR,
    LAYER,
    NET_LIST,
};
typedef enum data_type Data_Type;

enum bn_type{
    MEAN = 0,
    VAR,
    WEIGHT,
    BIAS,
    BN_NUM,
};
typedef enum bn_type BN_Type;

struct active{
    Data_Len binary;
    uint16_t size;
    uint16_t ch;
    void *active;
};
typedef struct active Activate_TypeDef;

struct net_list{
    Data_Type type;
    struct net_list *pre;
    struct net_list *next;
    void *data;
};
typedef struct net_list Net_List;

struct BatchNorm2d{
    Data_Type type;
    void *last_stage;
    uint32_t data_site;
    uint16_t major;
    uint16_t size;
    void* data[BN_NUM];
    Net_List *list;
};
typedef struct BatchNorm2d BatchNorm;

struct conv_kernel{
    Data_Type type;
    void *last_stage;
    uint32_t data_site;
    Data_Len binary;
    uint16_t major;
    uint16_t size;
    uint16_t in_ch;
    uint16_t out_ch;
    void *kernel;
    Net_List *list;
};
typedef struct conv_kernel CONV_KernelTypeDef;

struct linear_float{
    Data_Type type;
    void *last_stage;
    uint32_t data_site;
    uint16_t major;
    uint16_t size[2];
    void* weight;
    void* bias;
    Net_List *list;
};
typedef struct linear_float Linear;

struct layer{
    Data_Type type;
    void *last_stage;
    uint16_t major;
    uint16_t minor;
    void *data;
    Net_List *list; 
};
typedef struct layer Layer;

#if (DATA_LEN == 8)
#define intx_t int8_t
#elif (DATA_LEN == 16)
#define intx_t uint16_t
#elif (DATA_LEN == 32)
#define intx_t uint32_t
#elif (DATA_LEN == 64)
#define intx_t uint64_t
#else
#error "DATA_LEN error!"
#endif

#define align16(x)    (((x)+(16-1))-(((x)+(16-1))%16))
#define PARTSIZE(type,mem) ((unsigned long)(&(((type *)0)->mem)))

void net_lists_add(Net_List *list, void *obj);
void *net_obj_create(Data_Type type);
void BConvTest();
void bitcont_generate();
void Net_Binary(Net_List *Net);
void NetStorage(Net_List* Net, uint64_t *net_cnt, uint64_t *data_cnt, FILE* w, char *file_name);
Net_List *json_model_parse(char* file_name);
Net_List *load_net(char* file_name, uint64_t next, FILE *fp, Layer *last_stage);
void print_net_data(Net_List *Net);

Activate_TypeDef *batchnorm_pcs(Activate_TypeDef *activate, BatchNorm *bn);

#endif
