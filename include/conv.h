#ifndef __CONV_H__
#define __CONV_H__
#include "common.h"
#include "config.h"

#ifndef DATA_LEN
    #define DATA_LEN 8
#endif
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

struct active{
    Data_Len binary;
    uint16_t size;
    uint16_t ch;
    void *active;
};
typedef struct active Activate_TypeDef;

typedef struct conv_offset_t {
    int8_t x_start;
    int8_t y_start;
}conv_offset;

Activate_TypeDef *BinarizeConv2d(CONV_KernelTypeDef *kernel, Activate_TypeDef* input, uint8_t stride, uint8_t padding, bool depthwise);
Activate_TypeDef *Conv2d(CONV_KernelTypeDef *kernel, Activate_TypeDef* input, uint8_t stride, uint8_t padding, bool depthwise);

#endif