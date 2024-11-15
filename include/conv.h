#ifndef __CONV_H__
#define __CONV_H__
#include "common.h"
#include "config.h"
#include "core.h"

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

typedef struct conv_offset_t {
    int8_t x_start;
    int8_t y_start;
}conv_offset;

data_info_t *BinarizeConv2d(data_info_t *kernel, data_info_t* input, uint8_t stride, uint8_t padding, bool depthwise);
data_info_t *Conv2d(data_info_t *kernel, data_info_t* input, uint8_t stride, uint8_t padding, bool depthwise);

#endif