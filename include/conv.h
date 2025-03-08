#ifndef __CONV_H__
#define __CONV_H__
#include "common.h"
#include "config.h"
#include "core.h"

typedef struct conv_offset_t {
    int8_t x_start;
    int8_t y_start;
}conv_offset;

data_info_t *BinarizeConv2d(data_info_t *kernel, data_info_t* input, uint8_t stride, uint8_t padding);
data_info_t *Conv2d(data_info_t *kernel, data_info_t* input, uint8_t stride, uint8_t padding);

#endif