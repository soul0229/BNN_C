#ifndef _CONV_H
#define _CONV_H
#include "utils.h"

typedef struct conv_offset_t {
    int8_t x_start;
    int8_t y_start;
}conv_offset;

Activate_TypeDef *BinarizeConv2d(CONV_KernelTypeDef *kernel, Activate_TypeDef* input, uint8_t stride, uint8_t padding, bool depthwise);
Activate_TypeDef *Conv2d(CONV_KernelTypeDef *kernel, Activate_TypeDef* input, uint8_t stride, uint8_t padding, bool depthwise);

#endif