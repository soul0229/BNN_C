#include <stdio.h>
#include <stdlib.h>
#include "conv.h"

extern const conv_offset offset[];

data_info_t *Conv2d(data_info_t *kernel, data_info_t* input, uint8_t stride, uint8_t padding, bool depthwise){
    data_info_t *output = malloc(sizeof(data_info_t));
    output->dim[3] = (input->dim[3]+padding*2-kernel->dim[3])/stride+1;
    output->dim[1] = kernel->dim[0];
    output->dim[0] = 1;
    output->data = calloc(output->dim[2]*output->dim[3]*output->dim[1], sizeof(float)); 
    output->len = FLOAT_BYTE;

    if(depthwise){
        if(kernel->dim[1] != input->dim[1]){
            free(output->data); 
            free(output); 
            return NULL;
        }
        for (uint16_t out_ch = 0; out_ch < kernel->dim[0]; ++out_ch) {//kernel->out_ch:输出通道数
            for (uint16_t in_ch = 0; in_ch < input->dim[1]; ++in_ch) {
                for (uint16_t kernel_pos = 0; kernel_pos < (kernel->dim[2] * kernel->dim[3]); ++kernel_pos) {//卷积核元素9选1
                    for (uint16_t x_pos = 0; x_pos < input->dim[2]; x_pos += stride) {
                        for (uint16_t y_pos = 0; y_pos < input->dim[3]; y_pos += stride) {
                            int16_t x_input = x_pos + (offset[kernel_pos].x_start * padding);
                            int16_t y_input = y_pos + (offset[kernel_pos].y_start * padding);
                            if (x_input >= 0 && x_input < input->dim[2] && y_input >= 0 && y_input < input->dim[3]) {//判断是否与0同或
                                ((float*)(output->data))[(out_ch * output->dim[2]  + x_pos/stride) * output->dim[3] + y_pos/stride]+= \
                                    ((float*)(input->data))[(in_ch * input->dim[2] + x_input) * input->dim[3] + y_input] * \
                                    ((float*)(kernel->data))[(out_ch*kernel->dim[1] + in_ch) * kernel->dim[2] * kernel->dim[3] + kernel_pos];
                            }
                        } 
                    }
                }
            }
        }
    }
    else{
        if(kernel->dim[1] != 1){
            free(output->data); 
            free(output); 
            return NULL;
        }
        for (uint16_t out_ch = 0; out_ch < kernel->dim[0]; ++out_ch) {//kernel->out_ch:输出通道数
            for (uint16_t in_ch = 0; in_ch < input->dim[1]; ++in_ch) {
                for (uint16_t kernel_pos = 0; kernel_pos < (kernel->dim[2] * kernel->dim[3]); ++kernel_pos) {//卷积核元素9选1
                    for (uint16_t x_pos = 0; x_pos < input->dim[2]; x_pos += stride) {
                        for (uint16_t y_pos = 0; y_pos < input->dim[3]; y_pos += stride) {
                            int16_t x_input = x_pos + (offset[kernel_pos].x_start * padding);
                            int16_t y_input = y_pos + (offset[kernel_pos].y_start * padding);
                            if (x_input >= 0 && x_input < input->dim[2] && y_input >= 0 && y_input < input->dim[3]) {//判断是否与0同或
                                ((float*)(output->data))[(out_ch * output->dim[2] + x_pos/stride) * output->dim[3] + y_pos/stride]+= \
                                    ((float*)(input->data))[(in_ch*output->dim[2] + x_input)*output->dim[3] + y_input] * \
                                    ((float*)(kernel->data))[out_ch * kernel->dim[2] * kernel->dim[3] + kernel_pos];
                            }
                        } 
                    }
                }
            }
        }
    }
    return output;
}