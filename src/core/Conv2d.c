#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "conv.h"

static const conv_offset offset[3][3] = {
    {{-1, -1,},{-1, 0,},{-1, 1,},},
    {{0, -1,}, {0, 0,}, {0, 1,},},
    {{1, -1,}, {1, 0,}, {1, 1,},},
};

static const int8_t bit_cont[256] = {
    -8, -6, -6, -4, -6, -4, -4, -2, -6, -4, -4, -2, -4, -2, -2, 0 , 
    -6, -4, -4, -2, -4, -2, -2, 0 , -4, -2, -2, 0 , -2, 0 , 0 , 2 , 
    -6, -4, -4, -2, -4, -2, -2, 0 , -4, -2, -2, 0 , -2, 0 , 0 , 2 , 
    -4, -2, -2, 0 , -2, 0 , 0 , 2 , -2, 0 , 0 , 2 , 0 , 2 , 2 , 4 , 
    -6, -4, -4, -2, -4, -2, -2, 0 , -4, -2, -2, 0 , -2, 0 , 0 , 2 , 
    -4, -2, -2, 0 , -2, 0 , 0 , 2 , -2, 0 , 0 , 2 , 0 , 2 , 2 , 4 , 
    -4, -2, -2, 0 , -2, 0 , 0 , 2 , -2, 0 , 0 , 2 , 0 , 2 , 2 , 4 , 
    -2, 0 , 0 , 2 , 0 , 2 , 2 , 4 , 0 , 2 , 2 , 4 , 2 , 4 , 4 , 6 , 
    -6, -4, -4, -2, -4, -2, -2, 0 , -4, -2, -2, 0 , -2, 0 , 0 , 2 , 
    -4, -2, -2, 0 , -2, 0 , 0 , 2 , -2, 0 , 0 , 2 , 0 , 2 , 2 , 4 , 
    -4, -2, -2, 0 , -2, 0 , 0 , 2 , -2, 0 , 0 , 2 , 0 , 2 , 2 , 4 , 
    -2, 0 , 0 , 2 , 0 , 2 , 2 , 4 , 0 , 2 , 2 , 4 , 2 , 4 , 4 , 6 , 
    -4, -2, -2, 0 , -2, 0 , 0 , 2 , -2, 0 , 0 , 2 , 0 , 2 , 2 , 4 , 
    -2, 0 , 0 , 2 , 0 , 2 , 2 , 4 , 0 , 2 , 2 , 4 , 2 , 4 , 4 , 6 , 
    -2, 0 , 0 , 2 , 0 , 2 , 2 , 4 , 0 , 2 , 2 , 4 , 2 , 4 , 4 , 6 , 
    0 , 2 , 2 , 4 , 2 , 4 , 4 , 6 , 2 , 4 , 4 , 6 , 4 , 6 , 6 , 8 ,
    };

#if (DATA_LEN == 8)
#define BIT_CONT(x)  (bit_cont[((x)&0xff)])
#elif (DATA_LEN == 16)
#define BIT_CONT(x)  (bit_cont[((x)&0xff)] + bit_cont[(((x)>>8)&0xff)])
#elif (DATA_LEN == 32)
#define BIT_CONT(x)  (bit_cont[((x)&0xff)] + bit_cont[(((x)>>8)&0xff)] + bit_cont[(((x)>>16)&0xff)] + bit_cont[(((x)>>24)&0xff)])
#elif (DATA_LEN == 64)
#define BIT_CONT(x)  (bit_cont[((x)&0xff)] + bit_cont[(((x)>>8)&0xff)] + bit_cont[(((x)>>16)&0xff)] + bit_cont[(((x)>>24)&0xff)] \
                     + bit_cont[(((x)>>32)&0xff)] + bit_cont[(((x)>>40)&0xff)] + bit_cont[(((x)>>48)&0xff)] + bit_cont[(((x)>>56)&0xff)])
#endif

static const uint64_t xnor_in[2] = {0xffffffffffffffff, 0x0000000000000000};

data_info_t *BinarizeConv2d(data_info_t *kernel, data_info_t *input, uint8_t stride, uint8_t padding){
    if(input->dim[1]/DATA_LEN <= 0)
        return NULL;
    data_info_t *output = malloc(sizeof(data_info_t));
    output->dim[3] = (input->dim[3]+padding*2-kernel->dim[3])/stride+1;
    output->dim[2] = output->dim[3];
    output->dim[1] = kernel->dim[0];
    output->dim[0] = 1;
    output->data = calloc(output->dim[2]*output->dim[3]*output->dim[1], sizeof(float)); 
    output->len = FLOAT_BYTE;
    float (*data)[output->dim[2]][output->dim[3]][output->dim[1]] = output->data;
    intx_t (*kernel_data)[kernel->dim[2]][kernel->dim[3]][kernel->dim[1]/DATA_LEN] = kernel->data;
    intx_t (*input_data)[input->dim[2]][input->dim[3]][input->dim[1]/DATA_LEN] = input->data;


    if(input->dim[1] != kernel->dim[1] || input->dim[1]%DATA_LEN != 0){
        free(output->data); 
        free(output); 
        return NULL;
    }
    for (uint16_t out_ch = 0; out_ch < kernel->dim[0]; ++out_ch) {//kernel->out_ch:输出通道数
        for (uint16_t kernel_xpos = 0; kernel_xpos < kernel->dim[2]; ++kernel_xpos) {//卷积核元素3选1
            for (uint16_t kernel_ypos = 0; kernel_ypos < kernel->dim[3]; ++kernel_ypos){ //卷积核元素3选1
                for (uint16_t x_pos = 0; x_pos < input->dim[2]; x_pos += stride) {
                    int16_t x_input = x_pos + (offset[kernel_xpos][kernel_ypos].x_start * padding);
                    if(x_input < 0 || x_input >= input->dim[2])continue;
                    for (uint16_t y_pos = 0; y_pos < input->dim[3]; y_pos += stride) {
                        int16_t y_input = y_pos + (offset[kernel_xpos][kernel_ypos].y_start * padding);
                        if (y_input < 0 || y_input >= input->dim[3]) {//判断是否与0同或
                            #ifndef USE_PADDING_ZERO
                            for (uint16_t in_ch = 0; in_ch < input->dim[1]/DATA_LEN; ++in_ch) {//in_channel/8:所有输入通道所占字节数
                                data[0][x_pos/stride][y_pos/stride][out_ch] += \
                                    BIT_CONT((kernel_data[out_ch][kernel_xpos][kernel_ypos][in_ch] ^ xnor_in[0])&(~(xnor_in[0]<<DATA_LEN)));//异或1等于同或0
                            }
                            #else
                                continue;
                            #endif
                        } else {
                            for (uint16_t in_ch = 0; in_ch < input->dim[1]/DATA_LEN; ++in_ch) {//in_channel/8:所有输入通道所占字节数
                                data[0][x_pos/stride][y_pos/stride][out_ch] += \
                                    BIT_CONT((~(kernel_data[out_ch][kernel_xpos][kernel_ypos][in_ch] ^ input_data[0][x_input][y_input][in_ch]))&(~(xnor_in[0]<<DATA_LEN)));
                                printf("out.dim[0][%d][%d][%d] = ~%08x ^%08x  ",out_ch,x_pos/stride,y_pos/stride, \
                                    kernel_data[out_ch][kernel_xpos][kernel_ypos][in_ch],input_data[0][x_input][y_input][in_ch]);
                            }
                            printf("\n");
                        }
                    }
                }
            }
        }
    }

    return output;
}

data_info_t *Conv2d(data_info_t *kernel, data_info_t *input, uint8_t stride, uint8_t padding){
    if(input->len != FLOAT_BYTE)
        return NULL;
    data_info_t *output = malloc(sizeof(data_info_t));
    output->dim[3] = (input->dim[3]+padding*2-kernel->dim[3])/stride+1;
    output->dim[2] = output->dim[3];
    output->dim[1] = kernel->dim[0];
    output->dim[0] = 1;
    output->data = calloc(output->dim[2]*output->dim[3]*output->dim[1], sizeof(float)); 
    output->len = FLOAT_BYTE;
    float (*data)[output->dim[2]][output->dim[3]][output->dim[1]] = output->data;
    float (*kernel_data)[kernel->dim[1]][kernel->dim[2]][kernel->dim[3]] = kernel->data;


    float (*input_data)[input->dim[1]][input->dim[2]][input->dim[3]] = input->data;
    if(kernel->dim[1] != input->dim[1]){
        free(output->data); 
        free(output); 
        return NULL;
    }
    for (uint16_t out_ch = 0; out_ch < kernel->dim[0]; ++out_ch) {//kernel->out_ch:输出通道数
        for (uint16_t in_ch = 0; in_ch < input->dim[1]; ++in_ch) {
            for (uint16_t kernel_xpos = 0; kernel_xpos < kernel->dim[2]; ++kernel_xpos) {//卷积核元素3选1
                for (uint16_t kernel_ypos = 0; kernel_ypos < kernel->dim[3]; ++kernel_ypos){ //卷积核元素3选1
                    for (uint16_t x_pos = 0; x_pos < input->dim[2]; x_pos += stride) {
                        for (uint16_t y_pos = 0; y_pos < input->dim[3]; y_pos += stride) {
                            int16_t x_input = x_pos + (offset[kernel_xpos][kernel_ypos].x_start * padding);
                            int16_t y_input = y_pos + (offset[kernel_xpos][kernel_ypos].y_start * padding);
                            if (x_input >= 0 && x_input < input->dim[2] && y_input >= 0 && y_input < input->dim[3]) {//判断是否与0同或
                                data[0][x_pos/stride][y_pos/stride][out_ch]+= \
                                input_data[0][in_ch][x_input][y_input] * kernel_data[out_ch][in_ch][kernel_xpos][kernel_ypos];
                            }
                        } 
                    }
                }
            }
        }
    }


    return output;
}