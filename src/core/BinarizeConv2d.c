#include <stdio.h>
#include <stdlib.h>
#include "conv.h"

const conv_offset offset[] = {
    {-1, -1,},{-1, 0,},{-1, 1,},
    {0, -1,}, {0, 0,}, {0, 1,},
    {1, -1,}, {1, 0,}, {1, 1,},
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

data_info_t *BinarizeConv2d(data_info_t *kernel, data_info_t* input, uint8_t stride, uint8_t padding, bool depthwise){
    uint8_t byte_num = input->dim[1]/DATA_LEN;//in_channel/DATA_LEN:所有输入数据通道所占字节数
    if(byte_num <= 0)
        return NULL;
    data_info_t *output = malloc(sizeof(data_info_t));
    output->dim[3] = (input->dim[3]+padding*2-kernel->dim[3])/stride+1;
    output->dim[2] = output->dim[3];
    output->dim[1] = kernel->dim[0];
    output->dim[0] = 1;
    output->data = calloc(output->dim[2]*output->dim[3]*output->dim[1], sizeof(int16_t)); 
    output->len = TWO_BYTE;

    if(depthwise){
        if(input->dim[1] != kernel->dim[1] || input->dim[1]%DATA_LEN != 0){
            free(output->data); 
            free(output); 
            return NULL;
        }
        for (uint16_t out_ch = 0; out_ch < kernel->dim[0]; ++out_ch) {//kernel->out_ch:输出通道数
            for (uint16_t kernel_pos = 0; kernel_pos < (kernel->dim[2] * kernel->dim[3]); ++kernel_pos) {//卷积核元素9选1
                for (uint16_t x_pos = 0; x_pos < input->dim[2]; x_pos += stride) {
                    for (uint16_t y_pos = 0; y_pos < input->dim[3]; y_pos += stride) {
                        int16_t x_input = x_pos + (offset[kernel_pos].x_start * padding);
                        int16_t y_input = y_pos + (offset[kernel_pos].y_start * padding);
                        if (x_input >= 0 && x_input < input->dim[2] && y_input >= 0 && y_input < input->dim[3]) {//判断是否与0同或
                            for (uint16_t in_ch = 0; in_ch < byte_num; ++in_ch) {//in_channel/8:所有输入通道所占字节数
                                ((int16_t*)(output->data))[(out_ch * output->dim[2]  + x_pos/stride) * output->dim[3] + y_pos/stride] += \
                                    BIT_CONT(~((((intx_t*)(kernel->data))[(out_ch*(kernel->dim[2] * kernel->dim[3])*byte_num)+kernel_pos*byte_num+in_ch]) ^ \
                                        ((intx_t*)(input->data))[x_input * (input->dim[3]*byte_num) + (y_input*byte_num)+in_ch])&(~(xnor_in[0]<<DATA_LEN)));
                            }
                        } else {
                            #ifndef USE_PADDING_ZERO
                            for (uint16_t in_ch = 0; in_ch < byte_num; ++in_ch) {//in_channel/8:所有输入通道所占字节数
                                ((int16_t*)(output->data))[(out_ch * output->dim[2]  + x_pos/stride) * output->dim[3] + y_pos/stride] += \
                                    BIT_CONT(((((intx_t*)(kernel->data))[(out_ch*(kernel->dim[2] * kernel->dim[3])*byte_num)+kernel_pos*byte_num+in_ch]) ^ xnor_in[0])&(~(xnor_in[0]<<DATA_LEN)));//异或1等于同或0
                            }
                            #endif
                        }
                    }
                }
            }
        }
    }
    else{
        if(kernel->dim[1] != 1 || input->dim[1]%DATA_LEN != 0){
            free(output->data); 
            free(output); 
            return NULL;
        }
        for (uint16_t out_ch = 0; out_ch < kernel->dim[0]; ++out_ch) {//kernel->out_ch:输出通道数
            for (uint16_t kernel_pos = 0; kernel_pos < (kernel->dim[2] * kernel->dim[3]); ++kernel_pos) {//卷积核元素9选1
                for (uint16_t x_pos = 0; x_pos < output->dim[2]; x_pos += stride) {
                    for (uint16_t y_pos = 0; y_pos < output->dim[3]; y_pos += stride) {
                        int16_t x_input = x_pos + (offset[kernel_pos].x_start * padding);
                        int16_t y_input = y_pos + (offset[kernel_pos].y_start * padding);
                        if (x_input >= 0 && x_input < input->dim[2] && y_input >= 0 && y_input < input->dim[3]) {//判断是否与0同或
                            for (uint16_t in_ch = 0; in_ch < byte_num; ++in_ch) {//in_channel/8:所有输入通道所占字节数
                                ((int16_t*)(output->data))[(out_ch * output->dim[2]  + x_pos/stride) * output->dim[3] + y_pos/stride] +=
                                    BIT_CONT((xnor_in[((((intx_t*)(kernel->data))[kernel_pos*kernel->dim[0]/DATA_LEN+out_ch/DATA_LEN] >> out_ch)&0x01)] ^ 
                                        ((intx_t*)(input->data))[x_input * (input->dim[3]*byte_num) + (y_input*byte_num)+in_ch])&(~(xnor_in[0]<<DATA_LEN)));
                            }
                        } else {
                            #ifndef USE_PADDING_ZERO
                            for (uint16_t in_ch = 0; in_ch < byte_num; ++in_ch) {//in_channel/8:所有输入通道所占字节数
                                ((int16_t*)(output->data))[(out_ch * output->dim[2]  + x_pos/stride) * output->dim[3] + y_pos/stride] +=
                                    BIT_CONT((xnor_in[((((intx_t*)(kernel->data))[kernel_pos*kernel->dim[0]/DATA_LEN+out_ch/DATA_LEN] >> out_ch)&0x01)] ^ xnor_in[1])&(~(xnor_in[0]<<DATA_LEN)));//异或1等于同或0
                            }
                            #endif
                        }
                    }
                }
            }
        }
    }
    return output;
}

