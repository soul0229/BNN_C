#include "conv.h"

extern const conv_offset offset[];

Activate_TypeDef *Conv2d(CONV_KernelTypeDef *kernel, Activate_TypeDef* input, uint8_t stride, uint8_t padding, bool depthwise){
    Activate_TypeDef *output = (Activate_TypeDef *)calloc(1, sizeof(Activate_TypeDef));
    output->size = (input->size+padding*2-kernel->size)/stride+1;
    output->ch = kernel->out_ch;
    output->active = (float *)calloc(output->size*output->size*output->ch, sizeof(float)); 
    output->binary = FLOAT_BYTE;

    if(depthwise){
        if(kernel->in_ch != input->ch){
            free(output->active); 
            free(output); 
            return NULL;
        }
        for (uint16_t out_ch = 0; out_ch < kernel->out_ch; ++out_ch) {//kernel->out_ch:输出通道数
            for (uint16_t in_ch = 0; in_ch < input->ch; ++in_ch) {
                for (uint16_t kernel_pos = 0; kernel_pos < (kernel->size * kernel->size); ++kernel_pos) {//卷积核元素9选1
                    for (uint16_t x_pos = 0; x_pos < input->size; x_pos += stride) {
                        for (uint16_t y_pos = 0; y_pos < input->size; y_pos += stride) {
                            int16_t x_input = x_pos + (offset[kernel_pos].x_start * padding);
                            int16_t y_input = y_pos + (offset[kernel_pos].y_start * padding);
                            if (x_input >= 0 && x_input < input->size && y_input >= 0 && y_input < input->size) {//判断是否与0同或
                                ((float*)(output->active))[(out_ch * output->size  + x_pos/stride) * output->size + y_pos/stride]+= \
                                    ((float*)(input->active))[(in_ch * input->size + x_input) * input->size + y_input] * \
                                    ((float*)(kernel->kernel))[(out_ch*kernel->in_ch+in_ch)*kernel->size*kernel->size+kernel_pos];
                            }
                        } 
                    }
                }
            }
        }
    }
    else{
        if(kernel->in_ch != 1){
            free(output->active); 
            free(output); 
            return NULL;
        }
        for (uint16_t out_ch = 0; out_ch < kernel->out_ch; ++out_ch) {//kernel->out_ch:输出通道数
            for (uint16_t in_ch = 0; in_ch < input->ch; ++in_ch) {
                for (uint16_t kernel_pos = 0; kernel_pos < (kernel->size * kernel->size); ++kernel_pos) {//卷积核元素9选1
                    for (uint16_t x_pos = 0; x_pos < input->size; x_pos += stride) {
                        for (uint16_t y_pos = 0; y_pos < input->size; y_pos += stride) {
                            int16_t x_input = x_pos + (offset[kernel_pos].x_start * padding);
                            int16_t y_input = y_pos + (offset[kernel_pos].y_start * padding);
                            if (x_input >= 0 && x_input < input->size && y_input >= 0 && y_input < input->size) {//判断是否与0同或
                                ((float*)(output->active))[(out_ch * output->size  + x_pos/stride) * output->size + y_pos/stride]+= \
                                    ((float*)(input->active))[(in_ch*output->size+x_input)*output->size+y_input] * \
                                    ((float*)(kernel->kernel))[out_ch*kernel->size*kernel->size+kernel_pos];
                            }
                        } 
                    }
                }
            }
        }
    }
    return output;
}