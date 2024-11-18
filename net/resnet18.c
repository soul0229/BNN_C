#include "common.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h> 
#include "conv.h"
#include "utils.h"
#include "core.h"

extern net_t *net_start;

data_info_t *data_binary(data_info_t *input){
    if(!(input && input->data && input->len == FLOAT_BYTE))
        return NULL;
    float (*in_data)[input->dim[2]][input->dim[3]][input->dim[1]] = input->data;
    uint8_t (*out_data)[input->dim[2]][input->dim[3]][input->dim[1]/8] = calloc(input->dim[2]*input->dim[3]*input->dim[1]/8, sizeof(uint8_t));
    for(uint16_t dim1 = 0; dim1 < input->dim[1]; ++dim1)
        for(uint16_t dim2 = 0; dim2 < input->dim[2]; ++dim2)
            for(uint16_t dim3 = 0; dim3 < input->dim[3]; ++dim3)
                out_data[0][dim2][dim3][dim1/8] |= (in_data[0][dim2][dim3][dim1] >= 0.0f)?(0x80 >> dim1%8):0x00;
    input->len = BINARY;
    free(input->data);
    input->data = out_data;

    return input;
}

void net_nference(common_t *net, data_info_t *input){
    common_t *stage = net;
    data_info_t *output = NULL;
    while(stage != NULL){
        switch(stage->type){
            case NET_ROOT:
            case TLAYER:
                if(stage->child != NULL)
                    net_nference((common_t *)stage->child, input);
                break;
            case TSHORTCUT:
                if(stage->child != NULL)
                    net_nference((common_t *)stage->child, input);
                break;
            case TCONV:
                if(((common_t *)stage->parent)->type == NET_ROOT)
                    output = Conv2d((data_info_t *)stage, input, 1, 1, 1);
                else{
                    output = BinarizeConv2d((data_info_t *)stage, input, (input->dim[0] == input->dim[1])?1:2, 1, 1);
                }
                break;
            case TBATCHNORM:
                output = bachnorm(output, (data_info_t *)stage);
                if(((common_t *)stage->sibling)->type != TSHORTCUT && ((common_t *)stage->parent)->type != NET_ROOT)
                    output = hardtanh(output);
                break;
            case TLINER:
                output = linear_data(output, (data_info_t *)stage);
                break;
        default:break;
        }
        stage = (common_t *)stage->sibling;
    }
}

void resnet18(data_info_t *input){
    if(!input)
        return;
    net_t *net = net_start;

}