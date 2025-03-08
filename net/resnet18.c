#include "common.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h> 
#include "conv.h"
#include "utils.h"
#include "core.h"

#define MAX_ACTIVATE_SIZE 64*32*32*4

// data_info_t temp_Ab = {
//     .type = ACTIVATE,
//     .name = "binary activate",
//     .len = BINARY,
// };
data_info_t *temp_Ab;
static data_info_t *SignActivate(data_info_t *activate){
    if(!temp_Ab || !temp_Ab->data)
        return NULL;
    uint8_t (*data_in)[activate->dim[3]][activate->dim[1]] = activate->data;
    uint8_t (*data_out)[activate->dim[3]][activate->dim[1]/8] = temp_Ab->data;


    for(uint16_t dim2 = 0; dim2 < activate->dim[2]; ++dim2)
        for(uint16_t dim3 = 0; dim3 < activate->dim[3]; ++dim3)
            for(uint16_t dim1 = 0; dim1 < activate->dim[1]; ++dim1){
                data_out[dim2][dim3][dim1/8] |= ((data_in[dim2][dim3][dim1]>=0)?1:0)<<(8 - dim1%8);
        }

    return temp_Ab;
}



static data_info_t *data_binary(data_info_t *input){
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

static data_info_t *data_add(data_info_t *input1, data_info_t *input2){
    if(!(input1 && input1->data && input1->len == FLOAT_BYTE))
        return NULL;
    if(!(input2 && input2->data && input2->len == FLOAT_BYTE))
        return NULL;
    float (*data1)[input1->dim[2]][input1->dim[3]][input1->dim[1]] = input1->data;
    float (*data2)[input2->dim[2]][input2->dim[3]][input2->dim[1]] = input2->data;

    for(uint16_t dim1 = 0; dim1 < input1->dim[1]; ++dim1)
        for(uint16_t dim2 = 0; dim2 < input1->dim[2]; ++dim2)
            for(uint16_t dim3 = 0; dim3 < input1->dim[3]; ++dim3)
                data1[0][dim2][dim3][dim1] += data2[0][dim2][dim3][dim1];
    free(input2->data);
    free(input2);
    input2 = NULL;
    return input1;
}

static data_info_t *BasicBlock(data_info_t *input, common_t *basicBlock){
    if(!input || input->data)return NULL;
    data_info_t *net_args = (data_info_t *)basicBlock;
    data_info_t *Ab = SignActivate(input);
    data_info_t *output = BinarizeConv2d(net_args, Ab, net_args->dim[0]/net_args->dim[1], 1);
    output = bachnorm(output, net_args = (data_info_t *)net_args->sibling);
    output = hardtanh(output);
    Ab = SignActivate(output);
    free(output->data);
    free(output);
    output = BinarizeConv2d(net_args = (data_info_t *)net_args->sibling, Ab, net_args->dim[0]/net_args->dim[1], 1);
    output = bachnorm(output, net_args = (data_info_t *)net_args->sibling);
    data_info_t *shortcut = input;
    if(((data_info_t *)basicBlock)->dim[0]/((data_info_t *)basicBlock)->dim[1] == 2){
        shortcut = BinarizeConv2d(net_args = (data_info_t *)net_args->sibling, Ab, net_args->dim[0]/net_args->dim[1], 1);
        shortcut = bachnorm(shortcut, net_args = (data_info_t *)net_args->sibling);
        free(input->data);
        free(input);
    }
    output = data_add(output, shortcut);
    output = hardtanh(output);

    return output;
}

data_info_t *net_inference(common_t *net, data_info_t *input){
    common_t *stage = net;
    data_info_t *output = input, *shuortcut = NULL, *temp = NULL; //temp用于释放无用的中间结果
    
    while(stage != NULL){
        temp = output;
        switch(stage->type){
            case NET_ROOT:
            case TLAYER:
                if(stage->child != NULL)
                    output = net_inference((common_t *)stage->child, output);
                if((common_t *)stage->sibling && ((common_t *)stage->sibling)->type != TLAYER)
                    output = avg_pool(output, output->dim[3]);
                break;
            case TSHORTCUT:
                if(stage->child != NULL)
                    shuortcut = net_inference((common_t *)stage->child, input);

                output = data_add(output, shuortcut);
                output = hardtanh(output);
                break;
            case TCONV:
                if(((common_t *)stage->parent)->type == NET_ROOT)
                    output = Conv2d((data_info_t *)stage, output, 1, 1);
                else{
                    output = data_binary(output);
                    output = BinarizeConv2d((data_info_t *)stage, output, ((data_info_t *)stage)->dim[0]/((data_info_t *)stage)->dim[1], 1);
                }
                break;
            case TBATCHNORM:
                output = bachnorm(output, (data_info_t *)stage);
                if((common_t *)stage->sibling == NULL){
                    output = data_add(output, input);
                    output = hardtanh(output);
                }
                else if(((common_t *)stage->parent)->type != TSHORTCUT && ((common_t *)stage->sibling)->type != TSHORTCUT && ((common_t *)stage->parent)->type == TLAYER){
                    output = hardtanh(output);
                }
                break;
            case TLINER:
                output = linear_data(output, (data_info_t *)stage);
                break;
            default:break;
        }

        if(temp && temp != input){
            free(temp->data);
            free(temp);
            temp =NULL;
        }
        stage = (common_t *)stage->sibling;
    }
        free(input->data);
        free(input);
        input = NULL;

    return output;
}

void resnet18(data_info_t *input){
    if(!input)
        return;
    net_t *net = Net;
    data_info_t *output = net_inference((common_t*)net, input);
    if(!output){
        printf("net inference error!\n");
        return;
    }
    printf("dim[%d][%d][%d][%d]", output->dim[0], output->dim[1], output->dim[2], output->dim[3]);
}