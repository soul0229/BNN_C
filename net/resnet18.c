#include "common.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h> 
#include "conv.h"
#include "utils.h"
#include "core.h"
#include <string.h> 

#define MAX_ACTIVATE_SIZE 64*32*32*4
char *CIFAR10_tag[] = { "飞机 (airplane)",
                        "汽车 (automobile)",
                        "鸟 (bird)",
                        "猫 (cat)",
                        "鹿 (deer)",
                        "狗 (dog)",
                        "青蛙 (frog)",
                        "马 (horse)",
                        "船 (ship)",
                        "卡车 (truck)"};

extern const char *name_typ[];

static data_info_t *SignActivate(data_info_t *activate){
    if(activate->len != FLOAT_BYTE){
        printf("SignActivate input data len error!\n");
        return NULL;
    }

    data_info_t *temp = malloc(sizeof(data_info_t));
    temp->dim[0] = activate->dim[0];
    temp->dim[1] = activate->dim[1];
    temp->dim[2] = activate->dim[2];
    temp->dim[3] = activate->dim[3];
    temp->len = BINARY;
    temp->data = malloc(activate->dim[0]*activate->dim[1]*activate->dim[2]*activate->dim[3]/8);
    memset(temp->data,0x00,activate->dim[0]*activate->dim[1]*activate->dim[2]*activate->dim[3]/8);

    float (*data_in)[activate->dim[3]][activate->dim[1]] = activate->data;
    uint8_t (*data_out)[activate->dim[3]][activate->dim[1]/8] = temp->data;

    for(uint16_t dim2 = 0; dim2 < activate->dim[2]; ++dim2){
        for(uint16_t dim3 = 0; dim3 < activate->dim[3]; ++dim3){
            for(uint16_t dim1 = 0; dim1 < activate->dim[1]; ++dim1){
                data_out[dim2][dim3][dim1/8] |= (((data_in[dim2][dim3][dim1] +1e-5) < 0)?0x00:(0x01<<(dim1%8)));
            }
        }
    }
    return temp;
}


static data_info_t *data_add(data_info_t *input1, data_info_t *input2){
    if(!(input1 && input1->data && input1->len == FLOAT_BYTE))
        return NULL;
    if(!(input2 && input2->data && input2->len == FLOAT_BYTE))
        return NULL;
    if(input1->dim[0] != input2->dim[0] || input1->dim[1] != input2->dim[1] || input1->dim[2] != input2->dim[2] || input1->dim[3] != input2->dim[3]){
        printf("data_add input1->dim[%d][%d][%d][%d] != input2->dim[%d][%d][%d][%d]\n", \
            input1->dim[0], input1->dim[1], input1->dim[2], input1->dim[3], input2->dim[0], input2->dim[1], input2->dim[2], input2->dim[3]);
        return NULL;
    }
    float (*data1)[input1->dim[2]][input1->dim[3]][input1->dim[1]] = input1->data;
    float (*data2)[input2->dim[2]][input2->dim[3]][input2->dim[1]] = input2->data;

    
    for(uint16_t dim2 = 0; dim2 < input1->dim[2]; ++dim2)
        for(uint16_t dim3 = 0; dim3 < input1->dim[3]; ++dim3)
            for(uint16_t dim1 = 0; dim1 < input1->dim[1]; ++dim1)
                data1[0][dim2][dim3][dim1] += data2[0][dim2][dim3][dim1];
    free(input2->data);
    free(input2);
    input2 = NULL;
    return input1;
}

static data_info_t *BasicBlock(data_info_t *input, common_t *basicBlock){
    if(!input || !input->data){
        return NULL;
    }    
    // print_data(input);
    // return NULL;
    data_info_t *net_args = (data_info_t*)basicBlock->child;
    data_info_t *Ab = SignActivate(input);
    // binary_conv_data_trans(Ab, net_args);
    // return NULL;

    data_info_t *output = BinarizeConv2d(net_args, Ab, net_args->dim[0]/net_args->dim[1], 1);

    net_args = (data_info_t *)net_args->sibling;
    output = bachnorm(output, net_args);
    output = hardtanh(output);
    printf("%s:%d output->dim[%d][%d][%d][%d]\n",__FILE__,__LINE__, output->dim[0], output->dim[1], output->dim[2], output->dim[3]);

    data_info_t *Ab2 = SignActivate(output);
    free(output->data);
    free(output);
    net_args = (data_info_t *)net_args->sibling;
    output = BinarizeConv2d(net_args, Ab2, net_args->dim[0]/net_args->dim[1], 1);
    net_args = (data_info_t *)net_args->sibling;
    output = bachnorm(output, net_args);
    free(Ab2->data);
    free(Ab2);

    data_info_t *shortcut = input;
    if(((data_info_t *)basicBlock->child)->dim[0]/((data_info_t *)basicBlock->child)->dim[1] == 2 && ((data_info_t *)net_args->sibling)->type == TSHORTCUT){
        net_args = (data_info_t *)((common_t *)net_args->sibling)->child;
        printf("net_args type:%s \n",name_typ[TSHORTCUT]);

        shortcut = BinarizeConv2d(net_args, Ab, net_args->dim[0]/net_args->dim[1], 0);
        net_args = (data_info_t *)net_args->sibling;
        shortcut = bachnorm(shortcut, net_args);

        free(input->data);
        free(input);
    }
    free(Ab->data);
    free(Ab);

    output = data_add(output, shortcut);
    output = hardtanh(output);

    return output;
}

data_info_t *net_inference(common_t *net, data_info_t *input){
    common_t *stage = (net?net:(common_t *)Net);
    if(stage->child == NULL && stage->sibling == NULL)
        return NULL;

    data_info_t *output = input; //temp用于释放无用的中间结果
    Compose_RGB_data(input, CIFAR);

    // printf("%s:%d Compose_RGB_data ok\n",__FILE__,__LINE__);
    while(stage != NULL){
        if(output == NULL){ 
            printf("error output = NULL\n");
            return NULL;
        }
        // printf("%s:%d in while stage->name:%s \n",__FILE__,__LINE__,stage->name);
        switch(stage->type){
            case NET_ROOT:
                stage = (common_t *)stage->child;
                continue;
            case TLAYER:
                if(!stage->child || ((common_t *)stage->child)->type != TLAYER){
                    printf("error");
                    return NULL;
                }
                common_t *layer = (common_t *)stage->child;
                printf("layer name:%s\n", layer->name);
                output = BasicBlock(output,layer);
                if(output==NULL)printf("BasicBlock error\n");
                layer = (common_t *)layer->sibling;
                printf("layer name:%s\n", layer->name);
                output = BasicBlock(output,layer);
                break;
            case TSHORTCUT:
                printf("error");
                return NULL;
            case TCONV:
                output = Conv2d((data_info_t *)stage,output, 1,1);
                free(input->data);
                free(input);
                break;
            case TBATCHNORM:
                printf("sibling type:%s\n",name_typ[((common_t *)stage->sibling)->type]);
                if(((common_t *)stage->sibling)->type == TLINER){
                    output = avg_pool(output, 4);
                }
                output = bachnorm(output, (data_info_t *)stage);
                
                // if(strstr(stage->name,"bn1") !=0)
                //     output = hardtanh(output);
                break;
            case TLINER:
                output = linear_data(output, (data_info_t *)stage);
                break;
            default:
                break;
        }
        stage = (common_t *)stage->sibling;
    }

    return output;
}

typedef struct {
    float value;   // 数据值
    int index;   // 数据的原始位置
} Data;

Data result_out[10] = {
    {0,0},{0,1},{0,2},{0,3},{0,4},
    {0,5},{0,6},{0,7},{0,8},{0,9},
};

int compare(const void *a, const void *b) {
    Data *dataA = (Data *)a;
    Data *dataB = (Data *)b;

    // 按照 value 值进行升序排序
    return dataB->value - dataA->value;
}

void resnet18(data_info_t *input){
    if(!input)
        return;
    data_info_t *output = net_inference(NULL, input);
    if(!output){
        printf("net inference error!\n");
        return;
    }
    float *result = output->data;
    for(uint16_t dim=0;dim < output->dim[0];dim++){
        printf("%2.2f ",result[dim]);
        result_out[dim].value = result[dim];
    }
    printf("\n");
    qsort(result_out, 10, sizeof(Data), compare);
    printf("最大值: %.2f, 标签: %d, 类别：%s\n", result_out[0].value, result_out[0].index, CIFAR10_tag[result_out[0].index]);

}