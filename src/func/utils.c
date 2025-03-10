#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
// #define DEBUG
#include "utils.h"
#include "common.h"
#include "core.h"


float normalize[DSET_NUM][HANDLE_NUM][CHANNEL_NUM]={
    [CIFAR] = {
        [DSET_MEAN] = {0.485f, 0.456f, 0.406f},
        [DSET_STD] = {0.229f, 0.224f, 0.225f},
    },
    [IMAGENET] = {
        [DSET_MEAN] = {0.507f, 0.487f, 0.441f},
        [DSET_STD] = {0.267f, 0.256f, 0.276f},
    },
};

data_info_t *Compose_RGB_data(data_info_t *input, enum DATASET_INFO dset_sel){
    if(dset_sel >= DSET_NUM || input == NULL || input->data == NULL || input->len != ONE_BYTE)
        return NULL;

    uint8_t (*in_data)[input->dim[2]][input->dim[3]] = input->data;
    float (*data)[input->dim[2]][input->dim[3]] = malloc(sizeof(float)*input->dim[1]*input->dim[2]*input->dim[3]); //默认为RGB三个通道
    
    for(uint16_t channel = 0; channel < input->dim[1]; ++channel){
        for(uint16_t x_size = 0; x_size < input->dim[2]; ++x_size){
            for(uint16_t y_size = 0; y_size < input->dim[3]; ++y_size){
                data[channel][x_size][y_size] = ((float)in_data[channel][x_size][y_size]/255 - normalize[dset_sel][DSET_MEAN][channel]) \
                    / normalize[dset_sel][DSET_STD][channel];
            }
        }
    }
    free(input->data);
    input->data = data;
    input->len = FLOAT_BYTE;
    return input;
}

data_info_t *bachnorm(data_info_t *input, data_info_t *batchnorm){
    if(input->dim[1] != batchnorm->dim[0] || batchnorm->type != TBATCHNORM || input->len != FLOAT_BYTE){
        printf("bachnorm error\n");
        return NULL;
    }
    float (*bn)[batchnorm->dim[0]] = batchnorm->data;
    float (*activate)[input->dim[1]][input->dim[2]][input->dim[3]] = input->data;

    for(uint16_t dim2 = 0; dim2 < input->dim[2]; ++dim2){
        for(uint16_t dim3 = 0; dim3 < input->dim[3]; ++dim3)
            for(uint16_t ch = 0; ch < input->dim[1]; ++ch){
                activate[0][dim2][dim3][ch] = activate[0][dim2][dim3][ch]*bn[OA][ch] + bn[OB][ch];
        }
    }
    input->len = FLOAT_BYTE;
    return input;
}

data_info_t *Relu(data_info_t *input){
    if(!(input && input->data && input->len == FLOAT_BYTE))
        return NULL;
    float (*activate)[input->dim[2]][input->dim[3]] = input->data;
    for(uint16_t dim1 = 0; dim1 < input->dim[1]; ++dim1)
        for(uint16_t dim2 = 0; dim2 < input->dim[2]; ++dim2)
            for(uint16_t dim3 = 0; dim3 < input->dim[3]; ++dim3)
                activate[dim1][dim2][dim3] = (activate[dim1][dim2][dim3] < 0.0f)?0.0f:activate[dim1][dim2][dim3];
    return input;
}

data_info_t *PRelu(data_info_t *input, float alpha){
    if(!(input && input->data && input->len == FLOAT_BYTE))
        return NULL;
    float (*activate)[input->dim[2]][input->dim[3]] = input->data;
    for(uint16_t dim1 = 0; dim1 < input->dim[1]; ++dim1)
        for(uint16_t dim2 = 0; dim2 < input->dim[2]; ++dim2)
            for(uint16_t dim3 = 0; dim3 < input->dim[3]; ++dim3)
                activate[dim1][dim2][dim3] = (activate[dim1][dim2][dim3] < 0.0f)?alpha*activate[dim1][dim2][dim3]:activate[dim1][dim2][dim3];
    return input;
}

data_info_t *hardtanh(data_info_t *input){
    if(!(input && input->data && input->len == FLOAT_BYTE))
        return NULL;
    float (*activate)[input->dim[2]][input->dim[3]] = input->data;
    for(uint16_t dim1 = 0; dim1 < input->dim[1]; ++dim1)
        for(uint16_t dim2 = 0; dim2 < input->dim[2]; ++dim2)
            for(uint16_t dim3 = 0; dim3 < input->dim[3]; ++dim3)
                activate[dim1][dim2][dim3] = (activate[dim1][dim2][dim3] > 1.0f)?1.0f:((activate[dim1][dim2][dim3] < -1.0f)? -1.0f:activate[dim1][dim2][dim3]);

    return input;
}

data_info_t *avg_pool(data_info_t *input, uint8_t size){
    if(!input || !input->data || !size){
        printf("avg_pool error\n");
        return NULL;
    }
    float (*data)[input->dim[1]][input->dim[2]][input->dim[3]] = input->data;
    float (*output)[input->dim[1]][input->dim[2]/size][input->dim[3]/size] = calloc(input->dim[1]*(input->dim[2]/size)*(input->dim[3]/size), sizeof(float));
    
    for(uint16_t dim1 = 0; dim1 < input->dim[1]; ++dim1)
        for(uint16_t dim2 = 0; dim2 < input->dim[2]; ++dim2)
            for(uint16_t dim3 = 0; dim3 < input->dim[3]; ++dim3)
                output[0][dim1][dim2/size][dim3/size]+= data[0][dim2][dim3][dim1];
            
    for(uint16_t dim1 = 0; dim1 < input->dim[1]; ++dim1)
        for(uint16_t dim2 = 0; dim2 < input->dim[2]/size; ++dim2)
            for(uint16_t dim3 = 0; dim3 < input->dim[3]/size; ++dim3)
                output[0][dim1][dim2][dim3] /= size*size;
    
    input->dim[2] = input->dim[2]/size;
    input->dim[3] = input->dim[3]/size;
    free(input->data);
    input->data = output;
    input->len = FLOAT_BYTE;

    return input;
}

data_info_t *linear_data(data_info_t *input, data_info_t *linear){
    if(!(input && input->data && linear && linear->data && linear->dim[1] == input->dim[1]) || linear->type != TLINER){
        printf("linear data error!\n");
        return NULL;
    }
    float (*data_A)[linear->dim[1]] = linear->data;
    float *data_B = input->data;
    float *output = calloc(linear->dim[0], sizeof(float));
    
    for(uint16_t dim0 = 0; dim0 < linear->dim[0]; ++dim0)
        for(uint16_t dim1 = 0; dim1 < linear->dim[1]; ++dim1)
                output[dim0] += data_A[dim0][dim1] * data_B[dim1];

    for(uint16_t dim0 = 0; dim0 < linear->dim[0]; ++dim0){
        output[dim0]+= data_A[linear->dim[0]][dim0];
    }

    input->dim[0] = linear->dim[0];
    input->dim[1] = input->dim[2]*input->dim[3];
    input->dim[2] = 1;
    input->dim[3] = 1;
    free(input->data);
    input->data = output;

    return input;
}

