#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
// #define DEBUG
#include "utils.h"
#include "common.h"
#include "core.h"

data_info_t *bachnorm(data_info_t *input, data_info_t *batchnorm){
    if(input->dim[1] != batchnorm->dim[0])
        return NULL;
    float (*bn)[batchnorm->dim[0]] = batchnorm->data;

    if(input->len == FLOAT_BYTE){
        float (*activate)[input->dim[1]][input->dim[2]][input->dim[3]] = input->data;
        for(uint16_t ch = 0; ch < input->dim[1]; ++ch){
            for(uint16_t dim2 = 0; dim2 < input->dim[2]; ++dim2){
                for(uint16_t dim3 = 0; dim3 < input->dim[3]; ++dim3)
                    activate[0][ch][dim2][dim3] = (activate[0][ch][dim2][dim3] - bn[OMEAN][ch])/sqrt(bn[OVAR][ch]+ 1e-5) * bn[OWEIGHT][ch] + bn[OBIAS][ch];
            }
        }
    }
    else if(input->len == TWO_BYTE){
        int16_t (*activate)[input->dim[1]][input->dim[2]][input->dim[3]] = input->data;
        float (*output)[input->dim[1]][input->dim[2]][input->dim[3]] = calloc(input->dim[1]*input->dim[2]*input->dim[3], sizeof(float));
        for(uint16_t ch = 0; ch < input->dim[1]; ++ch){
            for(uint16_t dim2 = 0; dim2 < input->dim[2]; ++dim2){
                for(uint16_t dim3 = 0; dim3 < input->dim[3]; ++dim3)
                    output[0][ch][dim2][dim3] = (activate[0][ch][dim2][dim3] - bn[OMEAN][ch])/sqrt(bn[OVAR][ch]+ 1e-5) * bn[OWEIGHT][ch] + bn[OBIAS][ch];
            }
        }
        free(input->data);
        input->len = FLOAT_BYTE;
        input->data = output;
    }
    return input;
}

data_info_t *Relu(data_info_t *input){
    return input;
}

data_info_t *hardtanh(data_info_t *input){
    if(!input || !input->data || input->len != FLOAT_BYTE)
        return NULL;
    float (*activate)[input->dim[1]][input->dim[2]][input->dim[3]] = input->data;
    for(uint16_t dim1 = 0; dim1 < input->dim[1]; ++dim1)
        for(uint16_t dim2 = 0; dim2 < input->dim[2]; ++dim2)
            for(uint16_t dim3 = 0; dim3 < input->dim[3]; ++dim3)
                activate[0][dim1][dim2][dim3] = (activate[0][dim1][dim2][dim3] > 1.0f)?1.0f:((activate[0][dim1][dim2][dim3] < -1.0f)? -1.0f:activate[0][dim1][dim2][dim3]);
    return input;
}


