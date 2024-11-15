#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
// #define DEBUG
#include "utils.h"
#include "core.h"

data_info_t *bachnorm(data_info_t *input, data_info_t *batchnorm){
    if(!input || !batchnorm)
        return NULL;
    float (*bn)[OMAX][batchnorm->dim[0]] = batchnorm->data;
    int16_t (*activate)[input->dim[1]][input->dim[2]][input->dim[3]] = input->data;
    return NULL;
}


