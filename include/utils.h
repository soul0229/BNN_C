#ifndef __UTILS_H__
#define __UTILS_H__
#include <stdint.h>
#include <core.h>

enum DATASET_INFO {
    CIFAR = 0,
    IMAGENET,
    DSET_NUM,

    DSET_MEAN = 0,
    DSET_STD,
    HANDLE_NUM,
    
    RED = 0,
    GREEN,
    BLUE,
    CHANNEL_NUM
};

data_info_t *Compose_RGB_data(data_info_t *input, enum DATASET_INFO dset_sel);
data_info_t *SignActivate(data_info_t *activate);
data_info_t *data_add(data_info_t *input1, data_info_t *input2);
data_info_t *bachnorm(data_info_t *input, data_info_t *batchnorm);
data_info_t *linear_data(data_info_t *input, data_info_t *linear);
data_info_t *avg_pool(data_info_t *input, uint8_t size);
data_info_t *hardtanh(data_info_t *input);

int jpg_decode(char* name, data_info_t* data);
void print_RGB_data(data_info_t *jpg_RBG);
void binary_conv_data_trans(data_info_t *Ab, data_info_t* Wb);
void print_normal_order_data(data_info_t *input);
void print_data(data_info_t *input);


void json_model_parse_v2(char* file_name);
void load_ml_net(char *file_name);
void printf_net_structure(common_t *data);
void printf_appoint_data(char *str, common_t *data);
void free_net(common_t **net);
#endif
