#ifndef __UTILS_H__
#define __UTILS_H__
#include "stdio.h"
#include "conv.h"
#include <stdint.h>
#include <core.h>

void json_model_parse_v2(char* file_name);
void load_ml_net(char *file_name);
void printf_net_structure(common_t *data);
void printf_appoint_data(char *str, common_t *data);
void free_net(common_t **net);
#endif
