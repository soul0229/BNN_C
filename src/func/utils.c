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

data_info_t *SignActivate(data_info_t *activate){
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
                data_out[dim2][dim3][dim1/8] |= (((data_in[dim2][dim3][dim1] )< 0)?0x00:(0x01<<(dim1%8)));
            }
        }
    }
    return temp;
}

data_info_t *data_add(data_info_t *input1, data_info_t *input2){
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

// 位置敏感
data_info_t *bachnorm(data_info_t *input, data_info_t *batchnorm){
    if(input->dim[1] != batchnorm->dim[0] || batchnorm->dim[1] != 1 || batchnorm->dim[2] != 1 || batchnorm->dim[3] != 1 \
        || batchnorm->type != TBATCHNORM || input->len != FLOAT_BYTE){
        printf("bachnorm error\n");
        return NULL;
    }
    float (*bn)[batchnorm->dim[0]] = batchnorm->data;
    float (*activate)[input->dim[2]][input->dim[3]][input->dim[1]] = input->data;

    for(uint16_t dim2 = 0; dim2 < input->dim[2]; ++dim2){
        for(uint16_t dim3 = 0; dim3 < input->dim[3]; ++dim3)
            for(uint16_t dim1 = 0; dim1 < input->dim[1]; ++dim1){
                activate[0][dim2][dim3][dim1] = activate[0][dim2][dim3][dim1]*bn[OA][dim1] + bn[OB][dim1];
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
    if(!(input && input->data && input->len == FLOAT_BYTE) || input->dim[0] != 1){
        printf("hardtanh error!\n");
        return NULL;
    }
    float (*activate)[input->dim[2]][input->dim[3]][input->dim[1]] = input->data;
    for(uint16_t dim2 = 0; dim2 < input->dim[2]; ++dim2)
        for(uint16_t dim3 = 0; dim3 < input->dim[3]; ++dim3)
            for(uint16_t dim1 = 0; dim1 < input->dim[1]; ++dim1)
                activate[0][dim2][dim3][dim1] = (activate[0][dim2][dim3][dim1] > 1.0f)?1.0f:((activate[0][dim2][dim3][dim1] < -1.0f)? -1.0f:activate[0][dim2][dim3][dim1]);

    return input;
}


// 位置敏感
data_info_t *avg_pool(data_info_t *input, uint8_t size){
    if(!input || !input->data || !size){
        printf("avg_pool error\n");
        return NULL;
    }
    float (*data)[input->dim[2]][input->dim[3]][input->dim[1]] = input->data;
    float (*output)[input->dim[1]][input->dim[2]/size][input->dim[3]/size] = calloc(input->dim[1]*(input->dim[2]/size)*(input->dim[3]/size), sizeof(float));
    
    for(uint16_t dim2 = 0; dim2 < input->dim[2]; ++dim2)
        for(uint16_t dim3 = 0; dim3 < input->dim[3]; ++dim3)
            for(uint16_t dim1 = 0; dim1 < input->dim[1]; ++dim1)
                output[0][dim1][dim2/size][dim3/size] += data[0][dim2][dim3][dim1];
            
    for(uint16_t dim1 = 0; dim1 < input->dim[1]; ++dim1)
        for(uint16_t dim2 = 0; dim2 < input->dim[2]/size; ++dim2)
            for(uint16_t dim3 = 0; dim3 < input->dim[3]/size; ++dim3)
                output[0][dim1][dim2][dim3] /= (size*size);
    
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

static void print_rgb_pixel(uint16_t dim, uint8_t depth) {
  
    switch(dim){
        case 0:
            printf("\033[48;2;%d;0;0m%02x ", depth, depth);
            break;
        case 1:
            printf("\033[48;2;0;%d;0m%02x ", depth, depth);
            break;
        case 2:
            printf("\033[48;2;0;0;%dm%02x ", depth, depth);
            break;
        default:break;
    }
    
}

void print_RGB_data(data_info_t *jpg_RBG){
    uint8_t (*data)[jpg_RBG->dim[2]][jpg_RBG->dim[3]] = jpg_RBG->data;
    
    for(uint16_t dim2=0; dim2<jpg_RBG->dim[2];dim2++){
        for(uint16_t dim1=0; dim1<2;dim1++){
            for(uint16_t dim3=0; dim3<jpg_RBG->dim[3];dim3++){
                print_rgb_pixel(dim1, data[dim1][dim2][dim3]);
            }
        }
        printf("\033[48;2;255;255;255m  \n");
    }
    for(uint16_t dim2=0; dim2<jpg_RBG->dim[2];dim2++){
        for(uint16_t dim1=2; dim1<4;dim1++){
            if(dim1 < 3)
                for(uint16_t dim3=0; dim3<jpg_RBG->dim[3];dim3++){
                    print_rgb_pixel(dim1, data[dim1][dim2][dim3]);
                }
            else 
                for(uint16_t dim3=0; dim3<jpg_RBG->dim[3];dim3++){
                    printf("\033[48;2;%d;%d;%dm   ",data[0][dim2][dim3], data[1][dim2][dim3], data[2][dim2][dim3]);
            }
        }
        printf("\033[48;2;0;0;0m  \n");
    }

}

void binary_conv_data_trans(data_info_t *Ab, data_info_t* Wb){
    uint8_t (*Adata)[Ab->dim[2]][Ab->dim[3]][Ab->dim[1]/8] = Ab->data;
    uint8_t (*Wdata)[Wb->dim[2]][Wb->dim[3]][Wb->dim[1]/8] = Wb->data;
    printf("uint8_t kernel_L[%d][%d][%d][%d/8] = {\n", Wb->dim[0],Wb->dim[2],Wb->dim[3],Wb->dim[1]);
    for(uint16_t dim0=0;dim0<Wb->dim[0];dim0++){
        printf("\t{\n");
        for(uint16_t dim2=0;dim2<Wb->dim[2];dim2++){
            printf("\t\t{\n");
            for(uint16_t dim3=0;dim3<Wb->dim[3];dim3++){
                printf("\t\t\t{");
                for(uint16_t dim1=0;dim1<(Wb->dim[1]/8);dim1++){
                    printf("0x%02x, ",Wdata[dim0][dim2][dim3][dim1]);
                }
                printf("},\n");
            }
            printf("\t\t},\n");
        }
        printf("\t},\n");
    }
    printf("};\n");

    printf("uint8_t activate_L[%d][%d][%d][%d/8] = {\n", Ab->dim[0],Ab->dim[2],Ab->dim[3],Ab->dim[1]);
    for(uint16_t dim0=0;dim0<Ab->dim[0];dim0++){
        printf("\t{\n");
        for(uint16_t dim2=0;dim2<Ab->dim[2];dim2++){
            printf("\t\t{\n");
            for(uint16_t dim3=0;dim3<Ab->dim[3];dim3++){
                printf("\t\t\t{");
                for(uint16_t dim1=0;dim1<(Ab->dim[1]/8);dim1++){
                    printf("0x%02x, ",Adata[dim0][dim2][dim3][dim1]);
                }
                printf("},\n");
            }
            printf("\t\t},\n");
        }
        printf("\t},\n");
    }
    printf("};\n");

    printf("float kernel_L_f[%d][%d][%d][%d] = {\n", Wb->dim[0],Wb->dim[1],Wb->dim[2],Wb->dim[3]);
    for(uint16_t dim0=0;dim0<Wb->dim[0];dim0++){
        printf("\t{\n");
        for(uint16_t dim1=0;dim1<(Wb->dim[1]);dim1++){
            printf("\t\t{\n");
            for(uint16_t dim2=0;dim2<Wb->dim[2];dim2++){
                printf("\t\t\t{");
                for(uint16_t dim3=0;dim3<Wb->dim[3];dim3++){
                    printf("%d.0, ",(((Wdata[dim0][dim2][dim3][dim1/8]>>(dim1%8))&0x01)?1:-1));
                }
                printf("},\n");
            }
            printf("\t\t},\n");
        }
        printf("\t},\n");
    }
    printf("};\n");

    printf("float activate_L_f[%d][%d][%d][%d] = {\n", Ab->dim[0],Ab->dim[1],Ab->dim[2],Ab->dim[3]);
    for(uint16_t dim0=0;dim0<Ab->dim[0];dim0++){
        printf("\t{\n");
        for(uint16_t dim1=0;dim1<(Ab->dim[1]);dim1++){
            printf("\t\t{\n");
            for(uint16_t dim2=0;dim2<Ab->dim[2];dim2++){
                printf("\t\t\t{");
                for(uint16_t dim3=0;dim3<Ab->dim[3];dim3++){
                    printf("%d.0, ",(((Adata[dim0][dim2][dim3][dim1/8]>>(dim1%8))&0x01)?1:-1));
                }
                printf("},\n");
            }
            printf("\t\t},\n");
        }
        printf("\t},\n");
    }
    printf("};\n");

    printf("dim[%d][%d][%d][%d]\n", Wb->dim[0],Wb->dim[1],Wb->dim[2],Wb->dim[3]);
    printf("conv_kernel_L = torch.tensor([\n");
    for(uint16_t dim0=0;dim0<Wb->dim[0];dim0++){
        printf("\t[\n");
        for(uint16_t dim1=0;dim1<(Wb->dim[1]);dim1++){
            printf("\t\t[\n");
            for(uint16_t dim2=0;dim2<Wb->dim[2];dim2++){
                printf("\t\t\t[");
                for(uint16_t dim3=0;dim3<Wb->dim[3];dim3++){
                    printf("%d.0, ",((Wdata[dim0][dim2][dim3][dim1/8]>>(dim1%8))&0x01?1:-1));
                }
                printf("],\n");
            }
            printf("\t\t],\n");
        }
        printf("\t],\n");
    }
    printf("], dtype=torch.float32)\n");

    printf("mc_input_data_L = torch.tensor([\n");
    for(uint16_t dim0=0;dim0<Ab->dim[0];dim0++){
        printf("\t[\n");
        for(uint16_t dim1=0;dim1<(Ab->dim[1]);dim1++){
            printf("\t\t[\n");
            for(uint16_t dim2=0;dim2<Ab->dim[2];dim2++){
                printf("\t\t\t[");
                for(uint16_t dim3=0;dim3<Ab->dim[3];dim3++){
                    printf("%d.0, ",((Adata[dim0][dim2][dim3][dim1/8]>>(dim1%8))&0x01)?1:-1);
                }
                printf("],\n");
            }
            printf("\t\t],\n");
        }
        printf("\t],\n");
    }
    printf("], dtype=torch.float32)\n");
}

void print_normal_order_data(data_info_t *input){
    float (*data)[input->dim[1]][input->dim[2]][input->dim[3]] = input->data;
    printf("float data1[%d][%d][%d][%d] = {\n",input->dim[0],input->dim[1],input->dim[2],input->dim[3]);
    for(uint16_t dim0=0;dim0<input->dim[0];dim0++){
        printf("\t{\n");
        for(uint16_t dim1=0;dim1<(input->dim[1]);dim1++){
            printf("\t\t{\n");
            for(uint16_t dim2=0;dim2<input->dim[2];dim2++){
                printf("\t\t\t{");
                for(uint16_t dim3=0;dim3<input->dim[3];dim3++){
                    printf("%.3f, ",data[dim0][dim1][dim2][dim3]);
                }
                printf("},\n");
            }
            printf("\t\t},\n");
        }
        printf("\t},\n");
    }
    printf("};\n");
}


void print_data(data_info_t *input){
    float (*data)[input->dim[2]][input->dim[3]][input->dim[1]] = input->data;
    printf("float data1[%d][%d][%d][%d] = {\n",input->dim[0],input->dim[1],input->dim[2],input->dim[3]);
    for(uint16_t dim0=0;dim0<input->dim[0];dim0++){
        printf("    {\n");
        for(uint16_t dim1=0;dim1<(input->dim[1]);dim1++){
            printf("        {\n");
            for(uint16_t dim2=0;dim2<input->dim[2];dim2++){
                printf("            {");
                for(uint16_t dim3=0;dim3<input->dim[3];dim3++){
                    printf("%.3f, ",data[dim0][dim2][dim3][dim1]);
                }
                printf("},\n");
            }
            printf("        },\n");
        }
        printf("    },\n");
    }
    printf("};\n");
}
