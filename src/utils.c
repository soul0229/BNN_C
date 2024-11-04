// #define DEBUG
#include "utils.h"

void print_net_data(Net_List *Net){
    float *data;
    uint8_t *data_b;
    while(Net != NULL){
        switch(*((Data_Type*)(Net->data))){
            case KERNEL:
            CONV_KernelTypeDef *conv = Net->data;
            dbg_print("conv%d.weight",conv->major);
            if(conv->binary < FLOAT_BYTE && conv->binary != UNKNOW){
                data_b = conv->kernel;
                printf("\n\n");
                for(uint16_t dim0=0;dim0<conv->out_ch;dim0++){
                    printf("\n");
                    for(uint16_t dim2=0;dim2<conv->size;dim2++)
                        for(uint16_t dim3=0;dim3<conv->size;dim3++){
                            dbg_print("|");
                            for(uint16_t dim1=0;dim1<(conv->in_ch/8);dim1++){
                                printf("%2x ",*(data_b++));
                            }
                        }
                }
            }
            else if(conv->binary != UNKNOW){
                data = conv->kernel;
                for(uint16_t dim0=0;dim0<conv->out_ch;dim0++){
                    printf("\n\n");
                    for(uint16_t dim1=0;dim1<conv->in_ch;dim1++){
                        printf("\n");
                        for(uint16_t dim2=0;dim2<conv->size;dim2++)
                            for(uint16_t dim3=0;dim3<conv->size;dim3++){
                                printf("%.8f ",*(data++));
                            }
                    }
                }
            }
            else {
                wprint("kernel data type unknow!\n");
            }
            break;
            case BATCHNORM:
            BatchNorm *bn = Net->data;
            dbg_print("\nbn");

            dbg_print("\nweight:");
            data = bn->data[WEIGHT];
            for(uint16_t dim0=0;dim0<bn->size;dim0++){
                printf("%.8f ",*(data++));
            }
            dbg_print("\nbias:");
            data = bn->data[BIAS];
            for(uint16_t dim0=0;dim0<bn->size;dim0++){
                printf("%.8f ",*(data++));
            }
            dbg_print("\nmean:");
            data = bn->data[MEAN];
            for(uint16_t dim0=0;dim0<bn->size;dim0++){
                printf("%.8f ",*(data++));
            }
            dbg_print("\nvar:");
            data = bn->data[VAR];
            for(uint16_t dim0=0;dim0<bn->size;dim0++){
                printf("%.8f ",*(data++));
            }
            break;
            case LINEAR:
            dbg_print("\nLINEAR");
            Linear *linear = Net->data;
            dbg_print("\nweight:");
            data = linear->weight;
            for(uint16_t dim0=0;dim0<linear->size[0];dim0++){
                printf("\n");
                for(uint16_t dim1=0;dim1<linear->size[1];dim1++)
                    printf("%.8f ",*(data++));
            }
            dbg_print("\nbias:");
            data = linear->bias;
            for(uint16_t dim0=0;dim0<linear->size[0];dim0++){
                    printf("%.8f ",*(data++));
            }
            break;
            case LAYER:
            Layer *layer = Net->data;
            dbg_print("\nLAYER%d.%d", layer->major, layer->minor);
            print_net_data((Net_List*)(layer->data));
            break;
            case NET_LIST:
            dbg_print("NET_LIST\n");
            break;
            default:
            dbg_print("default%d",*((Data_Type*)(Net->data)));
            break;
        }
        Net = Net->next;
    }
}

void Net_Binary(Net_List *Net){
    float *data;
    uint8_t *binary;
    while(Net != NULL){
        switch(*((Data_Type*)(Net->data))){
            case KERNEL:
            if(Net->pre==NULL && ((CONV_KernelTypeDef*)(Net->data))->last_stage==NULL)
                break;
            CONV_KernelTypeDef *conv = Net->data;
            if(conv->binary < FLOAT_BYTE){
                wprint("data already binary!\n");
                break;
            }
            data = conv->kernel;
            binary = (uint8_t*)calloc(conv->out_ch*(conv->in_ch/8)*conv->size*conv->size, sizeof(uint8_t));
            for(uint16_t dim0=0;dim0<conv->out_ch;dim0++){
                for(uint16_t dim1=0;dim1<conv->in_ch;dim1++){
                    for(uint16_t dim2=0;dim2<conv->size;dim2++){
                        for(uint16_t dim3=0;dim3<conv->size;dim3++){
                            binary[dim0*(conv->in_ch/8)*conv->size*conv->size+dim2*conv->size*(conv->in_ch/8)+dim3*(conv->in_ch/8)+(dim1/8)] |= \
                            ((data[dim0*conv->size*conv->size*conv->in_ch+dim1*conv->size*conv->size+dim2*conv->size+dim3]>0)?(0x80>>(dim1%8)):((uint8_t)(0x00)));
                        }
                    }
                }
            }
            free(data);
            conv->kernel = binary;
            conv->binary = BINARY;
            dbg_print("\tbinary conv%d.weight\n",conv->major);
            break;
            case BATCHNORM:
            break;
            case LINEAR:
            break;
            case LAYER:
            Layer *layer = Net->data;
            dbg_print("\nLAYER%d.%d\n", layer->major, layer->minor);
            Net_Binary((Net_List*)(layer->data));
            break;
            case NET_LIST:
            break;
            default:
            dbg_print("default%d",*((Data_Type*)(Net->data)));
            break;
        }
        Net = Net->next;
    }
}
static uint64_t data[2] = {NET_START, DATA_START};
void NetStorage(Net_List* Net, uint64_t *net_cnt, uint64_t *data_cnt, FILE* w, char *file_name){
    uint64_t tmp=0;
    if(net_cnt==NULL && data_cnt==NULL && w==NULL)
    {
        net_cnt = &data[0];data_cnt = &data[1];
        w = fopen(file_name, "w");
        if (w == NULL) {
            eprint("打开文件失败!");
            return;
        }
        fseek(w, 0, SEEK_SET);
        fwrite("BinaryNeuralNetwork", sizeof("BinaryNeuralNetwork"), 1, w);
    }else tmp = *net_cnt;
    fseek(w, *net_cnt, SEEK_SET);
    CONV_KernelTypeDef conv;
    BatchNorm bn;
    Linear linear;
    Layer layer;
    while(Net != NULL){
        switch(*((Data_Type*)(Net->data))){
            case KERNEL:
                memset(&conv,0x00,sizeof(CONV_KernelTypeDef));
                memcpy(&conv,Net->data,sizeof(CONV_KernelTypeDef));
                conv.kernel=NULL;
                conv.data_site = *data_cnt;
                conv.last_stage = (void*)(uintptr_t)tmp;
                if(Net->next!=NULL)
                    conv.list = (Net_List*)align16(*net_cnt+sizeof(CONV_KernelTypeDef));
                else conv.list = NULL;
                fseek(w, *net_cnt, SEEK_SET);
                fwrite(&conv, sizeof(CONV_KernelTypeDef), 1, w);
                *net_cnt = align16(*net_cnt+sizeof(CONV_KernelTypeDef));
                fseek(w, *data_cnt, SEEK_SET);
                fwrite(((CONV_KernelTypeDef*)(Net->data))->kernel, \
                ((CONV_KernelTypeDef*)(Net->data))->out_ch*((CONV_KernelTypeDef*)(Net->data))->size*((CONV_KernelTypeDef*)(Net->data))->size*((CONV_KernelTypeDef*)(Net->data))->in_ch*4/(((CONV_KernelTypeDef*)(Net->data))->binary?32:1), \
                1, w);
                *data_cnt = align16(*data_cnt+((CONV_KernelTypeDef*)(Net->data))->out_ch*((CONV_KernelTypeDef*)(Net->data))->size*((CONV_KernelTypeDef*)(Net->data))->size*((CONV_KernelTypeDef*)(Net->data))->in_ch*4/(((CONV_KernelTypeDef*)(Net->data))->binary?32:1));
                break;
            case BATCHNORM:
                memset(&bn,0x00,sizeof(BatchNorm));
                memcpy(&bn,Net->data,sizeof(BatchNorm));
                bn.data[MEAN]=NULL;
                bn.data[VAR]=NULL;
                bn.data[WEIGHT]=NULL;
                bn.data[BIAS]=NULL;
                bn.data_site = *data_cnt;
                bn.last_stage = (void*)(uintptr_t)tmp;
                if(Net->next!=NULL)
                    bn.list = (Net_List*)align16(*net_cnt+sizeof(BatchNorm));
                else bn.list = NULL;
                fseek(w, *net_cnt, SEEK_SET);
                fwrite(&bn, sizeof(BatchNorm), 1, w);
                *net_cnt = align16(*net_cnt+sizeof(BatchNorm));
                fseek(w, *data_cnt, SEEK_SET);
                for(uint8_t cnt=0; cnt<BN_NUM; cnt++)
                    fwrite(((BatchNorm*)(Net->data))->data[cnt], ((BatchNorm*)(Net->data))->size*4, 1, w);
                *data_cnt = align16(*data_cnt+(((BatchNorm*)(Net->data))->size)*4*BN_NUM);
                break;
            case LINEAR:
                memset(&linear,0x00,sizeof(Linear));
                memcpy(&linear,Net->data,sizeof(Linear));
                linear.weight = NULL;
                linear.bias = NULL;
                linear.data_site = *data_cnt;
                linear.last_stage = (void*)(uintptr_t)tmp;
                if(Net->next!=NULL)
                    linear.list = (Net_List*)align16(*net_cnt+sizeof(Linear));
                else linear.list = NULL;
                fseek(w, *net_cnt, SEEK_SET);
                fwrite(&linear, sizeof(Linear), 1, w);
                *net_cnt = align16(*net_cnt+sizeof(Linear));
                fseek(w, *data_cnt, SEEK_SET);
                fwrite(((Linear*)(Net->data))->weight, ((Linear*)(Net->data))->size[0]*((Linear*)(Net->data))->size[1]*4, 1, w);
                fwrite(((Linear*)(Net->data))->bias, ((Linear*)(Net->data))->size[0]*4, 1, w);
                *data_cnt = align16(*data_cnt+(((Linear*)(Net->data))->size[0]+((Linear*)(Net->data))->size[0]*((Linear*)(Net->data))->size[1])*4);
                break;
            case LAYER:
                memset(&layer,0x00,sizeof(Layer));
                memcpy(&layer,Net->data,sizeof(Layer));
                uint32_t layer_site = *net_cnt;
                layer.last_stage = (void*)(uintptr_t)tmp;
                dbg_print("storage LAYER%d.%d\n", layer.major, layer.minor);
                *net_cnt = align16(*net_cnt+sizeof(Layer));
                layer.data = (void*)(uintptr_t)(*net_cnt);
                NetStorage((Net_List*)(((Layer*)(Net->data))->data), net_cnt, data_cnt, w, NULL);
                if(Net->next!=NULL)
                    layer.list = (Net_List*)(uintptr_t)(*net_cnt);
                else layer.list = NULL;
                fseek(w, layer_site, SEEK_SET);
                fwrite(&layer, sizeof(Layer), 1, w);
                break;
            case NET_LIST:
                break;
            default:
                dbg_print("default%d",*((Data_Type*)(Net->data)));
                break;
        }
        Net = Net->next;
    }
    if(tmp == 0){
        if(w!=NULL)
            fclose(w);
        else
            eprint("file point error!\n");
    }
}

Net_List *load_net(char* file_name, uint64_t next, FILE *fp, Layer *last_stage){
    Net_List *net = NULL;
    char buff[64];
    if(fp == NULL){
        fp = fopen(file_name, "r");
        if (fp == NULL) {
            perror("Error opening file");
            return NULL;
        }
        next = NET_START;
        fread(buff, 1, sizeof(buff), fp);
        if(strncmp(buff, "BinaryNeuralNetwork", sizeof("BinaryNeuralNetwork")) != 0)
            return NULL;
    }
    while(next!=0){
        fseek(fp, (uint64_t)next, SEEK_SET);
        fread(buff, 1, sizeof(Data_Type), fp);
        switch(*((Data_Type*)buff)){
            case KERNEL:
                CONV_KernelTypeDef *kernel = net_obj_create(KERNEL);
                fseek(fp, (uint64_t)next, SEEK_SET);
                fread(kernel, 1, PARTSIZE(CONV_KernelTypeDef, list), fp);
                fread(&next, 1, sizeof(next), fp);
                kernel->kernel = (uint8_t*)calloc(kernel->out_ch*kernel->in_ch*kernel->size*kernel->size*4/((kernel->binary==1)?32:1), sizeof(uint8_t));
                fseek(fp, kernel->data_site, SEEK_SET);
                fread(kernel->kernel, 1, kernel->out_ch*kernel->in_ch*kernel->size*kernel->size*4/((kernel->binary==1)?32:1), fp);
                if(net==NULL)net = kernel->list;
                else net_lists_add(net, kernel);
                kernel->last_stage = last_stage;
                break;
            case BATCHNORM:
                BatchNorm *bn = net_obj_create(BATCHNORM);
                fseek(fp, (uint64_t)next, SEEK_SET);
                fread(bn, 1, PARTSIZE(BatchNorm, list), fp);
                fread(&next, 1, sizeof(Net_List*), fp);
                bn->data[MEAN] = (uint8_t*)calloc(bn->size*BN_NUM*4, sizeof(uint8_t));
                bn->data[VAR] = bn->data[MEAN]+bn->size*VAR*4;
                bn->data[WEIGHT] = bn->data[MEAN]+bn->size*WEIGHT*4;
                bn->data[BIAS] = bn->data[MEAN]+bn->size*BIAS*4;
                fseek(fp, bn->data_site, SEEK_SET);
                fread(bn->data[MEAN], 1, bn->size*BN_NUM*4, fp);
                if(net==NULL)net = bn->list;
                else net_lists_add(net, bn);
                bn->last_stage = last_stage;
                break;
            case LINEAR:
                Linear *linear = net_obj_create(LINEAR);
                fseek(fp, (uint64_t)next, SEEK_SET);
                fread(linear, 1, PARTSIZE(Linear, list), fp);
                fread(&next, 1, sizeof(Net_List*), fp);
                linear->weight = (uint8_t*)calloc((linear->size[0]*linear->size[1]+linear->size[0])*4, sizeof(uint8_t));
                linear->bias = linear->weight+(linear->size[0]*linear->size[1])*4;
                fseek(fp, linear->data_site, SEEK_SET);
                fread(linear->weight, 1, (linear->size[0]*linear->size[1]+linear->size[0])*4, fp);
                if(net==NULL)net = linear->list;
                else net_lists_add(net, linear);
                linear->last_stage = last_stage;
                break;
            case LAYER:
                Layer *layer = net_obj_create(LAYER);
                fseek(fp, (uint64_t)next, SEEK_SET);
                fread(layer, 1, PARTSIZE(Layer, list), fp);
                fread(&next, 1, sizeof(Net_List*), fp);
                layer->data = load_net(NULL, (uint64_t)(layer->data), fp, layer);
                if(net==NULL)net = layer->list;
                else net_lists_add(net, layer);
                layer->last_stage = last_stage;
                break;
            case NET_LIST:
            default:break;
        }
    }
    if(last_stage==NULL){
        fclose(fp);
    }
    return net;
}

#include "math.h"
Activate_TypeDef *batchnorm_pcs(Activate_TypeDef *activate, BatchNorm *bn){
    if(activate->ch != bn->size)
        return NULL;
    if(activate->binary == FLOAT_BYTE){
        float *data = activate->active;
        for(uint16_t ch=0; ch<activate->ch; ++ch){
            for(uint16_t num=0; num<activate->size*activate->size; ++num){
                *data = ((*data - (((float*)(bn->data[MEAN])))[ch])/sqrt((((float*)(bn->data[VAR])))[ch] + 1e-5)) * \
                (((float*)(bn->data[WEIGHT])))[ch] + (((float*)(bn->data[BIAS])))[ch];
                data++;
            }
        }
    }
    else if(activate->binary == TWO_BYTE){
        int16_t *data = activate->active;
        int16_t *tmp = data;
        float *out = (float *)calloc(activate->size*activate->size*activate->ch, sizeof(float));
        activate->active = out;
        for(uint16_t ch=0; ch<activate->ch; ++ch){
            for(uint16_t num=0; num<activate->size*activate->size; ++num){
                *out = ((*data - (((float*)(bn->data[MEAN])))[ch])/sqrt((((float*)(bn->data[VAR])))[ch] + 1e-5)) * \
                (((float*)(bn->data[WEIGHT])))[ch] + (((float*)(bn->data[BIAS])))[ch];
                data++;
                out++;
            }
        }
        free(tmp);
    }
    return activate;
}
