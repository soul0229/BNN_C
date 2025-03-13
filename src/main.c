#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include "common.h"
#include "core.h"
#include "utils.h"
#include "loadNet.h"
#include "net.h"
#include "string.h"

typedef enum command{
    DEFAULT=0,
    RUN,
    PARSE,
    LOAD,
    TEST,
    DECODER,
    INFERENCE,
    PRINT,
    CMD_NUM,
}command_t;

void print_help() {
    printf("Usage: program_name [options]\n");
    printf("Options:\n");
    printf("  -R, --run             Run model inference\n");
    printf("  -P, --parse           Parse json file in list tjpg_RBGle\n");
    printf("  -L, --load            Load .ml file in list tjpg_RBGle\n");
    printf("  -p, --print           Parse json file in list tjpg_RBGle\n");
    printf("  -T, --test            Internal test use\n");
    printf("  -D, --decoder         Internal test use\n");
    printf("  -I, --inference       model inference test\n");
    printf("  -h, --help            Show help information\n");
    printf("  -o, --output <file>   Output file name\n");
    printf("  -m, --model <model>   The model to run\n");
    printf("  -f, --file <file>     Specify the input file\n");
    printf("  -n, --name <string>   Specify a name\n");
}

data_info_t * jpg_decoder_test(char *name){
    data_info_t *jpg_RBG = malloc(sizeof(data_info_t));
            
    jpg_decode(name, jpg_RBG);
    return jpg_RBG;
}

int main(int argc, char *argv[]) {
    int opt;
    char *file = NULL, *model = NULL, *output = NULL, *name = NULL;
    int number = 0;
    command_t cmd;

    // 定义长选项
    static struct option long_options[] = {
        {"run", no_argument, 0, 'R'},
        {"parse", no_argument, 0, 'P'},
        {"load", no_argument, 0, 'L'},
        {"print", no_argument, 0, 'p'},
        {"test", no_argument, 0, 'T'},
        {"decoder", no_argument, 0, 'D'},
        {"inference", no_argument, 0, 'I'},
        {"model", required_argument, 0, 'm'},
        {"output", required_argument, 0, 'o'},
        {"file", required_argument, 0, 'f'},
        {"name", required_argument, 0, 'n'},
        {"help", no_argument,       0, 'h'},
        {0, 0, 0, 0}  // 结束标志
    };

    // 解析命令行参数
    while ((opt = getopt_long(argc, argv, "RPTDLIpo:f:m:n:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 'R':cmd = RUN;
                    break;
            case 'P':cmd = PARSE;
                    break;
            case 'T':cmd = TEST;
                    break;
            case 'D':cmd = DECODER;
                    break;
            case 'I':cmd = INFERENCE;
                    break;
            case 'L':cmd = LOAD;
                    break;
            case 'p':cmd = PRINT;
                    break;
            case 'o':
                output = optarg;
                break;
            case 'f':
                file = optarg;
                break;
            case 'm':
                model = optarg;
                break;
            case 'n':
                name = optarg;
                break;
            case '?':eprint("arguments error!\n");
            case 'h':
            default:
                print_help();
                return 0;
        }
    }

    switch(cmd){
        case RUN:break;
        case PARSE:
            if(!file){
                eprint("file arguments error!\n");
                print_help();
                return -1;
            }
            
            if(!output)
                output = "model.ml";
            json_model_parse_v2(file);
            // NetStorage(json_model_parse(file), NULL, NULL, NULL, output);
            break;
        case TEST:
            extern void BConvTest();
            BConvTest();
            break;
        case DECODER:
            if(!file){
                eprint("file arguments error!\n");
                print_help();
                return -1;
            }
            jpg_decoder_test(file);
            break;
        case INFERENCE:
            if(!file || !model){
                eprint("file arguments error!%d,%d\n",file==0,model==0);
                print_help();
                return -1;
            }
            load_ml_net(model);
            resnet18(jpg_decoder_test(file),file);
            free_net((common_t**)&Net);
            break;
        case PRINT:
            load_ml_net(file);
            printf_appoint_data(name, (common_t*)Net);
            free_net((common_t**)&Net);
            break;
        case LOAD:
            load_ml_net(file);
            printf_net_structure((common_t*)Net);
            printf_appoint_data(name, (common_t*)Net);
            free_net((common_t**)&Net);
            break;
        default:
        print_help();
            return -1;
    }

    return 0;
}
