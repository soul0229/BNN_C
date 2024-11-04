#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include <getopt.h>

typedef enum command{
    DEFAULT=0,
    RUN,
    PARSE,
    TEST,
    CMD_NUM,
}command_t;

void print_help() {
    printf("Usage: program_name [options]\n");
    printf("Options:\n");
    printf("  -R, --run             Run model inference\n");
    printf("  -P, --parse           Parse json file in list table\n");
    printf("  -T, --test            Internal test use\n");
    printf("  -h, --help            Show help information\n");
    printf("  -o, --output <file>   Output file name\n");
    printf("  -m, --model <model>   The model to run\n");
    printf("  -f, --file <file>     Specify the input file\n");
    printf("  -n, --number <num>    Specify a number\n");
}

int main(int argc, char *argv[]) {
    int opt;
    char *file = NULL, *model = NULL, *output = NULL;
    int number = 0;
    command_t cmd;

    // 定义长选项
    static struct option long_options[] = {
        {"run", no_argument, 0, 'R'},
        {"parse", no_argument, 0, 'P'},
        {"test", no_argument, 0, 'T'},
        {"model", required_argument, 0, 'm'},
        {"output", required_argument, 0, 'o'},
        {"file", required_argument, 0, 'f'},
        {"number", required_argument, 0, 'n'},
        {"help", no_argument,       0, 'h'},
        {0, 0, 0, 0}  // 结束标志
    };

    // 解析命令行参数
    while ((opt = getopt_long(argc, argv, "RPTo:f:m:n:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 'R':cmd = RUN;
                    break;
            case 'P':cmd = PARSE;
                    break;
            case 'T':cmd = TEST;
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

            NetStorage(json_model_parse(file), NULL, NULL, NULL, output);
            break;
        case TEST:break;
        default:
        print_help();
            return -1;
    }

    return 0;
}
