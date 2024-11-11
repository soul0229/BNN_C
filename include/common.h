#ifndef __COMMOM_H__
#define __COMMOM_H__

#include <stdint.h>
#include <stdbool.h>

#define align16(x)    (((x)+(16-1))-(((x)+(16-1))%16))
#define PARTSIZE(type,mem) ((unsigned long)(&(((type *)0)->mem)))

#if defined(RGB_PRINT)
#define clrprint(clr,str,...) printf("\033[%dm" str "\033[0m",clr,##__VA_ARGS__)
#else
#define clrprint(clr,str,...) printf(str,##__VA_ARGS__)
#endif

#define eprint(str,...) clrprint(31,str,##__VA_ARGS__)
#define iprint(str,...) clrprint(32,str,##__VA_ARGS__)
#define wprint(str,...) clrprint(33,str,##__VA_ARGS__)

#ifdef DEBUG
    #define dbg_print(str,...) clrprint(32,str,##__VA_ARGS__)
#else
    #define dbg_print(str,...)
#endif

enum data_len{
    UNKNOW = 0,
    BINARY,
    TWO_BYTE,
    FLOAT_BYTE,
};
typedef enum data_len Data_Len;

#endif