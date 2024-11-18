#ifndef __COMMOM_H__
#define __COMMOM_H__

#include <stdint.h>
#include <stdbool.h>
#include "config.h"

#define align16(x)              (((x)+(16-1))-(((x)+(16-1))%16))
#define alignidx(size, idx)     (((size)+(idx-1))-(((size)+(idx-1))%idx))
#define PARTSIZE(type,mem) ((unsigned long)(&(((type *)0)->mem)))

#define RGB_PRINT
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

#ifndef DATA_LEN
    #define DATA_LEN 8
#endif
#if (DATA_LEN == 8)
    #define intx_t int8_t
    #define ONE     0x80
    #define ZERO    0x00
#elif (DATA_LEN == 16)
    #define intx_t uint16_t
    #define ONE     0x8000
    #define ZERO    0x0000
#elif (DATA_LEN == 32)
    #define intx_t uint32_t
    #define ONE     0x80000000
    #define ZERO    0x00000000
#elif (DATA_LEN == 64)
    #define intx_t uint64_t
    #define ONE     0x8000000000000000
    #define ZERO    0x0000000000000000
#else
    #error "DATA_LEN error!"
#endif

enum data_len{
    UNKNOW = 0,
    BINARY,
    TWO_BYTE,
    FLOAT_BYTE,
};
typedef enum data_len Data_Len;

#endif