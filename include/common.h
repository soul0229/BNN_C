#ifndef _COMMOM_H
#define _COMMOM_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define DATA_LEN 8
#define USE_PADDING_ZERO
#define SIZE 3
#define RGB_DBG
#define NET_START (64)
#define DATA_START (4096*5)

#if defined(RGB_DBG)
#define clrprint(clr,str,...) printf("\033[%dm" str "\033[0m",clr,##__VA_ARGS__)
#else
#define clrprint(clr,str,...)
#endif
#define eprint(str,...) clrprint(31,str,##__VA_ARGS__)
#define iprint(str,...) clrprint(32,str,##__VA_ARGS__)
#define wprint(str,...) clrprint(33,str,##__VA_ARGS__)

#ifdef DEBUG
    #define dbg_print(str,...) clrprint(32,str,##__VA_ARGS__)
#else
    #define dbg_print(str,...)
#endif

#endif