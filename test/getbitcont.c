#include <stdint.h>
#include <stdio.h>

void bitcont_generate(){
    uint16_t i, j;
    int16_t cnt = 0;
    for(i=0;i<256;++i){
        if(i%16==0)printf("\n");
        cnt = 0;
        for(j=0;j<8;++j){
            if(((i>>j)&0x01) == 1)
            cnt++;
            else cnt--;
        }
        printf("%-2d, ",cnt);
        
    }
}