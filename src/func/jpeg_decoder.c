#include <stdint.h>
#include <stdio.h>
#include <jpeglib.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>
#include "core.h"

struct error_mgr {
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

void error_exit(j_common_ptr cinfo) {
    struct error_mgr *myerr = (struct error_mgr *)cinfo->err;
    (*cinfo->err->output_message)(cinfo);
    longjmp(myerr->setjmp_buffer, 1);
}

int jpg_decode(char* name, data_info_t* data) {
    struct jpeg_decompress_struct cinfo;
    struct error_mgr jerr;
    FILE *infile;
    JSAMPARRAY buffer;
    unsigned char *r_channel, *g_channel, *b_channel;

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = error_exit;
    
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return 1;
    }

    jpeg_create_decompress(&cinfo);

    // 打开输入文件
    if ((infile = fopen(name, "rb")) == NULL) {
        fprintf(stderr, "Can't open %s\n", name);
        return 1;
    }
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    int row_stride = cinfo.output_width * cinfo.output_components;
    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);

    // 处理每一行数据
    
    uint8_t (*out)[cinfo.output_width][cinfo.output_height] = malloc(cinfo.output_components*cinfo.output_width*cinfo.output_width*sizeof(uint8_t));
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        unsigned char *row = buffer[0];

        // 分离通道
        for (int x = 0; x < cinfo.output_width; x++) {
            out[0][cinfo.output_scanline-1][x] = row[x*3];
            out[1][cinfo.output_scanline-1][x] = row[x*3+1];
            out[2][cinfo.output_scanline-1][x] = row[x*3+2];
        }
    }
    data->dim[0] = 1;
    data->dim[1] = cinfo.output_components;
    data->dim[2] = cinfo.output_height;
    data->dim[3] = cinfo.output_width;
    data->len = ONE_BYTE;
    data->data = out;

    // 清理资源
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);

    printf("Decoding completed.\n");
    return 0;
}