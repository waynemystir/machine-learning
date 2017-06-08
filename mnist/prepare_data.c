#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "prepare_data.h"
#include "matrix.h"

#define FOLDER "mnist"
#define IMAGEOFFSET 16
#define LABELOFFSET 8

// Byte swap unsigned int
// curtesy of https://stackoverflow.com/questions/2182002/convert-big-endian-to-little-endian-in-c-without-using-provided-func
uint32_t swap_uint32( uint32_t val ) {
	val = ((val << 8) & 0xFF00FF00 ) | ((val >> 8) & 0xFF00FF ); 
	return (val << 16) | (val >> 16);
}

int images_header(char *filename, uint32_t *magic_number, uint32_t *num_images, uint32_t *num_rows, uint32_t *num_cols) {
	char path[256] = {0};
	sprintf(path, "%s/%s", FOLDER, filename);
	FILE *f = fopen(path, "rb");
	if (!f) {
		printf("Could not open file (%s).\n", path);
		return -1;
	}

	fread(magic_number, sizeof(uint32_t), 1, f);
	*magic_number = swap_uint32(*magic_number);
	if (*magic_number != 2051) {
		printf("Magic number (%u) from file (%s) is wrong. It should be 2051.\n", *magic_number, path);
		return -1;
	}

	fread(num_images, sizeof(uint32_t), 1, f);
	fread(num_rows, sizeof(uint32_t), 1, f);
	fread(num_cols, sizeof(uint32_t), 1, f);
	*num_images = swap_uint32(*num_images);
	*num_rows = swap_uint32(*num_rows);
	*num_cols = swap_uint32(*num_cols);
	if (*num_rows != 28 || *num_cols != 28) {
		printf("Images file (%s): dimensions (%u x %u) not as expected (28 x 28).\n", path, *num_rows, *num_cols);
		return -1;
	}

	printf("images_header (%s)(%u)(%u)(%u)(%u)\n", path, *magic_number, *num_images, *num_rows, *num_cols);
	fclose(f);
	return 0;
}

int get_images(char *filename, uint32_t num_images, uint32_t num_rows, uint32_t num_cols, linked_list_t **pixels) {
        char path[256] = {0};
        sprintf(path, "%s/%s", FOLDER, filename);
        FILE *f = fopen(path, "rb");
        if (!f) {
                printf("Could not open file (%s).\n", path);
                return -1;
        }

	int fs = fseek(f, IMAGEOFFSET, SEEK_SET);
	if (fs == -1) {
		printf("A problem occurred while seeking (%s)(%d).\n", path, IMAGEOFFSET);
		return -1;
	}

	linked_list_t *pxs = malloc(SZ_LL);
	if (!pxs) return -1;
	memset(pxs, '\0', SZ_LL);
	if (pixels) *pixels = pxs;
	unsigned char pixel;

	for (int i = 0; i < num_images; i++) {
                matrix *m;
                matrix_init(&m, num_rows * num_cols, 1, NULL);
                for (int j = 0; j < num_rows * num_cols; j++) {
			fread(&pixel, sizeof(unsigned char), 1, f);
                        double value = (double)(pixel) / 256;
                        matrix_set(m, j, 0, value);
                }
		list_add_tail(pxs, m, NULL);
        }

	fclose(f);
	return 0;
}

//int main() {
//	printf("prepare_data-0 (%lu)(%lu)\n", sizeof(uint32_t), sizeof(unsigned int));
//
//	FILE *f;
//	uint32_t magic_number;
//	uint32_t num_images;
//	uint32_t num_rows;
//	uint32_t num_cols;
//	images_header(TRAIN_IMAGES_FILENAME, &magic_number, &num_images, &num_rows, &num_cols);
//	images_header(TEST_IMAGES_FILENAME, &magic_number, &num_images, &num_rows, &num_cols);
//
//	unsigned char pixel;
//	unsigned char arr[num_rows][num_cols];
//	for (int r = 0; r < num_images; r++) {
//		for (int i = 0; i < num_rows; i++) {
//			for (int j = 0; j < num_cols; j++) {
//				num_read = fread(&pixel, sizeof(unsigned char), 1, f);
//				arr[i][j] = pixel;
////				float w = pixel;
////				printf("BBB (%u)(%f)", arr[i][j], w);
//				if (r == 13131) printf("B(%u)(%f)", arr[i][j], (float)arr[i][j]/256);
//			}
//		}
//	}
//	printf("\n");
//
//	return 0;
//}
