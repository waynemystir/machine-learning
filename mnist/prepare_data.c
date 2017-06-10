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

	uint32_t expected_magic_number = 2051;
	fread(magic_number, sizeof(uint32_t), 1, f);
	*magic_number = swap_uint32(*magic_number);
	if (*magic_number != expected_magic_number) {
		printf("Magic number (%u) from file (%s) is wrong. It should be (%d).\n", *magic_number, path, expected_magic_number);
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

int labels_header(char *filename, uint32_t *magic_number, uint32_t *num_labels) {
	char path[256] = {0};
	sprintf(path, "%s/%s", FOLDER, filename);
	FILE *f = fopen(path, "rb");
	if (!f) {
		printf("Could not open file (%s).\n", path);
		return -1;
	}

	uint32_t expected_magic_number = 2049;
	fread(magic_number, sizeof(uint32_t), 1, f);
	*magic_number = swap_uint32(*magic_number);
	if (*magic_number != expected_magic_number) {
		printf("Magic number (%u) from file (%s) is wrong. It should be (%d).\n", *magic_number, path, expected_magic_number);
		return -1;
	}

	fread(num_labels, sizeof(uint32_t), 1, f);
	*num_labels = swap_uint32(*num_labels);

	printf("labels_header (%s)(%u)(%u)\n", path, *magic_number, *num_labels);
	fclose(f);
	return 0;
}

int get_images(char *filename,
	uint32_t num_images,
	uint32_t num_rows,
	uint32_t num_cols,
	size_t num_validation_pixels,
	list_t **train_pixels,
	list_t **validation_pixels) {

	if (num_validation_pixels > num_images) {
		printf("Error: num_validation_pixels > num_images\n");
		return -1;
	}

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

	size_t num_train_pixels = num_images - num_validation_pixels;
	list_t *tr_pxs, *vd_pxs = NULL;
	list_init(&tr_pxs, num_train_pixels, NULL, (free_fp)matrix_free);
	if (!tr_pxs) return -1;
	if (train_pixels) *train_pixels = tr_pxs;

	if (num_validation_pixels > 0 && validation_pixels) {
		list_init(&vd_pxs, num_validation_pixels, NULL, (free_fp)matrix_free);
		if (!vd_pxs) return -1;
		*validation_pixels = vd_pxs;
	}

	unsigned char pixel;

	for (int i = 0; i < num_images; i++) {
		matrix_t *m;
		matrix_init(&m, num_rows * num_cols, 1, NULL);
		for (int j = 0; j < num_rows * num_cols; j++) {
			fread(&pixel, sizeof(unsigned char), 1, f);
			double value = (double)(pixel) / 256;
			matrix_set(m, j, 0, value);
		}
		if (i < num_train_pixels)
			list_set(tr_pxs, i, m);
		else if (vd_pxs)
			list_set(vd_pxs, i - num_train_pixels, m);
		else free(m);
	}

	fclose(f);
	return 0;
}

int get_labels(char *filename,
	uint32_t num_labels,
	size_t num_validation_labels,
	list_t **train_labels,
	list_t **validation_labels) {

	if (num_validation_labels > num_labels) {
		printf("Error: num_validation_labels > num_labels\n");
		return -1;
	}

	char path[256] = {0};
	sprintf(path, "%s/%s", FOLDER, filename);
	FILE *f = fopen(path, "rb");
	if (!f) {
		printf("Could not open file (%s).\n", path);
		return -1;
	}

	int fs = fseek(f, LABELOFFSET, SEEK_SET);
	if (fs == -1) {
		printf("A problem occurred while seeking (%s)(%d).\n", path, LABELOFFSET);
		return -1;
	}

	size_t num_train_labels = num_labels - num_validation_labels;
	list_t *tr_lbs, *vd_lbs = NULL;
	list_init(&tr_lbs, num_train_labels, NULL, (free_fp)matrix_free);
	if (!tr_lbs) return -1;
	if (train_labels) *train_labels = tr_lbs;

	if (num_validation_labels > 0 && validation_labels) {
		list_init(&vd_lbs, num_validation_labels, NULL, (free_fp)matrix_free);
		if (!vd_lbs) return -1;
		*validation_labels = vd_lbs;
	}

	unsigned char label;

	for (int i = 0; i < num_labels; i++) {
		matrix_t *m;
		matrix_zero_init(&m, 10, 1);
		fread(&label, sizeof(unsigned char), 1, f);
		size_t lp = label;
		matrix_set(m, lp, 0, 1);

		if (i < num_train_labels)
			list_set(tr_lbs, i, m);
		else if (vd_lbs)
			list_set(vd_lbs, i - num_train_labels, m);
		else free(m);
	}

	fclose(f);
	return 0;
}

