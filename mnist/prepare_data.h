#include <stdint.h>

#include "common.h"

#define TRAIN_IMAGES_FILENAME "train-images-idx3-ubyte"
#define TRAIN_LABELS_FILENAME "train-labels-idx1-ubyte"
#define TEST_IMAGES_FILENAME "t10k-images-idx3-ubyte"
#define TEST_LABELS_FILENAME "t10k-labels-idx1-ubyte"

int images_header(char *filename, uint32_t *magic_number, uint32_t *num_images, uint32_t *num_rows, uint32_t *num_cols);
int labels_header(char *filename, uint32_t *magic_number, uint32_t *num_labels);
int get_images(char *filename,
	uint32_t num_images,
	uint32_t num_rows,
	uint32_t num_cols,
	size_t num_validation_pixels,
	linked_list_t **train_pixels,
	linked_list_t **validation_pixels);
int get_labels(char *filename,
	uint32_t num_labels,
	size_t num_validation_labels,
	linked_list_t **train_labels,
	linked_list_t **validation_labels);
