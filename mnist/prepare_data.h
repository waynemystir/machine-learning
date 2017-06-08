#include <stdint.h>

#include "common.h"

#define TRAIN_IMAGES_FILENAME "train-images-idx3-ubyte"
#define TRAIN_LABELS_FILENAME "train-labels-idx1-ubyte"
#define TEST_IMAGES_FILENAME "t10k-images-idx3-ubyte"
#define TEST_LABELS_FILENAME "t10k-labels-idx1-ubyte"

int images_header(char *filename, uint32_t *magic_number, uint32_t *num_images, uint32_t *num_rows, uint32_t *num_cols);
int get_images(char *filename, uint32_t num_images, uint32_t num_rows, uint32_t num_cols, linked_list_t **pixels);
