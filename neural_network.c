#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "common.h"
#include "maths.h"
#include "matrix.h"
#include "prepare_data.h"

typedef struct neural_network {
	int num_neurons[3];
	size_t num_weights;
	matrix *weights[2];
	matrix *biases[2];
} neural_network_t;

void neural_net_init(neural_network_t **neural_net, int num_neurons[3]) {
	if (!neural_net) return;

	neural_network_t *nn = malloc(sizeof(neural_network_t));
	if (!nn) return;
	memset(nn, '\0', sizeof(neural_network_t));

	*neural_net = nn;
	memcpy(nn->num_neurons, num_neurons, 3 * sizeof(int));

	int i, j;
	matrix *wts1, *wts2, *bias1, *bias2;

	matrix_init(&wts1, num_neurons[1], num_neurons[0], NULL);
	matrix_init(&wts2, num_neurons[2], num_neurons[1], NULL);
	nn->weights[0] = wts1;
	nn->weights[1] = wts2;

	matrix_elementwise_func_3(wts1, gaussrand);
	matrix_elementwise_func_3(wts2, gaussrand);

	matrix_init(&bias1, num_neurons[1], 1, NULL);
	matrix_init(&bias2, num_neurons[2], 1, NULL);
	nn->biases[0] = bias1;
	nn->biases[1] = bias2;

	matrix_elementwise_func_3(bias1, gaussrand);
	matrix_elementwise_func_3(bias2, gaussrand);
}

void sigmoid(double z, double *ret_value) {
	*ret_value = 1 / (1 + exp(-z));
//	*ret_value = 10 + z;
}

void sigmoid_arr(double *arr, size_t num, double **ret_array) {
	double *ret_arr = calloc(num, sizeof(double));
	double *iter = ret_arr;
	for (int i = 0; i < num; i++)
		sigmoid(arr[i], iter++);
	*ret_array = ret_arr;
}

void feedforward(neural_network_t *nn, matrix *a, matrix **output) {
	matrix *mp, *ms;
	for (int i = 0; i < 2; i++) {
		matrix_product(nn->weights[i], a, &mp);
		matrix_sum(mp, nn->biases[i], &ms);
		matrix_elementwise_func_2(ms, sigmoid);
		a = ms;
		mp = NULL;
		ms = NULL;
	}
	*output = a;
}

int main() {
	printf("network-0 (%lu)(%lu)(%lu)\n", sizeof(unsigned int), sizeof(uint32_t), sizeof(double));

        uint32_t magic_number, num_train_images, num_test_images, num_rows, num_cols;
        int rih = images_header(TRAIN_IMAGES_FILENAME, &magic_number, &num_train_images, &num_rows, &num_cols);
	if (rih == -1) {
		printf("There was a problem collecting training images.\n");
		return -1;
	}

        rih = images_header(TEST_IMAGES_FILENAME, &magic_number, &num_test_images, &num_rows, &num_cols);
	if (rih == -1) {
		printf("There was a problem collecting test images.\n");
		return -1;
	}

	linked_list_t *train_pixels, *test_pixels;
	get_images(TRAIN_IMAGES_FILENAME, num_train_images, num_rows, num_cols, &train_pixels);
	get_images(TEST_IMAGES_FILENAME, num_test_images, num_rows, num_cols, &test_pixels);
	printf("We got the images (%lu)(%lu)...\n", train_pixels->count, test_pixels->count);

	linked_list_node_t *n = list_get(train_pixels, 0);
	printf("Lets print matrix (%d)(%s)(%s)\n", 0, n ? "GOOD" : "BAD", n->data ? "GOOD" : "BAD");
	matrix_print(n->data, 8, 1);

	n = list_get(train_pixels, 3);
	printf("Lets print matrix (%d)(%s)(%s)\n", 3, n ? "GOOD" : "BAD", n->data ? "GOOD" : "BAD");
	matrix_print(n->data, 8, 1);

	n = list_get(train_pixels, 7);
	printf("Lets print matrix (%d)(%s)(%s)\n", 7, n ? "GOOD" : "BAD", n->data ? "GOOD" : "BAD");
	matrix_print(n->data, 8, 1);

	n = list_get(train_pixels, 13);
	printf("Lets print matrix (%d)(%s)(%s)\n", 13, n ? "GOOD" : "BAD", n->data ? "GOOD" : "BAD");
	matrix_print(n->data, 8, 1);

	neural_network_t *w;
	int sizes[] = {273, 397, 941};
	neural_net_init(&w, sizes);
	for (int i = 0; i < 3; i++) printf("sz(%d)(%d)\n", i, w->num_neurons[i]);

//	printf("XXXXX (%f)(%f)(%f)(%f)\n", exp(-6), sigmoid(6), exp(6), sigmoid(-6));

	double arr[4] = { -4, 8, 13, 31 };
	double *ra;
	sigmoid_arr(arr, 4, &ra);
	for (int j = 0; j < 4; j++) printf("DDD (%d)(%f)\n", j, ra[j]);

	return 0;
}
