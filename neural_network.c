#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "neural_network.h"
#include "maths.h"
#include "prepare_data.h"

struct neural_network {
	int num_neurons[3];
	size_t num_weights;
	matrix_t *weights[2];
	matrix_t *biases[2];
	list_t *training_data;
	list_t *training_labels;
	list_t *validation_data;
	list_t *validation_labels;
	list_t *test_data;
	list_t *test_labels;
};

void neural_net_init(neural_network_t **neural_net, int num_neurons[3]) {
	if (!neural_net) return;

	neural_network_t *nn = malloc(sizeof(neural_network_t));
	if (!nn) return;
	memset(nn, '\0', sizeof(neural_network_t));

	*neural_net = nn;
	memcpy(nn->num_neurons, num_neurons, 3 * sizeof(int));

	matrix_t *wts1, *wts2, *bias1, *bias2;

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

void feedforward(neural_network_t *nn, matrix_t *a, matrix_t **output) {
	matrix_t *mp, *ms;
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

void sgd(list_t *training_data, size_t epochs, size_t mini_batch_size, double eta, list_t *test_data) {
	if (!training_data) return;

	for (size_t i = 0; i < epochs; i++) {
		list_shuffle(training_data);
		for (size_t j = 0; j < list_len(training_data); j+= mini_batch_size)
			update_mini_batch(training_data, j, j + mini_batch_size, eta);
		if (test_data)
			printf("Epoch (%lu): (%lu) / (%lu) \n", i + 1, evaluate(test_data), list_len(test_data));
		else
			printf("Epoch (%lu) complete\n", i + 1);
	}
}

void update_mini_batch(list_t *training_data, size_t start, size_t end, double eta) {

}

size_t evaluate(list_t *test_data) {
	return 1;
}

int main() {
	printf("network-0 (%lu)(%lu)(%lu)\n", sizeof(unsigned int), sizeof(uint32_t), sizeof(double));

	uint32_t magic_number_images, magic_number_labels, num_train_images, num_train_labels, num_test_images, num_test_labels, num_rows, num_cols;
	int rih = images_header(TRAIN_IMAGES_FILENAME, &magic_number_images, &num_train_images, &num_rows, &num_cols);
	if (rih == -1) {
		printf("There was a problem collecting training images.\n");
		return -1;
	}

	rih = images_header(TEST_IMAGES_FILENAME, &magic_number_images, &num_test_images, &num_rows, &num_cols);
	if (rih == -1) {
		printf("There was a problem collecting test images.\n");
		return -1;
	}

	rih = labels_header(TRAIN_LABELS_FILENAME, &magic_number_labels, &num_train_labels);
	if (rih == -1) {
		printf("There was a problem collecting training labels.\n");
		return -1;
	}

	rih = labels_header(TEST_LABELS_FILENAME, &magic_number_labels, &num_test_labels);
	if (rih == -1) {
		printf("There was a problem collecting testing labels.\n");
		return -1;
	}

	neural_network_t *w;
	int sizes[] = {273, 397, 941};
	neural_net_init(&w, sizes);
	for (int i = 0; i < 3; i++) printf("sz(%d)(%d)\n", i, w->num_neurons[i]);

	size_t num_validation = 10 * 1000;
	get_images(TRAIN_IMAGES_FILENAME, num_train_images, num_rows, num_cols, num_validation, &w->training_data, &w->validation_data);
	get_images(TEST_IMAGES_FILENAME, num_test_images, num_rows, num_cols, 0, &w->test_data, NULL);
	printf("We got the images (%lu)(%lu)(%lu)...\n", list_len(w->training_data), list_len(w->validation_data), list_len(w->test_data));
	get_labels(TRAIN_LABELS_FILENAME, num_train_labels, num_validation, &w->training_labels, &w->validation_labels);
	get_labels(TEST_LABELS_FILENAME, num_test_labels, 0, &w->test_labels, NULL);
	printf("We got the labels (%lu)(%lu)(%lu)...\n", list_len(w->training_labels), list_len(w->validation_labels), list_len(w->test_labels));

	size_t wes = 43432;
	matrix_t *m = list_get(w->training_data, wes);
	printf("Lets print training matrix (%lu)(%s)\n", wes, m ? "GOOD" : "BAD");
//	matrix_print(m, 8, 1);
	m = list_get(w->training_labels, wes);
	printf("Lets print training label (%lu)(%s)\n", wes, m ? "GOOD" : "BAD");
	matrix_print(m, 8, 1);

	wes = 7777;
	m = list_get(w->validation_data, wes);
	printf("Lets print validation matrix (%lu)(%s)\n", wes, m ? "GOOD" : "BAD");
//	matrix_print(m, 8, 1);
	m = list_get(w->validation_labels, wes);
	printf("Lets print validation label (%lu)(%s)\n", wes, m ? "GOOD" : "BAD");
	matrix_print(m, 8, 1);

	wes = 6666;
	m = list_get(w->test_data, wes);
	printf("Lets print test matrix (%lu)(%s)\n", wes, m ? "GOOD" : "BAD");
	matrix_print(m, 8, 1);
	m = list_get(w->test_labels, wes);
	printf("Lets print test label (%lu)(%s)\n", wes, m ? "GOOD" : "BAD");
	matrix_print(m, 8, 1);

//	printf("XXXXX (%f)(%f)(%f)(%f)\n", exp(-6), sigmoid(6), exp(6), sigmoid(-6));

	double arr[4] = { -4, 8, 13, 31 };
	double *ra;
	sigmoid_arr(arr, 4, &ra);
	for (int j = 0; j < 4; j++) printf("DDD (%d)(%f)\n", j, ra[j]);

	return 0;
}
