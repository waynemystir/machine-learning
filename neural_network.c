#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "neural_network.h"
#include "maths.h"
#include "prepare_data.h"

#define NUMBER_OF_LAYERS 3 // including input, hidden, output

struct neural_network {
	int num_neurons[NUMBER_OF_LAYERS];
	list_t *weights;
	list_t *biases;
	list_t *training_data;
	list_t *training_results;
	list_t *validation_data;
	list_t *validation_results;
	list_t *test_data;
	list_t *test_results;
};

void neural_net_init(neural_network_t **neural_net, int num_neurons[3]) {
	if (!neural_net) return;

	neural_network_t *nn = malloc(sizeof(neural_network_t));
	if (!nn) return;
	memset(nn, '\0', sizeof(neural_network_t));

	*neural_net = nn;
	memcpy(nn->num_neurons, num_neurons, 3 * sizeof(int));

	list_init(&nn->weights, NUMBER_OF_LAYERS - 1, NULL, (free_fp)matrix_free);
	list_init(&nn->biases, NUMBER_OF_LAYERS - 1, NULL, (free_fp)matrix_free);

	matrix_t *wts1, *wts2, *bias1, *bias2;

	matrix_init(&wts1, num_neurons[1], num_neurons[0], NULL);
	matrix_init(&wts2, num_neurons[2], num_neurons[1], NULL);
	list_set(nn->weights, 0, wts1);
	list_set(nn->weights, 1, wts2);

	matrix_elementwise_func_3(wts1, gaussrand);
	matrix_elementwise_func_3(wts2, gaussrand);

	matrix_init(&bias1, num_neurons[1], 1, NULL);
	matrix_init(&bias2, num_neurons[2], 1, NULL);
	list_set(nn->biases, 0, bias1);
	list_set(nn->biases, 1, bias2);

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
	for (int i = 0; i < NUMBER_OF_LAYERS - 1; i++) {
		matrix_product(list_get(nn->weights, i), a, &mp);
		matrix_sum(mp, list_get(nn->biases, i), &ms);
		matrix_elementwise_func_2(ms, sigmoid);
		a = ms;
		mp = NULL;
		ms = NULL;
	}
	*output = a;
}

void sgd(neural_network_t *nn, size_t epochs, size_t mini_batch_size, double eta) {
	if (!nn || !nn->training_data) return;

	for (size_t i = 0; i < epochs; i++) {
		list_shuffle(nn->training_data);
		for (size_t j = 0; j < list_len(nn->training_data); j+= mini_batch_size)
			update_mini_batch(nn, j, j + mini_batch_size, eta);
		if (nn->test_data)
			printf("Epoch (%lu): (%lu) / (%lu) \n", i + 1, evaluate(nn->test_data), list_len(nn->test_data));
		else
			printf("Epoch (%lu) complete\n", i + 1);
	}
}

void update_mini_batch(neural_network_t *nn, size_t start, size_t end, double eta) {
	if (start >= end || !nn || !nn->biases || !nn->weights || list_len(nn->biases) != list_len(nn->weights)) return;
	list_t *nabla_b, *nabla_w;
	list_init(&nabla_b, list_len(nn->biases), NULL, (free_fp)matrix_free);
	list_init(&nabla_w, list_len(nn->weights), NULL, (free_fp)matrix_free);

	for (size_t i = 0; i < list_len(nn->biases); i++) {
		matrix_t *mb, *mw;
		matrix_zero_init(&mb, matrix_num_rows(list_get(nn->biases, i)), matrix_num_cols(list_get(nn->biases, i)));
		matrix_zero_init(&mw, matrix_num_rows(list_get(nn->weights, i)), matrix_num_cols(list_get(nn->weights, i)));
		list_set(nabla_b, i, mb);
		list_set(nabla_w, i, mw);
	}

	for (size_t i = start; i < end; i++) {
		matrix_t *x = list_get(nn->training_data, i);
		matrix_t *y = list_get(nn->training_results, i);
		list_t *delta_nabla_b, *delta_nabla_w;
		backprop(nn, x, y, &delta_nabla_b, &delta_nabla_w);

		for (size_t j = 0; j < list_len(nabla_b); j++) {
			matrix_t *nbsum, *wbsum;

			matrix_t *nb = list_get(nabla_b, j);
			matrix_t *dnb = list_get(delta_nabla_b, j);
			matrix_sum(nb, dnb, &nbsum);
			list_set(nabla_b, j, nbsum);

			matrix_t *wb = list_get(nabla_w, j);
			matrix_t *dwb = list_get(delta_nabla_w, j);
			matrix_sum(wb, dwb, &wbsum);
			list_set(nabla_w, j, wbsum);
		}
	}

	for (size_t i = 0; i < list_len(nn->biases); i++) {
		matrix_t *sw, *sb;
		matrix_t *w = list_get(nn->weights, i);
		matrix_t *nw = list_get(nabla_w, i);
		matrix_t *b = list_get(nn->biases, i);
		matrix_t *nb = list_get(nabla_b, i);
		double scalar = eta * (start - end);

		matrix_product_scalar(nw, scalar);
		matrix_product_scalar(nb, scalar);
		matrix_sum(w, nw, &sw);
		matrix_sum(b, nb, &sb);

		list_set(nn->weights, i, sw);
		list_set(nn->biases, i, sb);
	}
}

void backprop(neural_network_t *nn, matrix_t *x, matrix_t *y, list_t **nabla_b, list_t **nabla_w) {

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
	get_labels(TRAIN_LABELS_FILENAME, num_train_labels, num_validation, &w->training_results, &w->validation_results);
	get_labels(TEST_LABELS_FILENAME, num_test_labels, 0, &w->test_results, NULL);
	printf("We got the labels (%lu)(%lu)(%lu)...\n", list_len(w->training_results), list_len(w->validation_results), list_len(w->test_results));

	size_t wes = 43432;
	matrix_t *m = list_get(w->training_data, wes);
	printf("Lets print training matrix (%lu)(%s)\n", wes, m ? "GOOD" : "BAD");
//	matrix_print(m, 8, 1);
	m = list_get(w->training_results, wes);
	printf("Lets print training label (%lu)(%s)\n", wes, m ? "GOOD" : "BAD");
	matrix_print(m, 8, 1);

	wes = 7777;
	m = list_get(w->validation_data, wes);
	printf("Lets print validation matrix (%lu)(%s)\n", wes, m ? "GOOD" : "BAD");
//	matrix_print(m, 8, 1);
	m = list_get(w->validation_results, wes);
	printf("Lets print validation label (%lu)(%s)\n", wes, m ? "GOOD" : "BAD");
	matrix_print(m, 8, 1);

	wes = 6666;
	m = list_get(w->test_data, wes);
	printf("Lets print test matrix (%lu)(%s)\n", wes, m ? "GOOD" : "BAD");
	matrix_print(m, 8, 1);
	m = list_get(w->test_results, wes);
	printf("Lets print test label (%lu)(%s)\n", wes, m ? "GOOD" : "BAD");
	matrix_print(m, 8, 1);

//	printf("XXXXX (%f)(%f)(%f)(%f)\n", exp(-6), sigmoid(6), exp(6), sigmoid(-6));

	double arr[4] = { -4, 8, 13, 31 };
	double *ra;
	sigmoid_arr(arr, 4, &ra);
	for (int j = 0; j < 4; j++) printf("DDD (%d)(%f)\n", j, ra[j]);

	return 0;
}
