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

void neural_network_free(neural_network_t *nn) {
	if (!nn) return;

	list_free(nn->weights);
	list_free(nn->biases);
	list_free(nn->training_data);
	list_free(nn->training_results);
	list_free(nn->validation_data);
	list_free(nn->validation_results);
	list_free(nn->test_data);
	list_free(nn->test_results);

	free(nn);
}

double sigmoid(double z) { return 1 / (1 + exp(-z)); }

double sigmoid_prime(double z) { return sigmoid(z) * (1 - sigmoid(z)); }

void feedforward(neural_network_t *nn, matrix_t *a, matrix_t **output) {
	if (!nn || !nn->biases || !nn->weights || list_len(nn->biases) != list_len(nn->weights)) return;
	matrix_t *mp, *ms;
	for (int i = 0; i < list_len(nn->biases); i++) {
		matrix_product(list_get(nn->weights, i), a, &mp);
		matrix_sum(mp, list_get(nn->biases, i), &ms);
		a = matrix_elementwise_func_4_ret(ms, sigmoid);
		matrix_free(mp);
		matrix_free(ms);
	}
	*output = a;
}

void sgd(neural_network_t *nn, size_t epochs, size_t mini_batch_size, double eta) {
	if (!nn || !nn->training_data) return;

	if (mini_batch_size > list_len(nn->training_data)) {
		printf("Invalid mini batch size (%lu). Please choose a size less "
			"than the size of the training data (%lu).\n",
			mini_batch_size, list_len(nn->training_data));
		return;
	}

	for (size_t i = 0; i < epochs; i++) {
		list_shuffle(nn->training_data);
		for (size_t j = 0; j < list_len(nn->training_data); j+= mini_batch_size)
			update_mini_batch(nn, j, j + mini_batch_size, eta);
		if (nn->test_data)
			printf("%sEpoch (%lu): (%lu) / (%lu)%s\n", KRED, i + 1, evaluate(nn), list_len(nn->test_data), KNRM);
		else
			printf("%sEpoch (%lu) complete%s\n", KRED, i + 1, KNRM);
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
		list_free(delta_nabla_b);
		list_free(delta_nabla_w);
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

	list_free(nabla_b);
	list_free(nabla_w);
}

void cost_derivative(matrix_t *output_activations, matrix_t *y, matrix_t **deriv) {
	if (!output_activations || !y) return;
	matrix_t *m, *d;
	m = matrix_product_scalar_ret(y, -1);
	matrix_sum(output_activations, m, &d);
	matrix_free(m);
	if (deriv) *deriv = d;
}

void backprop(neural_network_t *nn, matrix_t *x, matrix_t *y, list_t **nabla_b, list_t **nabla_w) {
	if (!nn || !nn->biases || !nn->weights || list_len(nn->biases) != list_len(nn->weights)) return;
	list_t *nabla_bb, *nabla_ww;
	list_init(&nabla_bb, list_len(nn->biases), NULL, (free_fp)matrix_free);
	list_init(&nabla_ww, list_len(nn->weights), NULL, (free_fp)matrix_free);
	if (nabla_b) *nabla_b = nabla_bb;
	if (nabla_w) *nabla_w = nabla_ww;

	for (size_t i = 0; i < list_len(nn->biases); i++) {
		matrix_t *mb, *mw;
		matrix_zero_init(&mb, matrix_num_rows(list_get(nn->biases, i)), matrix_num_cols(list_get(nn->biases, i)));
		matrix_zero_init(&mw, matrix_num_rows(list_get(nn->weights, i)), matrix_num_cols(list_get(nn->weights, i)));
		list_set(nabla_bb, i, mb);
		list_set(nabla_ww, i, mw);
	}
	printf("x(%s)(%lu)(%lu)\n", x?"GGG":"BBB", matrix_num_rows(x), matrix_num_cols(x));
	matrix_print(x, 2, 0);

	matrix_t *activation = x;
	list_t *activations, *zs;
	list_init(&activations, 1 + list_len(nn->biases), NULL, (free_fp)matrix_free);
	list_init(&zs, list_len(nn->biases), NULL, (free_fp)matrix_free);
	list_set(activations, 0, x);

	for (size_t i = 0; i < list_len(nn->biases); i++) {
		matrix_t *b = list_get(nn->biases, i);
		matrix_t *w = list_get(nn->weights, i);
		printf("w(%s)(%lu)(%lu)\n", w?"GGG":"BBB", matrix_num_rows(w), matrix_num_cols(w));
		matrix_print(w, 2, 0);
		printf("b(%s)(%lu)(%lu)\n", b?"GGG":"BBB", matrix_num_rows(b), matrix_num_cols(b));
		matrix_print(b, 2, 0);
		matrix_t *p, *z;
		matrix_product(w, activation, &p);
		matrix_sum(p, b, &z);
		printf("z(%s)(%lu)(%lu)\n", z?"GGG":"BBB", matrix_num_rows(z), matrix_num_cols(z));
		matrix_print(z, 2, 0);
		list_set(zs, i, z);
		activation = matrix_elementwise_func_4_ret(z, sigmoid);
		printf("sigmoid-z(%s)(%lu)(%lu)\n", activation?"GGG":"BBB", matrix_num_rows(activation), matrix_num_cols(activation));
		matrix_print(activation, 2, 0);
		list_set(activations, 1 + i, activation);
		matrix_free(p);
	}

	printf("activations-1(%s)(%lu)(%lu)\n", list_get(activations, -1)?"GGG":"BBB", matrix_num_rows(list_get(activations, -1)), matrix_num_cols(list_get(activations, -1)));
	matrix_print(list_get(activations, -1), 2, 0);
	printf("y(%s)(%lu)(%lu)\n", y?"GGG":"BBB", matrix_num_rows(y), matrix_num_cols(y));
	matrix_print(y, 2, 0);
	matrix_t *cost_deriv = NULL, *delta = NULL, *delta_tmp = NULL, *wwlast = NULL;
	cost_derivative(list_get(activations, -1), y, &cost_deriv);
	printf("cost_deriv(%s)(%lu)(%lu)\n", cost_deriv?"GGG":"BBB", matrix_num_rows(cost_deriv), matrix_num_cols(cost_deriv));
	matrix_print(cost_deriv, 2, 0);
	matrix_t *zw = list_get(zs, -1);
	printf("zw(%s)(%lu)(%lu)\n", zw?"GGG":"BBB", matrix_num_rows(zw), matrix_num_cols(zw));
	matrix_print(zw, 2, 0);
	matrix_t *zsp = matrix_elementwise_func_4_ret(list_get(zs, -1), sigmoid_prime);
	printf("zsp(%s)(%lu)(%lu)\n", zsp?"GGG":"BBB", matrix_num_rows(zsp), matrix_num_cols(zsp));
	matrix_print(zsp, 2, 0);
	matrix_product_elementwise(cost_deriv, zsp, &delta);
	matrix_free(zsp);
	matrix_free(cost_deriv);
	list_set(nabla_bb, list_len(nabla_bb) - 1, delta);
	printf("delta(%s)(%lu)(%lu)\n", delta?"GGG":"BBB", matrix_num_rows(delta), matrix_num_cols(delta));
	matrix_print(delta, 2, 0);
	matrix_t *act_tr = matrix_transpose(list_get(activations, -2));
	matrix_product(delta, act_tr, &wwlast);
	matrix_free(act_tr);
	list_set(nabla_ww, list_len(nabla_ww) - 1, wwlast);

	for (size_t i = 2; i < NUMBER_OF_LAYERS; i++) {
		matrix_t *z = list_get(zs, -i);
		matrix_t *sp = matrix_elementwise_func_4_ret(z, sigmoid_prime);
		matrix_t *t = matrix_transpose(list_get(nn->weights, -i + 1));
		matrix_product(t, delta, &delta_tmp);
		matrix_free(t);
		delta = NULL;
		matrix_product_elementwise(delta_tmp, sp, &delta);
		matrix_free(sp);
		matrix_free(delta_tmp);
		list_set(nabla_bb, -i, delta);
		t = matrix_transpose(list_get(activations, -i - 1));
		matrix_product(delta, t, &delta_tmp);
		list_set(nabla_ww, -i, delta_tmp);
		matrix_free(delta_tmp);
		matrix_free(t);
	}

//	list_free(activations);
	list_free(zs);
}

size_t evaluate(neural_network_t *nn) {
	if (!nn || !nn->test_data || !nn->test_results || list_len(nn->test_data) != list_len(nn->test_results)) return 0;

	size_t correct = 0, ffr = 0, ffc = 0, yr = 0, yc = 0;
	double ffm = -1, ym = -1;
	for (size_t i = 0; i < list_len(nn->test_data); i++) {
		matrix_t *x = list_get(nn->test_data, i);
		matrix_t *y = list_get(nn->test_results, i);
		matrix_t *ffout;
		feedforward(nn, x, &ffout);
		ffm = matrix_argmax(ffout, &ffr, &ffc);
		ym = matrix_argmax(y, &yr, &yc);
		printf("XXXXX (%lu)(%lu):\n", matrix_num_rows(x), matrix_num_cols(x));
//		matrix_print(x, 3, 0);
		printf("FFOUT (%lu)(%lu)ffr(%lu)ffc(%lu)ffm(%.3f):\n", matrix_num_rows(ffout), matrix_num_cols(ffout), 1+ffr, 1+ffc, ffm);
		matrix_print(ffout, 9, 0);
		printf("YYYYY (%lu)(%lu)yr(%lu)yc(%lu)ym(%.3f):\n", matrix_num_rows(y), matrix_num_cols(y), 1+yr, 1+yc, ym);
		matrix_print(y, 0, 0);
		if (ffr == yr && ffc == yc) correct++;
		matrix_free(ffout);
	}

	return correct;
}

int run_mnist() {
	size_t epochs = 1;
	size_t mini_batch_size = 10;
	double eta = 3.0;
	size_t num_neurs_1 = 784;
	size_t num_neurs_2 = 30;
	size_t num_neurs_3 = 10;
	printf("run_mnist epochs(%lu)mbs(%lu)eta(%.1f)(%lu)(%lu)(%lu)\n", epochs, mini_batch_size, eta, num_neurs_1, num_neurs_2, num_neurs_3);

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
	int sizes[] = {num_neurs_1, num_neurs_2, num_neurs_3};
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
//
//	double arr[4] = { -4, 8, 13, 31 };
//	double *ra;
//	sigmoid_arr(arr, 4, &ra);
//	for (int j = 0; j < 4; j++) printf("DDD (%d)(%f)\n", j, ra[j]);

	sgd(w, epochs, mini_batch_size, eta);
	neural_network_free(w);

	return 0;
}

int run_dummy() {
	size_t epochs = 3;
	size_t mini_batch_size = 1;
	double eta = 3.0;
	size_t num_neurs_1 = 14;
	size_t num_neurs_2 = 3;
	size_t num_neurs_3 = 4;
	printf("run_dummy epochs(%lu)mbs(%lu)eta(%.1f)(%lu)(%lu)(%lu)\n", epochs, mini_batch_size, eta, num_neurs_1, num_neurs_2, num_neurs_3);
	size_t num_trd = 5;
	size_t num_vdd = 3;
	size_t num_ttd = 2;

	neural_network_t *nn;
	int sizes[] = {num_neurs_1, num_neurs_2, num_neurs_3};
	neural_net_init(&nn, sizes);
	list_init(&nn->training_data, num_trd, NULL, (free_fp)matrix_free);
	list_init(&nn->training_results, num_trd, NULL, (free_fp)matrix_free);
	list_init(&nn->validation_data, num_vdd, NULL, (free_fp)matrix_free);
	list_init(&nn->validation_results, num_vdd, NULL, (free_fp)matrix_free);
	list_init(&nn->test_data, num_ttd, NULL, (free_fp)matrix_free);
	list_init(&nn->test_results, num_ttd, NULL, (free_fp)matrix_free);

	for (size_t i = 0; i < num_trd; i++) {
		matrix_t *m, *mr;
		matrix_init(&m, num_neurs_1, 1, NULL);
		matrix_init(&mr, num_neurs_3, 1, NULL);
		for (int j = 0; j < num_neurs_1; j++)
			matrix_set(m, j, 0, gaussrand_0to1());
		int resval = mrand(num_neurs_3);
		for (int j = 0; j < num_neurs_3; j++)
			matrix_set(mr, j, 0, resval == j ? 1 : 0);
		list_set(nn->training_data, i, m);
		list_set(nn->training_results, i, mr);
	}

	for (size_t i = 0; i < num_vdd; i++) {
		matrix_t *m, *mr;
		matrix_init(&m, num_neurs_1, 1, NULL);
		matrix_init(&mr, num_neurs_3, 1, NULL);
		for (int j = 0; j < num_neurs_1; j++)
			matrix_set(m, j, 0, gaussrand_0to1());
		int resval = mrand(num_neurs_3);
		for (int j = 0; j < num_neurs_3; j++)
			matrix_set(mr, j, 0, resval == j ? 1 : 0);
		list_set(nn->validation_data, i, m);
		list_set(nn->validation_results, i, mr);
	}

	for (size_t i = 0; i < num_ttd; i++) {
		matrix_t *m, *mr;
		matrix_init(&m, num_neurs_1, 1, NULL);
		matrix_init(&mr, num_neurs_3, 1, NULL);
		for (int j = 0; j < num_neurs_1; j++)
			matrix_set(m, j, 0, gaussrand_0to1());
		int resval = mrand(num_neurs_3);
		for (int j = 0; j < num_neurs_3; j++)
			matrix_set(mr, j, 0, resval == j ? 1 : 0);
		list_set(nn->test_data, i, m);
		list_set(nn->test_results, i, mr);
	}

	sgd(nn, epochs, mini_batch_size, eta);
	neural_network_free(nn);

	return 0;
}

int main() {
	printf("network-0 (%lu)(%lu)(%lu)\n", sizeof(unsigned int), sizeof(uint32_t), sizeof(double));
//	run_mnist();
	run_dummy();
}
