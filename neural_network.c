#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "neural_network.h"
#include "maths.h"
#include "prepare_data.h"

struct neural_network {
	size_t num_layers;
	list_t *num_neurons;
	list_t *weights;
	list_t *biases;
	list_t *training_data;
	list_t *validation_data;
	list_t *test_data;
};

void neural_net_init(neural_network_t **neural_net, list_t *num_neurons) {
	if (!neural_net || !num_neurons) return;

	neural_network_t *nn = malloc(sizeof(neural_network_t));
	if (!nn) return;
	memset(nn, '\0', sizeof(neural_network_t));

	*neural_net = nn;
	nn->num_layers = list_len(num_neurons);
	nn->num_neurons = num_neurons;

	list_init(&nn->weights, nn->num_layers - 1, NULL, (free_fp)matrix_free);
	list_init(&nn->biases, nn->num_layers - 1, NULL, (free_fp)matrix_free);

	for (size_t i = 0; i < nn->num_layers - 1; i++) {
		matrix_t *w = NULL, *b = NULL;
		matrix_init(&w, *(int*)list_get(num_neurons, i + 1), *(int*)list_get(num_neurons, i), NULL);
		matrix_init(&b, *(int*)list_get(num_neurons, i + 1), 1, NULL);
		list_set(nn->weights, i, w);
		list_set(nn->biases, i, b);
		matrix_elementwise_func_3(w, gaussrand);
		matrix_elementwise_func_3(b, gaussrand);
	}

//	mllog(LOG_LEVEL_HIGH, 1, "NN_init wts1 (%s)(%lu)(%lu)\n", wts1?"GGG":"BBB", matrix_num_rows(wts1), matrix_num_cols(wts1));
////	nn_matrix_print(LOG_LEVEL_HIGH, wts1, 13, 0);
//	mllog(LOG_LEVEL_HIGH, 1, "NN_init wts2 (%s)(%lu)(%lu)\n", wts2?"GGG":"BBB", matrix_num_rows(wts2), matrix_num_cols(wts2));
////	nn_matrix_print(LOG_LEVEL_HIGH, wts2, 13, 0);
//	mllog(LOG_LEVEL_HIGH, 1, "NN_init bias1 (%s)(%lu)(%lu)\n", bias1?"GGG":"BBB", matrix_num_rows(bias1), matrix_num_cols(bias1));
////	nn_matrix_print(LOG_LEVEL_HIGH, bias1, 13, 0);
//	mllog(LOG_LEVEL_HIGH, 1, "NN_init bias2 (%s)(%lu)(%lu)\n", bias2?"GGG":"BBB", matrix_num_rows(bias2), matrix_num_cols(bias2));
////	nn_matrix_print(LOG_LEVEL_HIGH, bias2, 13, 0);
//
//	matrix_print_to_file(wts1, 15, 0, "wts1.txt");
//	matrix_print_to_file(wts2, 15, 0, "wts2.txt");
//	matrix_print_to_file(bias1, 15, 0, "bia1.txt");
//	matrix_print_to_file(bias2, 15, 0, "bia2.txt");

}

void neural_network_free(neural_network_t *nn) {
	if (!nn) return;

	list_free(nn->weights);
	list_free(nn->biases);
	list_free(nn->training_data);
	list_free(nn->validation_data);
	list_free(nn->test_data);

	free(nn);
}

double sigmoid(double z) { return 1 / (1 + exp(-z)); }

double sigmoid_prime(double z) { return sigmoid(z) * (1 - sigmoid(z)); }

void feedforward(neural_network_t *nn, matrix_t *a, matrix_t **output) {
	if (!nn || !nn->biases || !nn->weights || list_len(nn->biases) != list_len(nn->weights)) return;
	matrix_t *mp, *ms, *acopy;
	matrix_copy(a, &acopy);
	for (int i = 0; i < list_len(nn->biases); i++) {
		matrix_product(list_get(nn->weights, i), acopy, &mp);
		matrix_sum(mp, list_get(nn->biases, i), &ms);
		acopy = matrix_elementwise_func_4_ret(ms, sigmoid);
		matrix_free(mp);
		matrix_free(ms);
	}
	*output = acopy;
}

void sgd(neural_network_t *nn, size_t epochs, size_t mini_batch_size, double eta) {
	if (!nn || !nn->training_data) {
		printf("sgd ERROR: nn(%s) training_data(%s)\n", nn?"GGG":"BBB", nn->training_data?"GGG":"BBB");
		exit(1);
	}

	if (mini_batch_size > list_len(nn->training_data)) {
		mllog(LOG_LEVEL_HIGH, 1, "Invalid mini batch size (%lu). Please choose a size less "
			"than the size of the training data (%lu).\n",
			mini_batch_size, list_len(nn->training_data));
		exit(1);
	}

	for (size_t i = 0; i < epochs; i++) {
		list_shuffle(nn->training_data);
		for (size_t j = 0; j < list_len(nn->training_data); j+= mini_batch_size)
			update_mini_batch(nn, j, j + mini_batch_size, eta);
		if (nn->test_data)
			mllog(LOG_LEVEL_HIGH, 1, "%sEpoch (%lu): (%lu) / (%lu)%s\n", KRED, i + 1, evaluate(nn), list_len(nn->test_data), KNRM);
		else
			mllog(LOG_LEVEL_HIGH, 1, "%sEpoch (%lu) complete%s\n", KRED, i + 1, KNRM);
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
		matrix_t *x = tuple_get(list_get(nn->training_data, i), 0);
		matrix_t *y = tuple_get(list_get(nn->training_data, i), 1);
//		matrix_t *y = list_get(nn->training_results, i);
		list_t *delta_nabla_b, *delta_nabla_w;
		backprop(nn, x, y, &delta_nabla_b, &delta_nabla_w);

		for (size_t j = 0; j < list_len(nabla_b); j++) {
			matrix_t *nbsum, *nwsum;

			matrix_t *nb = list_get(nabla_b, j);
			matrix_t *dnb = list_get(delta_nabla_b, j);
			matrix_sum(nb, dnb, &nbsum);
			list_set(nabla_b, j, nbsum);

			matrix_t *nw = list_get(nabla_w, j);
			matrix_t *dnw = list_get(delta_nabla_w, j);
			matrix_sum(nw, dnw, &nwsum);
			list_set(nabla_w, j, nwsum);
		}
		list_free(delta_nabla_b);
		list_free(delta_nabla_w);
	}

	double scalar = eta / (double)((long)start - (long)end);

	for (size_t i = 0; i < list_len(nn->biases); i++) {
		matrix_t *sw, *sb;
		matrix_t *w = list_get(nn->weights, i);
		matrix_t *nw = list_get(nabla_w, i);
		matrix_t *b = list_get(nn->biases, i);
		matrix_t *nb = list_get(nabla_b, i);

		matrix_product_scalar(nw, scalar);
		matrix_product_scalar(nb, scalar);
		matrix_sum(w, nw, &sw);
		matrix_sum(b, nb, &sb);

		list_set(nn->weights, i, sw);
		list_set(nn->biases, i, sb);
	}

//	matrix_print(list_get(nn->weights, 0), 3, 3);
//	matrix_print(list_get(nn->biases, 0), 3, 3);
//	list_free(nabla_b);
//	list_free(nabla_w);
}

void cost_derivative(matrix_t *output_activations, matrix_t *y, matrix_t **deriv) {
	if (!output_activations || !y) {
		printf("cost_derivative: ERROR: output_activations (%s) y (%s)\n", output_activations?"GGG":"BBB", y?"GGG":"BBB");
		exit(1);
	}

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

	matrix_t *activation = x;
	list_t *activations, *zs;
	list_init(&activations, 1 + list_len(nn->biases), NULL, (free_fp)matrix_free);
	list_init(&zs, list_len(nn->biases), NULL, (free_fp)matrix_free);
	list_set(activations, 0, x);

	for (size_t i = 0; i < list_len(nn->biases); i++) {
		matrix_t *b = list_get(nn->biases, i);
		matrix_t *w = list_get(nn->weights, i);
		matrix_t *p, *z;
		matrix_product(w, activation, &p);
		matrix_sum(p, b, &z);
		list_set(zs, i, z);
		activation = matrix_elementwise_func_4_ret(z, sigmoid);
		list_set(activations, 1 + i, activation);
		matrix_free(p);
	}

	matrix_t *cost_deriv = NULL, *delta = NULL, *delta_tmp = NULL, *wwlast = NULL;
	cost_derivative(list_get(activations, -1), y, &cost_deriv);
	matrix_t *zsp = matrix_elementwise_func_4_ret(list_get(zs, -1), sigmoid_prime);
	matrix_product_elementwise(cost_deriv, zsp, &delta);
	matrix_free(zsp);
	matrix_free(cost_deriv);
	list_set(nabla_bb, -1, delta);
	matrix_t *act_tr = matrix_transpose(list_get(activations, -2));
	matrix_product(delta, act_tr, &wwlast);
	matrix_free(act_tr);
	list_set(nabla_ww, -1, wwlast);

	for (long i = 2; i < nn->num_layers; i++) {
		matrix_t *z = list_get(zs, -i);
		matrix_t *sp = matrix_elementwise_func_4_ret(z, sigmoid_prime);
		matrix_t *t = matrix_transpose(list_get(nn->weights, -i + 1));
		matrix_product(t, delta, &delta_tmp);
		matrix_free(t);
		delta = NULL;
		matrix_product_elementwise(delta_tmp, sp, &delta);
		matrix_free(sp);
		matrix_free(delta_tmp);
		list_set(nabla_bb, -1 * (long)i, delta);
		t = matrix_transpose(list_get(activations, -i - 1));
		matrix_product(delta, t, &delta_tmp);
		list_set(nabla_ww, -1 * (long)i, delta_tmp);
//		matrix_free(delta_tmp);
		delta_tmp = NULL;
		matrix_free(t);
	}

//	list_free(activations);
	list_free(zs);
}

size_t evaluate(neural_network_t *nn) {
	if (!nn || !nn->test_data) return 0;
	mllog(LOG_LEVEL_DEBUG, 1, "Lets EVALUATE\n");

	size_t correct = 0, ffr = 0, ffc = 0, yr = 0, yc = 0;
	for (size_t i = 0; i < list_len(nn->test_data); i++) {
		matrix_t *x = tuple_get(list_get(nn->test_data, i), 0);
		matrix_t *y = tuple_get(list_get(nn->test_data, i), 1);
//		matrix_t *y = list_get(nn->test_results, i);
		matrix_t *ffout;
		feedforward(nn, x, &ffout);
		matrix_argmax(ffout, &ffr, &ffc);
		matrix_argmax(y, &yr, &yc);
		if (ffr == yr && ffc == yc) correct++;
		matrix_free(ffout);
	}

	return correct;
}

void nn_matrix_print(LOG_LEVEL LL, matrix_t *m, int precision, int zero_precision) {
	switch (get_environment()) {
		case ENV_DEBUG: {
			matrix_print(m, precision, zero_precision);
			break;
		}
		case ENV_DEV: {
			if (LL >= LOG_LEVEL_WARNING) matrix_print(m, precision, zero_precision);
			break;
		}
		case ENV_PROD: {
			if (LL >= LOG_LEVEL_HIGH) matrix_print(m, precision, zero_precision);
			break;
		}
		default: break;
	}
}

int run_mnist() {
	size_t epochs = 3;
	size_t mini_batch_size = 10;
	double eta = 3.0;
	size_t num_neurs_1 = 784;
	size_t num_neurs_2 = 30;
	size_t num_neurs_3 = 10;
	mllog(LOG_LEVEL_HIGH, 1, "run_mnist epochs(%lu)mbs(%lu)eta(%.1f)(%lu)(%lu)(%lu)\n", epochs, mini_batch_size, eta, num_neurs_1, num_neurs_2, num_neurs_3);

	uint32_t magic_number_images, magic_number_labels, num_train_images, num_train_labels, num_test_images, num_test_labels, num_rows, num_cols;
	int rih = images_header(TRAIN_IMAGES_FILENAME, &magic_number_images, &num_train_images, &num_rows, &num_cols);
	if (rih == -1) {
		mllog(LOG_LEVEL_HIGH, 1, "There was a problem collecting training images.\n");
		return -1;
	}

	rih = images_header(TEST_IMAGES_FILENAME, &magic_number_images, &num_test_images, &num_rows, &num_cols);
	if (rih == -1) {
		mllog(LOG_LEVEL_HIGH, 1, "There was a problem collecting test images.\n");
		return -1;
	}

	rih = labels_header(TRAIN_LABELS_FILENAME, &magic_number_labels, &num_train_labels);
	if (rih == -1) {
		mllog(LOG_LEVEL_HIGH, 1, "There was a problem collecting training labels.\n");
		return -1;
	}

	rih = labels_header(TEST_LABELS_FILENAME, &magic_number_labels, &num_test_labels);
	if (rih == -1) {
		mllog(LOG_LEVEL_HIGH, 1, "There was a problem collecting testing labels.\n");
		return -1;
	}

	neural_network_t *w;
	list_t *sizes = NULL;
	list_init(&sizes, 3, NULL, NULL);
	list_set(sizes, 0, &num_neurs_1);
	list_set(sizes, 1, &num_neurs_2);
	list_set(sizes, 2, &num_neurs_3);
//	int sizes[] = {num_neurs_1, num_neurs_2, num_neurs_3};
	neural_net_init(&w, sizes);
	for (int i = 0; i < 3; i++) mllog(LOG_LEVEL_HIGH, 1, "sz(%d)(%d)\n", i, *(int*)list_get(w->num_neurons, i));

	list_t *tr_d = NULL, *tr_r = NULL, *vl_d = NULL, *vl_r = NULL, *te_d = NULL, *te_r = NULL;

	size_t num_validation = 10 * 1000;
	get_images(TRAIN_IMAGES_FILENAME, num_train_images, num_rows, num_cols, num_validation, &tr_d, &vl_d);
	get_images(TEST_IMAGES_FILENAME, num_test_images, num_rows, num_cols, 0, &te_d, NULL);
	mllog(LOG_LEVEL_HIGH, 1, "We got the images (%lu)(%lu)(%lu)...\n", list_len(tr_d), list_len(vl_d), list_len(te_d));
	get_labels(TRAIN_LABELS_FILENAME, num_train_labels, num_validation, &tr_r, &vl_r);
	get_labels(TEST_LABELS_FILENAME, num_test_labels, 0, &te_r, NULL);
	mllog(LOG_LEVEL_HIGH, 1, "We got the labels (%lu)(%lu)(%lu)...\n", list_len(tr_r), list_len(vl_r), list_len(te_r));

//	size_t num_validation = 0;
//	size_t num_images = 10000;
//	get_images(TRAIN_IMAGES_FILENAME, num_images, num_rows, num_cols, num_validation, &tr_d, &vl_d);
//	get_images(TEST_IMAGES_FILENAME, num_images, num_rows, num_cols, 0, &te_d, NULL);
//	mllog(LOG_LEVEL_HIGH, 1, "We got the images (%lu)(%lu)(%lu)...\n", list_len(tr_d), list_len(vl_d), list_len(te_d));
//	get_labels(TRAIN_LABELS_FILENAME, num_images, num_validation, &tr_r, &vl_r);
//	get_labels(TEST_LABELS_FILENAME, num_images, 0, &te_r, NULL);
//	mllog(LOG_LEVEL_HIGH, 1, "We got the labels (%lu)(%lu)(%lu)...\n", list_len(tr_r), list_len(vl_r), list_len(te_r));

	w->training_data = zip(tr_d, tr_r);
//	w->validation_data = zip(vl_d, vl_r);
	w->test_data = zip(te_d, te_r);

//	size_t wes = 4;
//	matrix_t *m = list_get(w->training_data, wes);
//	mllog(LOG_LEVEL_HIGH, 1, "Lets print training matrix (%lu)(%s)\n", wes, m ? "GOOD" : "BAD");
//	nn_matrix_print(LOG_LEVEL_HIGH, m, 14, 1);
//	m = list_get(w->training_results, wes);
//	mllog(LOG_LEVEL_HIGH, 1, "Lets print training label (%lu)(%s)\n", wes, m ? "GOOD" : "BAD");
//	nn_matrix_print(LOG_LEVEL_HIGH, m, 8, 1);

//	wes = 7777;
//	m = list_get(w->validation_data, wes);
//	mllog(LOG_LEVEL_DEBUG, 1, "Lets print validation matrix (%lu)(%s)\n", wes, m ? "GOOD" : "BAD");
////	nn_matrix_print(LOG_LEVEL_DEBUG, m, 8, 1);
//	m = list_get(w->validation_results, wes);
//	mllog(LOG_LEVEL_DEBUG, 1, "Lets print validation label (%lu)(%s)\n", wes, m ? "GOOD" : "BAD");
//	nn_matrix_print(LOG_LEVEL_DEBUG, m, 8, 1);
//
//	wes = 6666;
//	m = list_get(w->test_data, wes);
//	mllog(LOG_LEVEL_DEBUG, 1, "Lets print test matrix (%lu)(%s)\n", wes, m ? "GOOD" : "BAD");
//	nn_matrix_print(LOG_LEVEL_DEBUG, m, 8, 1);
//	m = list_get(w->test_results, wes);
//	mllog(LOG_LEVEL_DEBUG, 1, "Lets print test label (%lu)(%s)\n", wes, m ? "GOOD" : "BAD");
//	nn_matrix_print(LOG_LEVEL_DEBUG, m, 8, 1);

//	mllog(LOG_LEVEL_DEBUG, 1, "XXXXX (%f)(%f)(%f)(%f)\n", exp(-6), sigmoid(6), exp(6), sigmoid(-6));
//
//	double arr[4] = { -4, 8, 13, 31 };
//	double *ra;
//	sigmoid_arr(arr, 4, &ra);
//	for (int j = 0; j < 4; j++) mllog(LOG_LEVEL_DEBUG, 1, "DDD (%d)(%f)\n", j, ra[j]);

	sgd(w, epochs, mini_batch_size, eta);
	neural_network_free(w);

//	matrix_t *m = list_get(w->training_data, 0);
//	double wv = -100.0;
//	long wi = -1;
//	for (long i = 0; i < matrix_num_rows(m); i++) {
//		double v = matrix_get(m, i, 0);
//		if (v != 0) {
//			printf("WDWDWDWDWDWDWDWDWDWDWDWDWDWDWDWDWDWDW (%.15f)(%ld)\n", v, i);
//			break;
//		} else printf("(%ld)", i);
//	}

	return 0;
}

int run_toy() {
	size_t epochs = 300;
	size_t mini_batch_size = 1;
	double eta = 0.15;
	size_t num_neurs_1 = 1;
	size_t num_neurs_2 = 1;
	mllog(LOG_LEVEL_HIGH, 1, "run_toy epochs(%lu)mbs(%lu)eta(%.1f)(%lu)(%lu)\n", epochs, mini_batch_size, eta, num_neurs_1, num_neurs_2);

	neural_network_t *nn;
	list_t *sizes = NULL;
	list_init(&sizes, 2, NULL, NULL);
	list_set(sizes, 0, &num_neurs_1);
	list_set(sizes, 1, &num_neurs_2);
	neural_net_init(&nn, sizes);

	list_t *tr_d, *tr_r, *te_d, *te_r;
	list_init(&tr_d, 1, NULL, (free_fp)matrix_free);
	list_init(&tr_r, 1, NULL, (free_fp)matrix_free);
	list_init(&te_d, 1, NULL, (free_fp)matrix_free);
	list_init(&te_r, 1, NULL, (free_fp)matrix_free);

	matrix_t *trdm = NULL, *trrm = NULL, *tedm = NULL, *term = NULL;
	matrix_init(&trdm, 1, 1, NULL);
	matrix_init(&trrm, 1, 1, NULL);
	matrix_init(&tedm, 1, 1, NULL);
	matrix_init(&term, 1, 1, NULL);

	matrix_set(trdm, 0, 0, 1);
	matrix_set(trrm, 0, 0, 0);
	matrix_set(tedm, 0, 0, 1);
	matrix_set(term, 0, 0, 0);

	list_set(tr_d, 0, trdm);
	list_set(tr_r, 0, trrm);
	list_set(te_d, 0, tedm);
	list_set(te_r, 0, term);

	nn->training_data = zip(tr_d, tr_r);
	nn->test_data = zip(te_d, te_r);

	sgd(nn, epochs, mini_batch_size, eta);
	neural_network_free(nn);

	return 0;
}

int run_dummy() {
	size_t epochs = 3;
	size_t mini_batch_size = 10;
	double eta = 3.0;
	size_t num_neurs_1 = 256;
	size_t num_neurs_2 = 30;
	size_t num_neurs_3 = 4;
	mllog(LOG_LEVEL_HIGH, 1, "run_dummy epochs(%lu)mbs(%lu)eta(%.1f)(%lu)(%lu)(%lu)\n", epochs, mini_batch_size, eta, num_neurs_1, num_neurs_2, num_neurs_3);
	size_t num_trd = 500;
	size_t num_vdd = 100;
	size_t num_ttd = 100;

	neural_network_t *nn;
	list_t *sizes = NULL;
	list_init(&sizes, 3, NULL, NULL);
	list_set(sizes, 0, &num_neurs_1);
	list_set(sizes, 1, &num_neurs_2);
	list_set(sizes, 2, &num_neurs_3);
//	int sizes[] = {num_neurs_1, num_neurs_2, num_neurs_3};
	neural_net_init(&nn, sizes);
	list_t *tr_d, *tr_r, *vl_d, *vl_r, *te_d, *te_r;
	list_init(&tr_d, num_trd, NULL, (free_fp)matrix_free);
	list_init(&tr_r, num_trd, NULL, (free_fp)matrix_free);
	list_init(&vl_d, num_vdd, NULL, (free_fp)matrix_free);
	list_init(&vl_r, num_vdd, NULL, (free_fp)matrix_free);
	list_init(&te_d, num_ttd, NULL, (free_fp)matrix_free);
	list_init(&te_r, num_ttd, NULL, (free_fp)matrix_free);

	for (size_t i = 0; i < num_trd; i++) {
		matrix_t *m, *mr;
		matrix_init(&m, num_neurs_1, 1, NULL);
		matrix_init(&mr, num_neurs_3, 1, NULL);
		for (int j = 0; j < num_neurs_1; j++)
			matrix_set(m, j, 0, gaussrand_0to1());
		int resval = mrand(num_neurs_3);
		for (int j = 0; j < num_neurs_3; j++)
			matrix_set(mr, j, 0, resval == j ? 1 : 0);
		list_set(tr_d, i, m);
		list_set(tr_r, i, mr);
	}
	nn->training_data = zip(tr_d, tr_r);

	for (size_t i = 0; i < num_vdd; i++) {
		matrix_t *m, *mr;
		matrix_init(&m, num_neurs_1, 1, NULL);
		matrix_init(&mr, num_neurs_3, 1, NULL);
		for (int j = 0; j < num_neurs_1; j++)
			matrix_set(m, j, 0, gaussrand_0to1());
		int resval = mrand(num_neurs_3);
		for (int j = 0; j < num_neurs_3; j++)
			matrix_set(mr, j, 0, resval == j ? 1 : 0);
		list_set(vl_d, i, m);
		list_set(vl_r, i, mr);
	}
	nn->validation_data = zip(vl_d, vl_r);

	for (size_t i = 0; i < num_ttd; i++) {
		matrix_t *m, *mr;
		matrix_init(&m, num_neurs_1, 1, NULL);
		matrix_init(&mr, num_neurs_3, 1, NULL);
		for (int j = 0; j < num_neurs_1; j++)
			matrix_set(m, j, 0, gaussrand_0to1());
		int resval = mrand(num_neurs_3);
		for (int j = 0; j < num_neurs_3; j++)
			matrix_set(mr, j, 0, resval == j ? 1 : 0);
		list_set(te_d, i, m);
		list_set(te_r, i, mr);
	}
	nn->test_data = zip(te_d, te_r);

	sgd(nn, epochs, mini_batch_size, eta);
	neural_network_free(nn);

	return 0;
}

int main(int argc, char **argv) {
	printf("Initializing...\n");

	if (argc < 2) {
		printf("Please provide an environment ('dev' or 'prod') as the first argument\n");
		exit(1);
	}

	char *arg_env = argv[1];
	set_environment_from_str(arg_env);
	char *env_str = get_environment_as_str();
	mllog(LOG_LEVEL_HIGH, 1, "network-0 (%s)(%lu)(%lu)(%lu)\n", env_str, sizeof(unsigned int), sizeof(uint32_t), sizeof(double));
	run_mnist();
	double z = -4.79227437;
	double act = sigmoid(z);
	double zsp = sigmoid_prime(z);
	mllog(LOG_LEVEL_DEBUG, 1, "z(%.12f) activatttttion(%.12f) zsp(%.12f)\n", z, act, zsp);
//	run_dummy();
//	run_toy();
}
