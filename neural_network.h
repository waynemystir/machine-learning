#ifndef neural_network_h
#define neural_network_h

#include "common.h"
#include "matrix.h"

typedef struct neural_network neural_network_t;

void neural_net_init(neural_network_t **neural_net, list_t *num_neurons);
void feedforward(neural_network_t *nn, matrix_t *a, matrix_t **output);
void sgd(neural_network_t *nn, size_t epochs, size_t mini_batch_size, double eta);
void nn_matrix_print(LOG_LEVEL LL, matrix_t *m, int precision, int zero_precision);
void update_mini_batch(neural_network_t *nn, size_t start, size_t end, double eta);
void backprop(neural_network_t *nn, matrix_t *x, matrix_t *y, list_t **nabla_b, list_t **nabla_w);
size_t evaluate(neural_network_t *nn);

#endif /* neural_network_h */
