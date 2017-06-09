#ifndef neural_network_h
#define neural_network_h

#include "common.h"
#include "matrix.h"

typedef struct neural_network neural_network_t;

void neural_net_init(neural_network_t **neural_net, int num_neurons[3]);
void feedforward(neural_network_t *nn, matrix_t *a, matrix_t **output);
void sgd(linked_list_t *training_data, size_t epochs, size_t mini_batch_size, double eta, linked_list_t *test_data);
void update_mini_batch(linked_list_t *training_data, size_t start, size_t end, double eta);
size_t evaluate(linked_list_t *test_data);

#endif /* neural_network_h */
