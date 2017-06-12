#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#include "matrix.h"

struct matrix {
	size_t num_rows;
	size_t num_cols;
	double **data;
};

void matrix_init(matrix_t **m, size_t num_rows, size_t num_cols, double **data) {
	if (!m) return;
	matrix_t *mx = malloc(SZ_MATRIX);
	if (!mx) return;
	memset(mx, '\0', SZ_MATRIX);
	*m = mx;
	mx->num_rows = num_rows;
	mx->num_cols = num_cols;
	if (data) {
		mx->data = data;
		return;
	}

	mx->data = calloc(num_rows, sizeof(double*));
	for (int i = 0; i < num_rows; i++)
		mx->data[i] = calloc(num_cols, sizeof(double));
}

void matrix_zero_init(matrix_t **m, size_t num_rows, size_t num_cols) {
	if (!m) return;
	matrix_init(m, num_rows, num_cols, NULL);
	for (size_t i = 0; i < num_rows; i++)
		for (size_t j = 0; j < num_cols; j++)
			matrix_set(*m, i, j, 0);
}

size_t matrix_num_rows(matrix_t *m) {
	if (!m) return 0;
	return m->num_rows;
}

size_t matrix_num_cols(matrix_t *m) {
	if (!m) return 0;
	return m->num_cols;
}

double matrix_get(matrix_t *m, size_t row, size_t col) {
	if (!m || !m->data) return DBL_MAX;
	return m->data[row][col];
}

void matrix_set(matrix_t *m, size_t row, size_t col, double value) {
	if (!m || !m->data) return;
	m->data[row][col] = value;
}

void matrix_product(matrix_t *m1, matrix_t *m2, matrix_t **product) {
	if (!m1 || !m2 || m1->num_cols != m2->num_rows)
		return;

	matrix_init(product, m1->num_rows, m2->num_cols, NULL);

	double sum = 0;
	for (int i = 0; i < m1->num_rows; i++)
		for (int j = 0; j < m2->num_cols; j++) {
			for (int k = 0; k < m1->num_cols; k++)
				sum += m1->data[i][k] * m2->data[k][j];
			matrix_set(*product, i, j, sum);
			sum = 0;
		}
}

void matrix_product_scalar(matrix_t *m, double scalar) {
	if (!m) return;

	for (int i = 0; i < m->num_rows; i++)
		for (int j = 0; j < m->num_cols; j++)
			matrix_set(m, i, j, scalar * matrix_get(m, i, j));
}

matrix_t *matrix_product_scalar_ret(matrix_t *m, double scalar) {
	if (!m) return NULL;

	matrix_t *ms;
	matrix_init(&ms, m->num_rows, m->num_cols, NULL);

	for (int i = 0; i < m->num_rows; i++)
		for (int j = 0; j < m->num_cols; j++)
			matrix_set(ms, i, j, scalar * matrix_get(m, i, j));

	return ms;
}

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t **sum) {
	if (!m1 || !m2 || m1->num_rows != m2->num_rows || m1->num_cols != m2->num_cols)
		return;
	matrix_init(sum, m1->num_rows, m1->num_cols, NULL);
	for (int i = 0; i < m1->num_rows; i++)
		for (int j = 0; j < m1->num_cols; j++)
			matrix_set(*sum, i, j, matrix_get(m1, i, j) + matrix_get(m2, i, j));
}

void matrix_elementwise_func_1(matrix_t *m, elementwise_function_1 ef) {
	if (!m || !m->data || !ef)
		return;

	for (int i = 0; i < m->num_rows; i++)
		for (int j = 0; j < m->num_cols; j++)
			ef(&m->data[i][j]);
}

void matrix_elementwise_func_2(matrix_t *m, elementwise_function_2 ef) {
	if (!m || !m->data || !ef)
		return;

	for (int i = 0; i < m->num_rows; i++)
		for (int j = 0; j < m->num_cols; j++)
			ef(m->data[i][j], &m->data[i][j]);
}

void matrix_elementwise_func_3(matrix_t *m, elementwise_function_3 ef) {
	if (!m || !m->data || !ef)
		return;

	for (int i = 0; i < m->num_rows; i++)
		for (int j = 0; j < m->num_cols; j++)
			matrix_set(m, i, j, ef());
}

void matrix_elementwise_func_4(matrix_t *m, elementwise_function_4 ef) {
	if (!m || !m->data || !ef)
		return;

	for (int i = 0; i < m->num_rows; i++)
		for (int j = 0; j < m->num_cols; j++)
			matrix_set(m, i, j, ef(matrix_get(m, i, j)));
}

matrix_t *matrix_elementwise_func_4_ret(matrix_t *m, elementwise_function_4 ef) {
	if (!m || !ef) return NULL;

	matrix_t *mret;
	matrix_init(&mret, m->num_rows, m->num_cols, NULL);
	for (int i = 0; i < m->num_rows; i++)
		for (int j = 0; j < m->num_cols; j++)
			matrix_set(mret, i, j, ef(matrix_get(m, i, j)));
	return mret;
}

matrix_t *matrix_transpose(matrix_t *m) {
	if (!m) return NULL;

	matrix_t *t;
	matrix_init(&t, m->num_cols, m->num_rows, NULL);
	for (int i = 0; i < t->num_rows; i++)
		for (int j = 0; j < t->num_cols; j++)
			matrix_set(t, i, j, matrix_get(m, j, i));

	return t;
}

double matrix_argmax(matrix_t *m, size_t *row, size_t *col) {
	if (!m) return -DBL_MAX;

	double max = matrix_get(m, 0, 0);
	for (size_t i = 0; i < m->num_rows; i++)
		for (size_t j = 0; j < m->num_cols; j++)
			if (max < matrix_get(m, i, j)) {
				max = matrix_get(m, i, j);
				*row = i;
				*col = j;
			}
	return max;
}

void matrix_print(matrix_t *m, int precision, int zero_precision) {
	if (!m) {
		printf("matrix_print-NULL\n");
		return;
	}
	for (int i = 0; i < m->num_rows; i++) {
		if (i != 0) printf("\n");
		for (int j = 0; j < m->num_cols; j++) {
			double w = matrix_get(m, i, j);
			printf("%.*f  ", w == 0 ? zero_precision : precision, w);
		}
	}
	printf("\n");
}

void matrix_free(matrix_t *m) {
	if (!m) return;
	if (m->data) {
		for (size_t i = 0; i < m->num_rows; i++)
			free(m->data[i]);
		free(m->data);
	}
	free(m);
}

