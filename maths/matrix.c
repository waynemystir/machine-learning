#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#include "matrix.h"

struct _matrix {
	size_t num_rows;
	size_t num_cols;
	double **data;
};

void matrix_init(matrix **m, size_t num_rows, size_t num_cols, double **data) {
	matrix *mx = malloc(SZ_MATRIX);
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

size_t matrix_num_rows(matrix *m) {
	if (!m) return 0;
	return m->num_rows;
}

size_t matrix_num_cols(matrix *m) {
	if (!m) return 0;
	return m->num_cols;
}

double matrix_get(matrix *m, size_t row, size_t col) {
	if (!m || !m->data) return DBL_MAX;
	return m->data[row][col];
}

void matrix_set(matrix *m, size_t row, size_t col, double value) {
	if (!m || !m->data) return;
	m->data[row][col] = value;
}

void matrix_product(matrix *m1, matrix *m2, matrix **product) {
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

void matrix_sum(matrix *m1, matrix *m2, matrix **sum) {
	if (!m1 || !m2 || m1->num_rows != m2->num_rows || m1->num_cols != m2->num_cols)
		return;
	matrix_init(sum, m1->num_rows, m1->num_cols, NULL);
	for (int i = 0; i < m1->num_rows; i++)
		for (int j = 0; j < m1->num_cols; j++)
			matrix_set(*sum, i, j, matrix_get(m1, i, j) + matrix_get(m2, i, j));
}

void matrix_elementwise_func_1(matrix *m, elementwise_function_1 ef) {
	if (!m || !m->data)
		return;

	for (int i = 0; i < m->num_rows; i++)
		for (int j = 0; j < m->num_cols; j++)
			ef(&m->data[i][j]);
}

void matrix_elementwise_func_2(matrix *m, elementwise_function_2 ef) {
	if (!m || !m->data)
		return;

	for (int i = 0; i < m->num_rows; i++)
		for (int j = 0; j < m->num_cols; j++)
			ef(m->data[i][j], &m->data[i][j]);
}

void matrix_elementwise_func_3(matrix *m, elementwise_function_3 ef) {
	if (!m || !m->data)
		return;

	for (int i = 0; i < m->num_rows; i++)
		for (int j = 0; j < m->num_cols; j++)
			matrix_set(m, i, j, ef());
}

void matrix_print(matrix *m) {
	if (!m) {
		printf("matrix_print-NULL\n");
		return;
	}
	for (int i = 0; i < m->num_rows; i++) {
		if (i != 0) printf("\n");
		for (int j = 0; j < m->num_cols; j++)
			printf("%.1f  ", matrix_get(m, i, j));
	}
	printf("\n");
}
