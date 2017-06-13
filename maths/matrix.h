#ifndef matrix_h
#define matrix_h

typedef struct matrix matrix_t;
typedef void (*elementwise_function_1)(double *ret_value);
typedef void (*elementwise_function_2)(double arg, double *ret_value);
typedef double (*elementwise_function_3)(void);
typedef double (*elementwise_function_4)(double arg);

void matrix_init(matrix_t **m, size_t num_rows, size_t num_cols, double **data);
void matrix_zero_init(matrix_t **m, size_t num_rows, size_t num_cols);
size_t matrix_num_rows(matrix_t *);
size_t matrix_num_cols(matrix_t *);
double matrix_get(matrix_t *m, size_t row, size_t col);
void matrix_set(matrix_t *m, size_t row, size_t col, double value);
void matrix_product(matrix_t *m1, matrix_t *m2, matrix_t **product);
void matrix_product_elementwise(matrix_t *m1, matrix_t *m2, matrix_t **product);
void matrix_product_scalar(matrix_t *m, double scalar);
matrix_t *matrix_product_scalar_ret(matrix_t *m, double scalar);
void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t **sum);
void matrix_elementwise_func_1(matrix_t *m, elementwise_function_1 ef);
void matrix_elementwise_func_2(matrix_t *m, elementwise_function_2 ef);
void matrix_elementwise_func_3(matrix_t *m, elementwise_function_3 ef);
void matrix_elementwise_func_4(matrix_t *m, elementwise_function_4 ef);
matrix_t *matrix_elementwise_func_4_ret(matrix_t *m, elementwise_function_4 ef);
matrix_t *matrix_transpose(matrix_t *m);
double matrix_argmax(matrix_t *m, size_t *row, size_t *col);
void matrix_print(matrix_t *m, int precision, int zero_precision);
void matrix_free(matrix_t *m);

#define SZ_MATRIX sizeof(matrix_t)

#endif /* define matrix_h */
