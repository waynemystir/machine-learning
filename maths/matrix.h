#ifndef matrix_h
#define matrix_h

typedef struct _matrix matrix;
typedef void (*elementwise_function_1)(double *ret_value);
typedef void (*elementwise_function_2)(double arg, double *ret_value);
typedef double (*elementwise_function_3)(void);
typedef double (*elementwise_function_4)(double arg);

void matrix_init(matrix **m, size_t num_rows, size_t num_cols, double **data);
size_t matrix_num_rows(matrix *);
size_t matrix_num_cols(matrix *);
double matrix_get(matrix *m, size_t row, size_t col);
void matrix_set(matrix *m, size_t row, size_t col, double value);
void matrix_product(matrix *m1, matrix *m2, matrix **product);
void matrix_sum(matrix *m1, matrix *m2, matrix **sum);
void matrix_elementwise_func_1(matrix *m, elementwise_function_1 ef);
void matrix_elementwise_func_2(matrix *m, elementwise_function_2 ef);
void matrix_elementwise_func_3(matrix *m, elementwise_function_3 ef);
void matrix_elementwise_func_4(matrix *m, elementwise_function_4 ef);
void matrix_print(matrix *m, int precision, int zero_precision);

#define SZ_MATRIX sizeof(matrix)

#endif /* define matrix_h */
