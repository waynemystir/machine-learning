#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

int main() {
	printf("common_test-0\n");
	long lc = 5;
	list_t *lst;
	list_init(&lst, lc, NULL, NULL);
	for (int i = 0; i < lc; i++) {
		int *x = malloc(sizeof(int));
		*x = i;
		list_set(lst, i, x);
	}

	for (int i = 0; i < lc; i++) {
		int x = *(int*)list_get(lst, i);
		printf("G i(%d)l(%d)\n", i, x);
	}

	printf("----------------------------------------\n");
	list_shuffle(lst);
	for (int i = 0; i < lc; i++) {
		int x = *(int*)list_get(lst, i);
		printf("S i(%d)l(%d)\n", i, x);
	}

	list_t *l1, *l2;
	list_init(&l1, 3, NULL, NULL);
	list_init(&l2, 3, NULL, NULL);

	for (int i = 0; i < 3; i++) {
		char *s1 = malloc(5);
		char *s2 = malloc(5);
		memset(s1, '\0', 5);
		memset(s2, '\0', 5);
		sprintf(s1, "%d%d%d", i, i, i);
		sprintf(s2, "%d%d%d", i+3, i+3, i+3);
		list_set(l1, i, s1);
		list_set(l2, i, s2);
	}

	list_t *l = zip(l1, l2);
	for (int i = 0; i < 3; i++) {
		tuple_t *t = list_get(l, i);
		char *s1 = tuple_get(t, 0);
		char *s2 = tuple_get(t, 1);
		printf("XXX (%s)(%s)\n", s1, s2);
	}
	return 0;
}
