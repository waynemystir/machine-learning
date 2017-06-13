#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>

#include "common.h"

static ENVIRONMENT environment = ENV_PROD;

struct list {
	void **data;
	size_t count;
	free_fp ffp;
};

typedef struct linked_list_node {
	void *data;
	struct linked_list_node *next;
} linked_list_node_t;

struct linked_list {
	linked_list_node_t *head;
	linked_list_node_t *tail;
	size_t count;
};

void list_init(list_t **lst, size_t count, void **data, free_fp ffp) {
	list_t *l = malloc(SZ_LIST);
	memset(l, '\0', SZ_LIST);
	l->count = count;
	l->ffp = ffp;
	if (lst) *lst = l;
	if (data) {
		l->data = data;
		return;
	}

	l->data = calloc(count, sizeof(void *));
}

size_t list_len(list_t *lst) {
	return lst ? lst->count : 0;
}

void *list_get(list_t *lst, long index) {
	if (!lst || !lst->data || (long)lst->count <= index || abs((long)lst->count) < abs(index)) {
		printf("list_ggggget-NUUULLLLLLLLLL (%s)(%s)(%lu)(%ld)(%s)\n", lst?"G":"B", lst->data?"G":"B", lst->count, index, lst->count <= index?"G":"B");
		return NULL;
	}

	if (index < 0)
		return lst->data[(long)lst->count + index];

	return lst->data[index];
}

void list_set(list_t *lst, size_t index, void *value) {
	if (!lst || !lst->data || lst->count <= index)
		return;
	if (lst->ffp) lst->ffp(lst->data[index]);
	else free(lst->data[index]);
	lst->data[index] = value;
}

void *list_set_get_existing(list_t *lst, size_t index, void *value) {
	if (!lst || !lst->data || lst->count <= index)
		return NULL;
	void *existing = lst->data[index];
	lst->data[index] = value;
	return existing;
}

void list_swap(list_t *lst, size_t i, size_t j) {
	if (!lst || !lst->data || lst->count <= i || lst->count <= j)
		return;

	void *wi = list_get(lst, i);
	void *wj = list_get(lst, j);
	list_set_get_existing(lst, i, wj);
	list_set_get_existing(lst, j, wi);
}

void list_shuffle(list_t *lst) {
	if (!lst || !lst->data) return;

	static int rand_seeded = 0;

	if (!rand_seeded) {
		time_t t;
		srand((unsigned) time(&t));
	}
	rand_seeded = 1;

	size_t last_index = lst->count - 1;
	for (size_t i = 0; i < lst->count; i++)
		list_swap(lst, rand() % lst->count, last_index);
}

void list_free(list_t *lst) {
	if (!lst) return;

	if (lst->data) {
		for (int i = 0; i < lst->count; i++)
			if (lst->ffp) lst->ffp(lst->data[i]);
			else free(lst->data[i]);
		free(lst->data);
	}
	free(lst);
}

void linked_list_add_head(linked_list_t *list, void *data, linked_list_node_t **new_node) {
	if (!list || !data) return;

	linked_list_node_t *node = malloc(SZ_LLND);
	if (!node) return;
	memset(node, '\0', SZ_LLND);
	node->data = data;
	if (new_node) *new_node = node;

	if (!list->head) {
		list->head = node;
		list->tail = node;
	} else {
		node->next = list->head;
		list->head = node;
	}

	list->count++;
}

void linked_list_add_tail(linked_list_t *list, void *data, linked_list_node_t **new_node) {
	if (!list || !data) return;

	linked_list_node_t *node = malloc(SZ_LLND);
	if (!node) return;
	memset(node, '\0', SZ_LLND);
	node->data = data;
	if (new_node) *new_node = node;

	if (!list->head) {
		list->head = node;
		list->tail = node;
	} else {
		list->tail->next = node;
		list->tail = node;
	}

	list->count++;
}

linked_list_node_t *linked_list_get(linked_list_t *list, size_t index) {
	if (!list || list->count < index) return NULL;
	linked_list_node_t *n = list->head;
	for (size_t i = 0; i <= index && n != NULL; i++, n = n->next)
		if (i == index) return n;
	return NULL;
}

void linked_list_split(linked_list_t *list, size_t index, linked_list_t **new_list_1, linked_list_t **new_list_2) {
	if (!list || !new_list_1 || !new_list_2) return;

	linked_list_node_t *new_tail = linked_list_get(list, index);
	if (!new_tail || !new_tail->next) return;
	linked_list_node_t *new_head = new_tail->next;

	linked_list_t *nl1 = malloc(SZ_LL);
	if (!nl1) return;
	memset(nl1, '\0', SZ_LL);
	*new_list_1 = nl1;

	linked_list_t *nl2 = malloc(SZ_LL);
	if (!nl2) return;
	memset(nl2, '\0', SZ_LL);
	*new_list_2 = nl2;

	nl1->head = list->head;
	nl2->head = new_head;
	nl1->tail = new_tail;
	nl2->tail = list->tail;
	new_tail->next = NULL;

	nl1->count = index;
	nl2->count = list->count - index;

	free(list);
}

void set_environment(ENVIRONMENT env) {
	environment = env;
}

void set_environment_from_str(char *env_str) {
	if (strcmp(env_str, "dev") == 0)
		return set_environment(ENV_DEV);

	if (strcmp(env_str, "prod") == 0)
		return set_environment(ENV_PROD);

	printf("Invalid environment name given (%s)\n", env_str);
	exit(1);
}

ENVIRONMENT get_environment() {
	return environment;
}

char *get_environment_as_str() {
	switch (environment) {
		case ENV_DEV: return "ENV_DEV";
		case ENV_PROD: return "ENV_PROD";
		default: return "ENV_UNKNOWN";
	}
}

int mllog(LOG_LEVEL ll, int with_time, const char *fmt, ...) {
	if (environment == ENV_PROD && ll < LOG_LEVEL_WARNING)
		return 0;

	int ret;
	size_t len = strlen(fmt);
	char wes[len + 256];
	memset(wes, '\0', len + 256);

	if (with_time) {
		char time_text_delimit[] = "*";

		time_t rawtime;
		struct tm * timeinfo;

		time ( &rawtime );
		timeinfo = localtime ( &rawtime );
		sprintf(wes, "%s%s ", asctime (timeinfo), time_text_delimit);
		wes[strlen(wes) - strlen(time_text_delimit) - 2] = ' ';
		strcat(wes, fmt);
	} else
		strcpy(wes, fmt);

	/* Declare a va_list type variable */
	va_list myargs;

	/* Initialise the va_list variable with the ... after fmt */
	va_start(myargs, fmt);

	/* Forward the '...' to vprintf */
	ret = vprintf(wes, myargs);

	/* Clean up the va_list */
	va_end(myargs);

	return ret;
}
