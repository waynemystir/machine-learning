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

struct tuple {
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
	if (!lst || !lst->data) {
		printf("list_ggggget-NUUULLLLLLLLLL (%s)(%s)\n", lst?"G":"B", lst->data?"G":"B");
		exit(1);
	}

	long calced_index = index < 0 ? (long)lst->count + index : index;
	if (calced_index < 0) {
		printf("list_get ERROR: calced_index (%ld) < 0\n", calced_index);
		exit(1);
	}
	if (calced_index >= (long)lst->count) {
		printf("list_get ERROR: calced_index (%ld) >= lst->count (%lu)\n", calced_index, lst->count);
		exit(1);
	}

	return lst->data[calced_index];
}

void list_set(list_t *lst, long index, void *value) {
	if (!lst || !lst->data) {
		printf("list_set: Something went wrong\n");
		exit(1);
		return;
	}

	long calced_index = index < 0 ? (long)lst->count + index : index;
	if (calced_index < 0) {
		printf("list_set ERROR: calced_index (%ld) < 0\n", calced_index);
		exit(1);
	}
	if (calced_index >= (long)lst->count) {
		printf("list_set ERROR: calced_index (%ld) >= lst->count (%lu)\n", calced_index, lst->count);
		exit(1);
	}

	if (lst->ffp) lst->ffp(lst->data[calced_index]);
	else free(lst->data[calced_index]);
	lst->data[calced_index] = value;
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
		list_swap(lst, rand() / (RAND_MAX / lst->count + 1), last_index);
//		list_swap(lst, rand() / (RAND_MAX / (lst->count - i) + 1), last_index);
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

tuple_t *tuple_init(size_t count, void **data, free_fp ffp) {
	tuple_t *t = malloc(SZ_TPLE);
	memset(t, '\0', SZ_TPLE);
	t->count = count;
	t->ffp = ffp;
	if (data) {
		t->data = data;
		return t;
	}

	t->data = calloc(count, sizeof(void *));
	return t;
}

void *tuple_get(tuple_t *t, long index) {
	if (!t || !t->data) {
		printf("t_ggggget-NUUULLLLLLLLLL (%s)(%s)\n", t?"G":"B", t->data?"G":"B");
		exit(1);
	}

	long calced_index = index < 0 ? (long)t->count + index : index;
	if (calced_index < 0 || calced_index >= (long)t->count) exit(1);

	return t->data[calced_index];
}

void tuple_set(tuple_t *t, long index, void *value) {
	if (!t || !t->data) {
		printf("t_set: Something went wrong\n");
		exit(1);
		return;
	}

	long calced_index = index < 0 ? (long)t->count + index : index;
	if (calced_index < 0 || calced_index >= (long)t->count) exit(1);

	if (t->ffp) t->ffp(t->data[calced_index]);
	else free(t->data[calced_index]);
	t->data[calced_index] = value;
}

void tuple_free(tuple_t *t) {
	if (!t) return;

	if (t->data) {
		for (int i = 0; i < t->count; i++)
			if (t->ffp) t->ffp(t->data[i]);
			else free(t->data[i]);
		free(t->data);
	}
	free(t);
}

list_t *zip(list_t *l1, list_t *l2) {
	if (!l1 || !l2) {
		printf("zip problem - given list is NULL\n");
		exit(1);
	}

	if (l1->count != l2->count) {
		printf("zip problem - counts not equal\n");
		exit(1);
	}

	list_t *l;
	list_init(&l, l1->count, NULL, (free_fp)tuple_free);
	for (long i = 0; i < l1->count; i++) {
		tuple_t *t = tuple_init(2, NULL, l1->ffp);
		tuple_set(t, 0, list_get(l1, i));
		tuple_set(t, 1, list_get(l2, i));
		list_set(l, i, t);
	}

	return l;
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
	if (strcmp(env_str, "debug") == 0)
		return set_environment(ENV_DEBUG);

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
		case ENV_DEBUG: return "ENV_DEBUG";
		case ENV_DEV: return "ENV_DEV";
		case ENV_PROD: return "ENV_PROD";
		default: return "ENV_UNKNOWN";
	}
}

int mllog(LOG_LEVEL ll, int with_time, const char *fmt, ...) {
	if (environment == ENV_DEV && ll < LOG_LEVEL_INFO)
		return 0;
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
