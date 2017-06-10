#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common.h"

struct list {
	size_t count;
	void **data;
};

struct linked_list {
	linked_list_node_t *head;
	linked_list_node_t *tail;
	size_t count;
};

void list_init(list_t **lst, size_t count, void **data) {
	list_t *l = malloc(SZ_LIST);
	memset(l, '\0', SZ_LIST);
	l->count = count;
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

void *list_get(list_t *lst, size_t index) {
	if (!lst || !lst->data || lst->count <= index)
		return NULL;

	return lst->data[index];
}

void list_set(list_t *lst, size_t index, void *value) {
	if (!lst || !lst->data || lst->count <= index)
		return;

	lst->data[index] = value;
}

void list_swap(list_t *lst, size_t i, size_t j) {
	if (!lst || !lst->data || lst->count <= i || lst->count <= j)
		return;

	void *wi = list_get(lst, i);
	void *wj = list_get(lst, j);
	list_set(lst, i, wj);
	list_set(lst, j, wi);
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

void list_free(list_t *lst, free_fp ffp) {
	if (!lst) return;

	if (lst->data) {
		for (int i = 0; i < lst->count; i++)
			if (ffp) ffp(lst->data[i]);
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
