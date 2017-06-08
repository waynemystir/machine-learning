#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

void list_add_head(linked_list_t *list, void *data, linked_list_node_t **new_node) {
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

void list_add_tail(linked_list_t *list, void *data, linked_list_node_t **new_node) {
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

linked_list_node_t *list_get(linked_list_t *list, size_t index) {
	if (!list || list->count < index) return NULL;
	linked_list_node_t *n = list->head;
	for (size_t i = 0; i <= index && n != NULL; i++, n = n->next)
		if (i == index) return n;
	return NULL;
}

void list_split(linked_list_t *list, size_t index, linked_list_t **new_list_1, linked_list_t **new_list_2) {
	if (!list || !new_list_1 || !new_list_2) return;

	linked_list_node_t *new_tail = list_get(list, index);
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
