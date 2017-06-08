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
