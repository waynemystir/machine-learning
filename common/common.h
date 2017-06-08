#ifndef common_h
#define common_h

typedef struct linked_list_node {
	void *data;
	struct linked_list_node *next;
} linked_list_node_t;

typedef struct linked_list {
	linked_list_node_t *head;
	linked_list_node_t *tail;
	size_t count;
} linked_list_t;

void list_add_head(linked_list_t *list, void *data, linked_list_node_t **new_node);
void list_add_tail(linked_list_t *list, void *data, linked_list_node_t **new_node);
linked_list_node_t *list_get(linked_list_t *list, size_t index);

#define SZ_LLND sizeof(linked_list_node_t)
#define SZ_LL sizeof(linked_list_t)

#endif /* define common_h */
