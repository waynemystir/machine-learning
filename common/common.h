#ifndef common_h
#define common_h

typedef struct list list_t;

typedef struct linked_list_node {
	void *data;
	struct linked_list_node *next;
} linked_list_node_t;

typedef struct linked_list linked_list_t;

typedef void (*free_fp)(void *type);

void list_init(list_t **lst, size_t count, void **data, free_fp ffp);
size_t list_len(list_t *lst);
void *list_get(list_t *lst, size_t index);
void list_set(list_t *lst, size_t index, void *value);
void list_swap(list_t *lst, size_t i, size_t j);
void list_shuffle(list_t *lst);
void list_free(list_t *lst);

void linked_list_add_head(linked_list_t *list, void *data, linked_list_node_t **new_node);
void linked_list_add_tail(linked_list_t *list, void *data, linked_list_node_t **new_node);
linked_list_node_t *linked_list_get(linked_list_t *list, size_t index);
void linked_list_split(linked_list_t *list, size_t index, linked_list_t **new_list_1, linked_list_t **new_list_2);

#define SZ_LIST sizeof(list_t)
#define SZ_LLND sizeof(linked_list_node_t)
#define SZ_LL sizeof(linked_list_t)

#endif /* define common_h */
