#ifndef common_h
#define common_h

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

typedef struct list list_t;

typedef struct linked_list_node linked_list_node_t;
typedef struct linked_list linked_list_t;

typedef void (*free_fp)(void *type);

typedef enum ENVIRONMENT {
	ENV_DEBUG = 0,
	ENV_DEV = 1,
	ENV_PROD = 2,
} ENVIRONMENT;

typedef enum LOG_LEVEL {
	LOG_LEVEL_ALL = 0,
	LOG_LEVEL_DEBUG = 1,
	LOG_LEVEL_INFO = 2,
	LOG_LEVEL_WARNING = 3,
	LOG_LEVEL_HIGH = 4,
} LOG_LEVEL;

void list_init(list_t **lst, size_t count, void **data, free_fp ffp);
size_t list_len(list_t *lst);
void *list_get(list_t *lst, long index);
void list_set(list_t *lst, long index, void *value);
void *list_set_get_existing(list_t *lst, size_t index, void *value);
void list_swap(list_t *lst, size_t i, size_t j);
void list_shuffle(list_t *lst);
void list_free(list_t *lst);

void linked_list_add_head(linked_list_t *list, void *data, linked_list_node_t **new_node);
void linked_list_add_tail(linked_list_t *list, void *data, linked_list_node_t **new_node);
linked_list_node_t *linked_list_get(linked_list_t *list, size_t index);
void linked_list_split(linked_list_t *list, size_t index, linked_list_t **new_list_1, linked_list_t **new_list_2);

void set_environment(ENVIRONMENT env);
void set_environment_from_str(char *env_str);
ENVIRONMENT get_environment();
char *get_environment_as_str();
int mllog(LOG_LEVEL ll, int with_time, const char *fmt, ...);

#define SZ_LIST sizeof(list_t)
#define SZ_LLND sizeof(linked_list_node_t)
#define SZ_LL sizeof(linked_list_t)

#endif /* define common_h */
