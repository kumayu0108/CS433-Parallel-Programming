#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <assert.h>

#define SIZE 100000

#define OP_INSERT 0
#define OP_DELETE 1
#define OP_FIND 2

typedef struct list_node_s {
	int v;
	pthread_mutex_t lock;
	struct list_node_s *next;
} list_node_t;

list_node_t *head = NULL;
list_node_t *tail = NULL;

typedef struct op_node_s {
	int op;
	int v;
} op_node_t;

op_node_t *op_array = NULL;

int num_threads, op_count;

int insert_seq (int v)
{
	list_node_t *prev=NULL, *curr=head;
	list_node_t *node=NULL;
	while (curr && (curr->v < v)) {
		prev = curr;
		curr = curr->next;
	}
	if (curr && (curr->v == v)) return 0;
	node = (list_node_t*)malloc(sizeof(list_node_t));
	assert(node != NULL);
	node->v = v;
	node->next = curr;
	if (prev) prev->next = node;
	else {
		assert(curr == head);
		head = node;
	}
	return 1;
}

int delete_seq (int v)
{
	list_node_t *prev=NULL, *curr=head;
	while (curr && (curr->v < v)) {
                prev = curr;
                curr = curr->next;
        }
	if (curr && (curr->v == v)) {
		if (prev) prev->next = curr->next;
		else {
			assert(curr == head);
			head = curr->next;
		}
		free(curr);
		return 1;
	}
	return 0;
}

int find_seq (int v)
{
	list_node_t *prev=NULL, *curr=head;
        while (curr && (curr->v < v)) {
                prev = curr;
                curr = curr->next;
        }
	if (curr && (curr->v == v)) return 1;
	return 0;
}

int insert_parallel (int v)
{
	list_node_t *prev=NULL, *curr;
        list_node_t *node=NULL;
	if (head) pthread_mutex_lock(&head->lock);
	curr = head;
        while (curr && (curr->v < v)) {
		if (prev != NULL) pthread_mutex_unlock(&prev->lock);
		prev = curr;
                curr = curr->next;
		if (curr) pthread_mutex_lock(&curr->lock);
        }
        if (curr && (curr->v == v)) {
		pthread_mutex_unlock(&curr->lock);
		if (prev != NULL) pthread_mutex_unlock(&prev->lock);
		return 0;
	}
        node = (list_node_t*)malloc(sizeof(list_node_t));
        assert(node != NULL);
        node->v = v;
	pthread_mutex_init(&node->lock, NULL);
        node->next = curr;
        if (prev) prev->next = node;
        else {
                assert(curr == head);
                head = node;
        }
	if (curr) pthread_mutex_unlock(&curr->lock);
        if (prev != NULL) pthread_mutex_unlock(&prev->lock);
        return 1;
}

int delete_parallel (int v)
{
	list_node_t *prev=NULL, *curr;
	if (head) pthread_mutex_lock(&head->lock);
	curr = head;
        while (curr && (curr->v < v)) {
		if (prev != NULL) pthread_mutex_unlock(&prev->lock);
                prev = curr;
                curr = curr->next;
		if (curr) pthread_mutex_lock(&curr->lock);
        }
        if (curr && (curr->v == v)) {
                if (prev) prev->next = curr->next;
                else {
                        assert(curr == head);
                        head = curr->next;
                }
		pthread_mutex_unlock(&curr->lock);
                free(curr);
		if (prev != NULL) pthread_mutex_unlock(&prev->lock);
                return 1;
        }
	if (curr) pthread_mutex_unlock(&curr->lock);
        if (prev != NULL) pthread_mutex_unlock(&prev->lock);
        return 0;
}

int find_parallel (int v)
{
	list_node_t *prev=NULL, *curr;
	if (head) pthread_mutex_lock(&head->lock);
	curr = head;
        while (curr && (curr->v < v)) {
                prev = curr;
                curr = curr->next;
		if (curr) pthread_mutex_lock(&curr->lock);
		pthread_mutex_unlock(&prev->lock);
        }
        if (curr && (curr->v == v)) {
		pthread_mutex_unlock(&curr->lock);
		return 1;
	}
	if (curr) pthread_mutex_unlock(&curr->lock);
        return 0;
}

void *work (void *param)
{
	int i, id = *(int*)(param), dummy=0;

	if (num_threads == 1) {
		for (i=0; i<op_count; i++) {
			if (op_array[i].op == OP_INSERT) dummy += insert_seq(op_array[i].v);
			else if (op_array[i].op == OP_DELETE) dummy += delete_seq(op_array[i].v);
			else dummy += find_seq(op_array[i].v);
		}
	}
	else {	
		for (i=(op_count/num_threads)*id; i<(op_count/num_threads)*(id+1); i++) {
			if (op_array[i].op == OP_INSERT) dummy += insert_parallel(op_array[i].v);
			else if (op_array[i].op == OP_DELETE) dummy += delete_parallel(op_array[i].v);
			else dummy += find_parallel(op_array[i].v);
		}
	}

	printf("Sum: %d\n", dummy);
}	

int main (int argc, char *argv[])
{
	int i, *id, v, op, size;
	FILE *fp;
	list_node_t *node;
	pthread_t *tid;
	pthread_attr_t attr;
	struct timeval tv0, tv1;
	struct timezone tz0, tz1;

	if (argc != 4) {
		printf ("Need number of threads, number of operations, file name to read opeartions from.\n");
		exit(1);
	}
	num_threads = atoi(argv[1]);
	op_count = atoi(argv[2]);
	assert((op_count % num_threads) == 0);	// Simplifying assumption
	op_array = (op_node_t*)malloc(op_count*sizeof(op_node_t));
	assert(op_array != NULL);

	fp = fopen(argv[3], "r");
	assert(fp != NULL);
	i=0;
	while (i < op_count) {
		fscanf(fp, "%d %d", &op, &v);
		op_array[i].op = op;
		op_array[i].v = v;
		i++;
	}
	fclose(fp);

	i=0;
	while (i < SIZE) {
		node = (list_node_t*)malloc(sizeof(list_node_t));
		assert(node != NULL);
		node->v = i;
		pthread_mutex_init(&node->lock, NULL);
		node->next = NULL;
		if (tail) {
			assert(tail->next == NULL);
			tail->next = node;
			tail = node;
		}
		else {
			assert(head == NULL);
			head = node;
			tail = node;
		}
		i++;
	}
		
	tid = (pthread_t*)malloc(num_threads*sizeof(pthread_t));
	id = (int*)malloc(num_threads*sizeof(int));
 	for (i=0; i<num_threads; i++) id[i] = i;

	pthread_attr_init(&attr);

	gettimeofday(&tv0, &tz0);

	for (i=1; i<num_threads; i++) {
		pthread_create(&tid[i], &attr, work, &id[i]);
   	}
	work(&id[0]);
	
	for (i=1; i<num_threads; i++) {
		pthread_join(tid[i], NULL);
	}

	gettimeofday(&tv1, &tz1);

	printf("Time: %ld microseconds, throughput: %.2f ops per second\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec), ((float)op_count*1000000.0)/((tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec)));
	return 0;
}
