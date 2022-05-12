#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define OP_INSERT 0
#define OP_DELETE 1
#define OP_FIND 2

int main (int argc, char **argv)
{
	int op_count, inserts, deletes, finds, max_val, op, v;
	FILE *fp;

	if (argc != 5) {
		printf("Need op count, percentage of inserts, percentage of deletes, max value.\nAborting...\n");
		exit(0);
	}

	op_count = atoi(argv[1]);
	inserts = (atoi(argv[2])*op_count)/100;
	deletes = (atoi(argv[3])*op_count)/100;
	finds = op_count - inserts - deletes;
	max_val = atoi(argv[4]);
	max_val++;

	fp = fopen("list_ops.txt", "w");
	assert(fp != NULL);
	while (inserts || deletes || finds) {
		op = random() % 3;
		v = random() % max_val;
		if ((op == OP_INSERT) && inserts) {
			inserts--;
			fprintf(fp, "%d %d\n", op, v);
		}
		else if ((op == OP_DELETE) && deletes) {
			deletes--;
			fprintf(fp, "%d %d\n", op, v);
		}
		else if ((op == OP_FIND) && finds) {
			finds--;
			fprintf(fp, "%d %d\n", op, v);
		}
	}
	fclose(fp);
	return 0;
}
