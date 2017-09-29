#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>


int main(int argc, char **argv)
{	
	int seed = time(NULL);
    srand(seed);
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    	
	int range = 1;
	char *fp = malloc(sizeof(char) * 150);
	FILE *f;
	int i,j;
	for(i = 0; i < N; i++)
	{
		sprintf(fp,"synthetic/data%d.txt",i);
		f = fopen(fp, "w");
		for(j = 0; j < M;j ++)
			fprintf(f,"%lf\n", (double)rand()/(double)(RAND_MAX/range));
		
		fclose(f);
	}
	
	free(fp);
	return 0;
}
