/**notice: the thread number is determined by openmp instead of input paramemter,
 * use blocking two pass algorithm to avoid cache miss**/
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#define _BSD_SOURCEs
#define max(a,b) ( (a >=b) ? a:b)
#define  min(a,b) ((a <= b) ? a:b )
char *output_hpc = "correlation_matrix.txt"; 
char *input_hpc = "data.txt";
char *directory_name = "/project/parallel/tao_tang/all_results/omp3_29_Aug";
#define VECTOR_LENGTH 2725784

int interval;

char *mark="";

/** global variable used to store input data and related information**/
double *vectors_mean;
double *vectors_var;
double *matrix;
double **blocks;
double *blocks_mv;
double **blocks_mean;
double **blocks_variance;

int *blocks_size;
/** id of the first vector of each block **/
int *blocks_vector_id;
double *result_buffer;
/** stored in the format of correaltion matrix, results[i][j] indicates the result between ith and (i+j+1)th vector, e.g., results[2][3] is correaltion between vector 2 and 6**/
double **results;
int n_threads,N;
pthread_barrier_t barrier;
pthread_mutex_t finished_mutex; /* global mutex */
int n_threads = 0;
/** length of each vector**/
//int M = VECTOR_LENGTH;
int M;

/**calculation of between two given vectors**/
double correlation(double *x, double *y, int len, double x_mean,double x_var, double y_mean, double y_var)
{
	double cov = 0;
	int i;
	for( i = 0; i < len ; i++)
		cov += (x[i]-x_mean)*(y[i]-y_mean);
		
	cov = cov/(len-1);
	return cov/(sqrt(x_var)*sqrt(y_var));
}

/** covariance between two vectors**/
double update_covariance(double *x, double *y,int len,double x_mean,double y_mean)
{
	double cov = 0;
	int i;
	for( i = 0; i < len ; i++)
		cov += (x[i]-x_mean)*(y[i]-y_mean);
	
	return cov;	
}

/**calculate mean and variance of given vector with given length**/
void vector_mean_variance(double *vector,int len, double *i_mean,double *variance)
{
	int i;
	double var = 0,mean = 0;
	for(i=0; i < len; i++)
		mean += vector[i];

	mean = mean/len;
	
	for(i = 0; i < len; i ++)
		var += (vector[i]-mean)*(vector[i]-mean);
	
	var = var/(len-1);
	
	(*i_mean) = mean;
	*variance = var;
	return;
}


void read_files(double *matrix, int initial_id,int n, int m)
{
	int l = 100;
	char buffer[l];
	puts("loading file");
	char *name = malloc(sizeof(char)*100);
	int i,j,garbage,count = 0;
	FILE *input ;
	for(i = initial_id; i < initial_id + n; i++)
	{
		sprintf(name,"/project/parallel/tao_tang/src/%d.bed_signal.txt",i); 
		
		if(!(input = fopen(name,"r")))
		{
			printf("soruce file %s doesn't exist,try to use artifical\n",name );
			sprintf(name,"/project/parallel/tao_tang/src/%d.bed_signal_artifical.txt",i); 
			input = fopen(name,"r");
		}
		/**skip the first row**/
		fgets(buffer,l,input);	
		for(j = 0 ; j < m; j++)
			fscanf(input,"%s %d %lf",buffer, &garbage,matrix+(count++));
		
		fclose(input);	
		printf("read file %d\n",i);
	}
	free(name);
	return;
}

void read_synthetic_data(double *matrix, int initial_id,int n, int m)
{
	int l = 100;
	char buffer[l];
	puts("loading file");
	char *name = malloc(sizeof(char)*100);
	int i,j,garbage,count = 0;
	FILE *input ;
	for(i = initial_id; i < initial_id + n; i++)
	{
		sprintf(name,"/project/parallel/tao_tang/synthetic/data%d.txt",i); 
		for(j = 0 ; j < m; j++)
			fscanf(input,"%lf",matrix+(count++));
		
		fclose(input);	
		printf("read file %d\n",i);
	}
	free(name);
	return;
}


int main (int argc, char **argv)
{
	int n_threads,tid,n_core;

	mkdir(directory_name, 0777);
		
	int i,j,k;
	/** input paramemters: number of cores, number of threads(noramlly the same as number of cores), N:number of vectors, M: length of each vector, interval: size of interval **/
	n_core = atoi(argv[1]);  
	n_threads = n_core;
	N = atoi(argv[2]);  

	M = VECTOR_LENGTH; 
	if(argc > 3)
		M = atoi(argv[3]);

	if(argc > 4)
		mark = argv[4];
		
	int chunk_size = 1;	
	if(argc > 5)
		chunk_size = atoi(argv[5]);
		
	int interval = M;
	if(argc > 6)
		interval = atoi(argv[6]);
		
	matrix = malloc(sizeof(double) * M *N);	
	read_files(matrix,1,N,M);
	printf("matrix content:\n");
	for(i = 0; i < 20; i++)
		printf(" %lf",matrix[i]);
	puts("");
	
	vectors_mean = malloc(sizeof(double)*N);
	vectors_var = malloc(sizeof(double)*N);

	result_buffer = malloc(sizeof(double) * N * (N-1)/2);
	results = malloc(sizeof(double *) * (N-1));	
	int count = 0;
	
	for (i =0; i < N * (N-1)/2; i++)
		result_buffer[i] = 0;
	
	for(i = 0; i < N-1; i++)
	{
		results[i] = result_buffer + count;
		count += (N-1-i);
	}
	
	#pragma omp parallel 
	{
		if(omp_get_thread_num()==0)
			n_threads = omp_get_num_threads();

    }
	printf("Number of threads = %d\n", n_threads);
	printf("calculate mean and variance\n");
	struct timeval t0, t1, dt;
	gettimeofday(&t0, NULL);

	#pragma omp parallel for schedule(dynamic,chunk_size) private(tid)
	for(i=0; i < N; i++)
	{	
		tid = omp_get_thread_num();
		printf("calculate mean thread %d\n",tid);
		vector_mean_variance(matrix+i*M,M,vectors_mean+i,vectors_var+i);
	}
	printf("begin calculation\n");

	int interval_number = M/interval;
	int k_m;
	for(k=0; k< interval_number; k++)	
	{
		k_m = k * interval;	
		#pragma omp parallel for schedule(dynamic,chunk_size) private(tid,i,j)
		for(i = 0; i < N-1; i++)
		{
			for(j = i+1; j < N; j++)		
				results[i][j-i-1] += update_covariance(matrix+i*M+k_m, matrix+j*M+k_m,interval, vectors_mean[i],vectors_mean[j]);
		}
	}
	
	k_m = interval_number * interval;
	#pragma omp parallel for schedule(dynamic,chunk_size) private(tid,i,j)
	for(i = 0; i < N-1; i++)
	{
		for(j = i+1; j < N; j++)
		{		
			results[i][j-i-1] += update_covariance(matrix+i*M+k_m, matrix+j*M+k_m,M%interval, vectors_mean[i],vectors_mean[j]);
			double cov = results[i][j-i-1]/(M-1);
			results[i][j-i-1] = cov/(sqrt(vectors_var[i]) * sqrt(vectors_var[j]));
		}
	}	

	gettimeofday(&t1, NULL);
	timersub(&t1, &t0, &dt);	

	n_core = n_threads;
	/** write correaltion matrix to output file**/
	char *fp = malloc(sizeof(char) * 200);
	sprintf(fp,"%s/result_%dvectors_%dthreads_chunk%d_interval%d_%s.txt\n",directory_name,N,n_threads,chunk_size,interval,mark);
	printf("file path: %s",fp);
	FILE *f = fopen(fp, "w");
	fprintf(f,"runningtime: %d.%06d seconds\n",(int)dt.tv_sec,(int)dt.tv_usec);
	fprintf(f,"complexity: %d\n ", N*(N-1)/2);    
	fprintf(f,"core_number: %d\n ", n_core);    
	fprintf(f,"thread_number: %d\n",n_threads);
	fprintf(f,"interval_size: %d\n",interval);
	fprintf(f,"chunk_size size: %d\n",chunk_size);	
	free(fp);		
	for(i=0; i<N-1; i++) 
	{
		for(j= 0;j < N-i-1; j++)
			fprintf(f,"%lf ",results[i][j]);
		fprintf(f,"\n");
	}
	fclose(f);
	puts("final correaltion matrix is:");
	for(i = 0; i < N-1; i++)
	{
		puts("");
		for(j=0; j < N-1-i; j++)
			printf("%lf ",results[i][j]);
	} 
	puts("");

	pthread_barrier_destroy(&barrier);
	pthread_exit (NULL);		

	free(matrix);
	free(results);
	free(result_buffer);
	free(vectors_mean);
	free(vectors_var);
	return 0;
}
