/**
 * pure MPI version, non-blocking communication**/
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
//message used to exchange block use TAG1,send/recv inner product use TAG2,exchange mean/variance use block3
#define TAG1  0
#define TAG2  1
#define TAG3  2

char *output_hpc = "correlation_matrix.txt"; 
char *input_hpc = "data.txt";
char *directory_name = "/project/parallel/tao_tang/all_results/ring_ordering1_18_Sep";
char *mark="";
/** length of each vector**/
#define VECTOR_LENGTH 2725784
/** length of each interval in pairwise algorithm**/
int interval = 10000;
int n_node = 1;
int core_per_node = 1,n_threads;
int numprocs,N,v_b,steps;
pthread_barrier_t barrier;
pthread_barrier_t barrier1;
/**size of each vector**/
int	M = 49500;
//int	M = VECTOR_LENGTH;

//my_barrier barrier;
//measure time
struct timeval t0, t1, dt, t_start, t_end, t_pass;
double computation_time = 0;
double total_computation_time = 0;
int chunk_size = 3;
// all_block_id:steps*numprocs matrix
void set_id_each_iteration(int **all_block1_id,int **all_block2_id,int numprocs,int steps)
{
	int i,j;
	int *block_buffer = malloc(sizeof(int)*numprocs);
	//step 0
	for(i =0; i < numprocs; i++)
	{
		all_block1_id[0][i] = 2*i;
		all_block2_id[0][i] = 2*i +1;
	}

	for(i = 1; i < steps; i++)
	{
		memcpy(all_block1_id[i],all_block1_id[i-1], sizeof(int) * numprocs);
		memcpy(block_buffer,all_block2_id[i-1], sizeof(int) * numprocs);
		block_buffer[numprocs - 1- (i-1)/2] = all_block1_id[i-1][numprocs - 1- (i-1)/2];
		all_block1_id[i][numprocs - 1- (i-1)/2] = all_block2_id[i-1][numprocs - 1- (i-1)/2];;
		
		for(j =0 ; j < numprocs; j++)
			all_block2_id[i][j] = block_buffer[(j+1)%numprocs];		
	}
	
	free(block_buffer);
	
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

/** covariance between two vectors**/
double update_covariance(double *x, double *y,int len,double x_mean,double y_mean)
{
	double cov = 0;
	int i;
	for( i = 0; i < len ; i++)
		cov += (x[i]-x_mean)*(y[i]-y_mean);
	
	return cov;	
}

/**used to get start and end id of a worker when need to assign n jobs eaqually to some workers(thread or processors)**/
void get_start_end_id(int *s_id,int *e_id, int work_number, int worker_number, int worker_id)
{
	int avg = work_number/worker_number;
	int remainder = work_number%worker_number;
	(*s_id) = worker_id * avg + min(worker_id,remainder) * 1;
	(*e_id) =  worker_id < remainder ? (*s_id) + avg +1 : (*s_id) + avg;
	return;
}

void write_results(double **results, int N,struct timeval dt)
{
	int i,j;
	
	puts("final correaltion matrix is:");
	for(i = 0; i < N-1; i++)
	{
		puts("");
		for(j=0; j < N-1-i; j++)
			printf("%lf ",results[i][j]);
	}	
	puts("");
	char *fp = malloc(sizeof(char) * 200);
	sprintf(fp,"%s/result_%dvectors_%dnodes_%dcores_interval%d_%s.txt",directory_name,N,n_node,numprocs,interval,mark);
	FILE *f = fopen(fp, "w");
	fprintf(f,"runningtime: %d.%06d seconds\n",(int)dt.tv_sec,(int)dt.tv_usec);
	fprintf(f,"complexity: %d\n ", N*(N-1)/2);    
	fprintf(f,"nodenumber: %d\n ",n_node);
	fprintf(f,"core_pernode: %d\n ",core_per_node);
	fprintf(f,"interval_size: %d\n ",interval);	 
	fprintf(f,"total_computation_time: %lf\n ",total_computation_time);
	fprintf(f,"computation_time: %lf\n ",computation_time);
		
	for(i=0; i<N-1; i++) 
	{
		for(j= 0;j < N-i-1; j++)
			fprintf(f,"%lf ",results[i][j]);
		fprintf(f,"\n");
	}
	
	printf("file path is %s\n",fp);	
	fclose(f);
	free(fp);

	
	return; 	
}

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

void single_block_calculation(double *block,int block_size,double *result,int result_count,double *block_mean,double *block_var)
{

	gettimeofday(&t_start, NULL);	
	
	int i,j,k,k_m,index;
	int interval_number = M/interval;	
	
	for(k=0; k< interval_number; k++)
	{
		k_m = k*interval;		
		for(i =0; i < block_size-1; i++)
		{
			for(j = i+1; j < block_size; j++)
			{
				index = result_count+i *( 2*block_size -i -1)/2   + (j-i-1);		
				result[index] += update_covariance(block+i*M+ k_m, block+j*M+k_m, interval,block_mean[i],block_mean[j]);
			}
		}	
	}

	k_m = interval*interval_number;	
	for(i =0; i < block_size-1; i++)
	{
		for(j = i+1; j < block_size; j++)
		{
			index = result_count+i *( 2*block_size -i -1)/2   + (j-i-1);	
			result[index] += update_covariance(block+i*M+ k_m, block+j*M+k_m, M%interval,block_mean[i],block_mean[j]);
			result[index]= result[index]/(M-1);
			/** corr(x,y) = cov(s,y)/ sqrt(var(x) * var(y)**/
			result[index]= result[index]/(sqrt(block_var[i]) * sqrt(block_var[j]));
		}
	}
	
	gettimeofday(&t_end, NULL);
	timersub(&t_end, &t_start, &t_pass);
	computation_time += (int)t_pass.tv_sec + ((int)t_pass.tv_usec)/1000000.0;	
	
}

void pair_block_calculation(double *b1,double *b2,int b1_size,int b2_size,double *result,int result_count,double *b1_mean,
							double *b1_var,double *b2_mean,double *b2_var)
{
	gettimeofday(&t_start, NULL);	
	int i,j,k,k_m,index;
	int interval_number = M/interval;
	for(k=0; k< interval_number; k++)	
	{
		k_m = k * interval;		
		for(i = 0; i < b1_size; i++)
		{
			for(j = 0; j < b2_size; j++)		
				result[result_count+i*b2_size+j] += update_covariance(b1+i*M+k_m, b2+j*M+k_m,interval, b1_mean[i],b2_mean[j]);
		}
	}
	
	k_m = interval_number * interval;	
	for(i = 0; i < b1_size; i++)
	{		
		for(j = 0; j< b2_size; j++)
		{
			index = result_count+i*b2_size+j;		
			result[index] += update_covariance(b1+i*M+k_m, b2+j*M+k_m,M%interval, b1_mean[i],b2_mean[j]);
			double cov = result[index]/(M-1);
			result[index] = cov/(sqrt(b1_var[i]) * sqrt(b2_var[j]));

		}
	}
	
	gettimeofday(&t_end, NULL);
	timersub(&t_end, &t_start, &t_pass);
	computation_time += (int)t_pass.tv_sec + ((int)t_pass.tv_usec)/1000000.0;		
}


/**
void *thread_work_seq(void *thrd_arg)
{
	int i,j,k;


	struct thrd_data *t_data;	

	t_data = (struct thrd_data*)thrd_arg;
		
	int t_id = t_data -> t_id;
	double *matrix = t_data -> block1;
	double *mean = t_data -> block1_mean;
	double *var = t_data -> block1_var;
	double *result = t_data -> result;
	
	int pairs_number = N*( N-1)/2;
	int start_id=0, end_id=0;
	int interval_number = M/interval;
	
	get_start_end_id(&start_id,&end_id,pairs_number,n_threads,t_id);
	int work_this_thread = end_id - start_id;

	
	printf("start id: %d  end id: %d\n",start_id,end_id);
	
	int count = 0;
	//correaltion of block1
	for(k=0; k< interval_number; k++)
	{
		for(i =0; i < N-1; i++)
		{
			for(j = i+1; j < N; j++)
			{
				if( count >= start_id && count < end_id)
					result[count] += update_covariance(matrix+i*M+ k*interval, matrix+j*M+k*interval, interval,mean[i],mean[j]);
		
				count++;
			}
			
		}	
		count=0;
	}

	for(i =0; i < N-1; i++)
	{
		for(j = i+1; j < N; j++)
		{
			if( count >= start_id && count < end_id)
			{
				result[count] += update_covariance(matrix+i*M+ interval_number*interval, matrix+j*M+interval_number*interval, 
				M%interval,mean[i],mean[j]);
				result[count]= result[count]/(M-1);
				// corr(x,y) = cov(s,y)/ sqrt(var(x) * var(y)
				result[count]= result[count]/(sqrt(var[i]) * sqrt(var[j]));
			
			}
			count++;
		}
	}	

}


void sequential_process(double *matrix,int M, int N)
{
	int i,j,count=0;	
	int T_size = M/interval + 1;
	double *mean = malloc(sizeof(double) * N);
	double *variance = malloc(sizeof(double) * N);
	double *result = malloc(sizeof(double)*N *(N-1)/2);
	for(i = 0; i < N*(N-1)/2; i++)
		result[i] = 0;
	
	
	for ( i = 0; i < N; i++)
		vector_mean_variance(matrix + i * M, M, mean+i, variance+i);
		
	pthread_attr_t attr;
//	my_barrier_init (&barrier);
	pthread_barrier_init(&barrier,NULL,n_threads+1);
	pthread_barrier_init(&barrier1,NULL,n_threads);
	pthread_t *thread_id = (pthread_t *)malloc(sizeof(pthread_t)*n_threads);
	struct thrd_data *t_arg = (struct thrd_data *)malloc(sizeof(struct thrd_data)*n_threads);	
	
	for(i = 0; i < n_threads; i++)
	{
		//define thread argument
		t_arg[i].t_id = i;
		t_arg[i].block1 = matrix;
		t_arg[i].block1_mean = mean;
		t_arg[i].block1_var = variance;
		t_arg[i].result = result;
	}		
	
	gettimeofday(&t0, NULL);
		

	for(i = 0; i < n_threads; i++)
		pthread_create(thread_id+i,NULL, &thread_work_seq,t_arg+i);		
		
	
	for( i = 0; i < n_threads; i++)
	{
		pthread_join(thread_id[i],NULL);
		printf("thread  %d finished\n",i);	
	}
	
	gettimeofday(&t1, NULL);
	timersub(&t1, &t0, &dt);	
		
	double **correlation_matrix = malloc(sizeof(double *) * (N-1));	
	
	count = 0;
	for( i = 0 ; i < N-1; i++)
	{
		correlation_matrix[i] = result + count;
		count += (N-1-i);
	}
		
	write_results(correlation_matrix,N,dt);
	
	free(correlation_matrix);			
	free(result);
	free(mean);
	free(variance);
	free(thread_id);
	free(t_arg);	
	return;
} **/

//note, this algorithm only for condition that numporcs is odd 
int main(int argc, char **argv)
{
	int myid,length;
	//determine rank of each preocessor
	MPI_Init(&argc,&argv);	
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	
	int i,j,k,l,iter,count = 0;

	n_node = atoi(argv[1]);
	core_per_node = atoi(argv[2]);
	N = atoi(argv[3]);	
	if(argc > 4)
		M = atoi(argv[4]);
	if(argc > 5)
		interval = atoi(argv[5]);
	if(argc > 6)
		mark = argv[6];

	n_threads = core_per_node;
	mkdir(directory_name, 0777);

/**	
	if(numprocs == 1)
	{
		double *matrix = malloc(sizeof(double)* N * M);
		read_synthetic_data(matrix,1,N,M);
		sequential_process(matrix,M,N);
		free(matrix);
		return;
	}		
**/	
	steps = 2 * numprocs - 1;
	int block_number = 2 * numprocs;	
		
	MPI_Status stat,stat1;
	MPI_Request req,req1;
	
	v_b = N / block_number;
	int remainder = N%block_number;
	//each process contains two blocks,each block contain v_b or v_b+1 vectors
	double *block1 = malloc(sizeof(double) * (v_b+1) * M);
	double *block2 = malloc(sizeof(double) * (v_b+1) * M);	
	double *block_buffer = malloc(sizeof(double) * (v_b+1) * M);	
		
	//mean and variance of block1,2
	double *block1_mv = malloc(sizeof(double)* 2*(v_b+1)); 
	double *block1_mean = (block1_mv+0);
	double *block1_variance = (block1_mv+ (v_b+1));	
	double *block2_mv = malloc(sizeof(double)* 2*(v_b+1)); 
	double *block2_mean = (block2_mv+0);
	double *block2_variance = (block2_mv+ (v_b+1));
	double *block_mv_buffer = malloc(sizeof(double)* 2*(v_b+1)); 

	printf("begin to set id\n");
	//the ids of second block each process keeps in each iteration, (frist block is fixed during computation)
	int **all_block1_id = malloc(sizeof(int *) * steps);
	int **all_block2_id = malloc(sizeof(int *) * steps);

	for( i = 0; i < steps; i++)
	{
		all_block1_id[i] = malloc(sizeof(int) * numprocs);
		all_block2_id[i] = malloc(sizeof(int) * numprocs);
	}

	set_id_each_iteration(all_block1_id,all_block2_id,numprocs,steps );

	if(myid == 0)
	{
		puts("blocks id:\n");
		for( i = 0; i < steps; i++)
		{
			puts("");
			for(j = 0; j < numprocs; j++)
				printf("%d ",all_block1_id[i][j]);

			puts("");
			for(j = 0; j < numprocs; j++)
				printf("%d ",all_block2_id[i][j]);
				
			puts("");			
				
		}
	}


	int *blocks1_id = malloc(sizeof(int) * steps);
	int *blocks2_id = malloc(sizeof(int) * steps);
	
	for(i = 0; i < steps; i++)
	{
		blocks1_id[i] = all_block1_id[i][myid];
		blocks2_id[i] = all_block2_id[i][myid];
	}
	
	//number of vectors each block contain
	int *blocks_size = malloc(sizeof(int)*block_number);
	int *blocks_vector_id = malloc(sizeof(int)*block_number);

	// id of the first vectors each block contain 
	blocks_vector_id[0] = 0;
	blocks_size[0] = (0 < remainder) ? v_b+1 : v_b;

	printf("id of the first vector in each process:");		
	for(i = 1; i < block_number; i++)
	{
		blocks_size[i] = (i < remainder) ? v_b+1 : v_b;
		blocks_vector_id[i] = blocks_vector_id[i-1] + blocks_size[i-1];
		printf(" %d",blocks_vector_id[i]);
	}
	
	puts("");
	printf("blocks size:");
	for( i = 0; i < block_number; i++)
		printf(" %d ", blocks_size[i]);	
	puts("");
	int v_b_buffer = v_b+1;
	//maximum number of size of results computed by each process
	int max_results_size = v_b_buffer*(v_b_buffer-1)/2 + v_b_buffer * v_b_buffer * steps;
	//store results computed by this process
	double *result_each_process = malloc( sizeof(double) * max_results_size );
	for(i = 0; i < max_results_size; i++)
		result_each_process[i] = 0;
	
	puts("loading file");
	//each process read the a part of the whole matrix
	int block1_id = blocks1_id[0];
	int block2_id = blocks2_id[0];
 	read_files(block1,blocks_vector_id[block1_id] + 1,blocks_size[block1_id], M);
	read_files(block2,blocks_vector_id[block2_id] + 1, blocks_size[block2_id], M);
	
	gettimeofday(&t0, NULL);	
	
	for(i = 0; i <blocks_size[block1_id]; i++)
		vector_mean_variance(block1+i*M,M,block1_mean+i,block1_variance+i);
		
	for(i = 0; i <blocks_size[block2_id]; i++)
		vector_mean_variance(block2+i*M,M,block2_mean+i,block2_variance+i);		
		
	int result_count = 0;
	int result_count_buffer = result_count;
	
	single_block_calculation(block1,blocks_size[block1_id],result_each_process,result_count,block1_mean,block1_variance);
	result_count += blocks_size[block1_id] * (blocks_size[block1_id]-1)/2;
	single_block_calculation(block2,blocks_size[block2_id],result_each_process,result_count,block2_mean,block2_variance);
	result_count += blocks_size[block2_id] * (blocks_size[block2_id]-1)/2;
	
//	printf(" a line 257 process %d,blocks2_id %d\n",myid,blocks2_id[0]);		
	double *data_buffer;
	printf("line 454 process %d\n",myid);
	for( iter = 0; iter < steps; iter++)
	{	
//		int swapper_id = numprocs - iter/2;	
		int swapper_id = numprocs - 1- iter/2;		
		int receiver_id = myid == 0 ? numprocs-1:myid-1 ;	
		
		block1_id = blocks1_id[iter];
		block2_id = blocks2_id[iter];
					
		if(iter < steps - 1)
		{			
			if(myid == swapper_id)
			{								
				MPI_Isend(block1, M * blocks_size[block1_id], MPI_DOUBLE, receiver_id,TAG1,MPI_COMM_WORLD,&req);	
				MPI_Isend(block1_mv, 2*(v_b+1) , MPI_DOUBLE,receiver_id,TAG3,MPI_COMM_WORLD, &req1);	
			}
			
			else
			{
				MPI_Isend(block2, M * blocks_size[block2_id], MPI_DOUBLE, receiver_id,TAG1,MPI_COMM_WORLD,&req);	
				MPI_Isend(block2_mv, 2*(v_b+1) , MPI_DOUBLE,receiver_id,TAG3,MPI_COMM_WORLD, &req1);				
			}
		} 
		
		pair_block_calculation(block1,block2,blocks_size[block1_id],blocks_size[block2_id],
		result_each_process,result_count,block1_mean,block1_variance,block2_mean,block2_variance);
		
		result_count += blocks_size[block1_id] * blocks_size[block2_id];		
		
/**		if(iter < steps -1)
		{
			int b1_id = blocks1_id[iter+1], b2_id = blocks2_id[iter+1];
			
			read_files(block1,blocks_vector_id[blocks1_id[iter+1]] + 1,blocks_size[blocks1_id[iter+1]], M);
			read_files(block2,blocks_vector_id[blocks2_id[iter+1]] + 1,blocks_size[blocks2_id[iter+1]], M);			
			
			for(i = 0; i <blocks_size[b1_id]; i++)
				vector_mean_variance(block1+i*M,M,block1_mean+i,block1_variance+i);
		
			for(i = 0; i <blocks_size[b2_id]; i++)
				vector_mean_variance(block2+i*M,M,block2_mean+i,block2_variance+i);				
		}
**/
	
		if(iter < steps -1)
		{
				
			MPI_Recv(block_buffer, M * blocks_size[blocks2_id[iter+1]],MPI_DOUBLE, (myid+1)%numprocs,TAG1,MPI_COMM_WORLD,&stat);
			MPI_Recv(block_mv_buffer, 2*(v_b+1) , MPI_DOUBLE,(myid+1)%numprocs,TAG3,MPI_COMM_WORLD, &stat1);				
				
			MPI_Wait(&req,&stat);
			MPI_Wait(&req1,&stat1);
			
			//swap block1 and block2
			if(myid == swapper_id)
			{
				printf("swapper id is %d, iter %d\n", swapper_id,iter);
				data_buffer = block1;
				block1 = block2;
				block2 = block_buffer;
				block_buffer = data_buffer;	
				
				data_buffer = block1_mv;
				block1_mv = block2_mv;
				block2_mv = block_mv_buffer;
				block_mv_buffer = data_buffer;	
	
			}
			
			else
			{
				data_buffer = block2;
				block2 = block_buffer;
				block_buffer = data_buffer;
				
				data_buffer = block2_mv;
				block2_mv = block_mv_buffer;
				block_mv_buffer = data_buffer;
			}				
			block1_mean = (block1_mv+0);
			block1_variance = (block1_mv+ (v_b+1));	
			block2_mean = (block2_mv+0);
			block2_variance = (block2_mv+ (v_b+1));					
		} 	
	}

	gettimeofday(&t1, NULL);
	timersub(&t1, &t0, &dt);
	
	MPI_Reduce(&computation_time, &total_computation_time, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
		
	if(myid > 0)
	{
	    printf("process %d: send results to main process\n",myid);
		MPI_Send(result_each_process,max_results_size,MPI_DOUBLE,0,TAG2,MPI_COMM_WORLD);
		
		printf("result of  process %d is:",myid);
		for( j = 0; j < max_results_size; j++)
			printf("%lf ",result_each_process[j]);
		puts("");
	}		
		
	if(myid == 0)
	{
		MPI_Request *reqs = malloc(sizeof(MPI_Request)*numprocs);
		MPI_Status *stats = malloc(sizeof(MPI_Status)*numprocs);		
		double **results_from_each_process = malloc(sizeof(double *)*numprocs);
		double *results = malloc(sizeof(double) * max_results_size * (numprocs-1));
		
		results_from_each_process[0] = result_each_process;		
		for (i = 1; i < numprocs; i++)
		{
			results_from_each_process[i] = results + (i-1)*max_results_size;
			printf("get results from process %d\n",i);
			MPI_Irecv(results_from_each_process[i],max_results_size,MPI_DOUBLE,i,TAG2,MPI_COMM_WORLD,reqs+i-1);
		}	
		MPI_Waitall(numprocs-1,reqs, stats);			
		
		double *correlations = malloc(sizeof(double) * N*(N-1)/2);
		double **correlation_matrix = malloc(sizeof(double *) * (N-1));		
		count = 0;
		for(i=0; i < N-1; i++)
		{
			correlation_matrix[i] = correlations + count;
			count += (N-i-1);
		}
		
		//sort result
		int x,y;
		int block1_size,block2_id,block2_size;
		
		for( i = 0; i < numprocs; i++)
		{
			count = 0;			
			for( j = 0; j < steps; j++)
			{
				block1_id = all_block1_id[j][i];
				block2_id = all_block2_id[j][i];
		//		printf("block1 id: %d   block2 id: %d\n",block1_id,block2_id);
				
				block1_size = blocks_size[block1_id];
				block2_size = blocks_size[block2_id];

				// the first step include calculation of single block
				if(j == 0)
				{
					for(k = 0; k <block1_size-1; k++)
					{
						for(l = 0; l < block1_size-1-k; l++)
							correlation_matrix[blocks_vector_id[block1_id]+k][l] = results_from_each_process[i][count++];
					}
					
					for(k = 0; k <block2_size-1; k++)
					{
						for(l = 0; l < block2_size-1-k; l++)
							correlation_matrix[blocks_vector_id[block2_id]+k][l] = results_from_each_process[i][count++];
					}					
					
				}
				
				for( k = 0; k < block1_size; k++)
				{		
					for ( l = 0; l < block2_size; l++)
					{
						x = min(blocks_vector_id[block1_id] + k, blocks_vector_id[block2_id] + l);
						y = max(blocks_vector_id[block1_id] + k, blocks_vector_id[block2_id] + l);
						y -= (x+1);
						correlation_matrix[x][y] = results_from_each_process[i][count++];	
//						printf(" correaltion matrix[%d][%d] = %lf\n",x,y,correlation_matrix[x][y]);
					}
				}					
				
				
			}
	
		}
			
	    printf("write result, N is:%d\n",N);
		write_results(correlation_matrix,N,dt);
		
		free(correlation_matrix);
		free(correlations);		
		free(results);
		free(results_from_each_process);
		free(reqs);
		free(stats);
	}

	length = 20;
	char name[50];
	MPI_Get_processor_name(name, &length);
	//record which node this process belongs to
	char *fp = malloc(sizeof(char) * 200);
	sprintf(fp,"%s/core_record_%dvectors_%dnodes_%dcores_interval%d_%s.txt",directory_name,N,n_node,numprocs,interval,mark);	
	FILE *f = fopen(fp, "a");
	fprintf(f,"record from process %s,id %d of %d\n",name,myid,numprocs);
	fclose(f);
	free(fp);

	printf("process %d finished,waiting others\n",myid);
	MPI_Barrier(MPI_COMM_WORLD);	
	
	for(i = 0; i < steps; i++)
	{
		free(all_block1_id[i]);
		free(all_block2_id[i]);
	}

	free(all_block1_id);
	free(all_block2_id);
	free(blocks1_id);
	free(blocks2_id);

	free(block1);
	free(block2);
	free(block1_mv);
	free(block2_mv);
	free(block_buffer);
	free(block_mv_buffer);

	free(blocks_size);
	free(blocks_vector_id);	
	
	MPI_Finalize();
	return 0;
	
}
