/**
 * combine thread and MPI**/
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#define _BSD_SOURCEs
#define max(a,b) ( (a >=b) ? a:b)
#define  min(a,b) ((a <= b) ? a:b )
//message used to exchange block use TAG1,send/recv inner product use TAG2,exchange mean/variance use block3
#define TAG1  0
#define TAG2  1
#define TAG3  2

struct thrd_data
{
	int t_id;
	int myid;
	double *block1;
	double *block2;
	double *block1_mean;
	double *block1_var;
	double *block2_mv;
	double *result;
	int *blocks_size;
	int *blocks2_id;
};

typedef struct {
	pthread_mutex_t count_lock;
	pthread_cond_t if_continnue;
	int count;
}  my_barrier;

void my_barrier_init(my_barrier *b)
{
	b-> count =0;
	pthread_mutex_init(&(b->count_lock),NULL);
	pthread_cond_init(&(b->if_continnue),NULL);
	return;
}

void my_barrier_destroy(my_barrier *b)
{
	pthread_mutex_destroy(&(b->count_lock));
	pthread_cond_destroy(&(b->if_continnue));
}

void wait_my_barrier(my_barrier *b, int num_threads)
{
	pthread_mutex_lock(&(b->count_lock));
	(b->count)++;
//	printf("count is %d\n",b->count);
	if(b->count == num_threads)
	{
		puts("continue");
		b->count = 0;
		pthread_mutex_unlock(&(b->count_lock));
		pthread_cond_broadcast(&(b->if_continnue));
		return;
	}
	pthread_cond_wait(&(b->if_continnue),&(b->count_lock));
	pthread_mutex_unlock(&(b->count_lock));	
	return;
}

char *output_hpc = "correlation_matrix.txt"; 
char *input_hpc = "data.txt";
char *directory_name = "/project/parallel/tao_tang/all_results/results_v12";
/** length of each vector**/
#define VECTOR_LENGTH 2725784
/** length of each interval in pairwise algorithm**/
int interval = 10000;
int n_node = 1;
int core_per_node = 1,n_threads;
int numprocs,N,v_b,steps;
/**size of each vector**/
int	M = 49500;
//int	M = VECTOR_LENGTH;
pthread_barrier_t barrier;
pthread_barrier_t barrier1;
//my_barrier barrier;
//measure time
struct timeval t0, t1, dt;

void vector_mean_variance(double *vector,int len, double *mean,double *variance)
{
	int i;
	double var = 0,sum = 0;
	for(i=0; i < len; i++)
	{
		var += vector[i] * vector[i];
		sum += vector[i];
	}
	*mean = sum/len;
	var -= (*mean) * (*mean) *len;
	*variance = var;
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
	char *fp = malloc(sizeof(char) * 150);
	sprintf(fp,"%s/result_%dvectors_%dnodes_%dcores.txt",directory_name,N,n_node,numprocs);
	FILE *f = fopen(fp, "w");
	fprintf(f,"runningtime: %d.%06d seconds\n",(int)dt.tv_sec,(int)dt.tv_usec);
	fprintf(f,"complexity: %d\n ", N*(N-1)/2);    
	fprintf(f,"nodenumber: %d\n ",n_node);
	fprintf(f,"core_pernode: %d\n ",core_per_node);
	fprintf(f,"process_number: %d\n ",numprocs);	 
	for(i=0; i<N-1; i++) 
	{
		for(j= 0;j < N-i-1; j++)
			fprintf(f,"%lf ",results[i][j]);
		fprintf(f,"\n");
	}
	
	fclose(f);
	free(fp);

	
	return; 	
}

double correlation(double *x, double *y, int len, double x_mean,double x_var, double y_mean, double y_var)
{
	double cov = 0;
	int i;
	for( i = 0; i < len ; i++)
		cov += x[i]*y[i];
		
	cov -= x_mean * y_mean * len;	
	return cov/(sqrt(x_var)*sqrt(y_var));
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

/**used to get start and end id of a worker when need to assign n jobs eaqually to some workers(thread or processors)**/
void get_start_end_id(int *s_id,int *e_id, int work_number, int worker_number, int worker_id)
{
	int avg = work_number/worker_number;
	int remainder = work_number%worker_number;
	(*s_id) = worker_id * avg + min(worker_id,remainder) * 1;
	(*e_id) =  worker_id < remainder ? (*s_id) + avg +1 : (*s_id) + avg;
	return;
}

void *thread_work_seq(void *thrd_arg)
{
	int i,j,k;


	struct thrd_data *t_data;	
	/* Initialize my part of the global array and keep local sum */
	t_data = (struct thrd_data*)thrd_arg;
		
	int t_id = t_data -> t_id;
	double *matrix = t_data -> block1;
	double *mean = t_data -> block1_mean;
	double *var = t_data -> block1_var;
	double *result = t_data -> result;
	
	int pairs_number = N*( N-1)/2;
	int start_id=0, end_id=0;
	
	get_start_end_id(&start_id,&end_id,pairs_number,n_threads,t_id);
	printf("start id: %d  end id: %d\n",start_id,end_id);
	
	int count = 0;
	//correaltion of block1
	for(i =0; i < N-1; i++)
	{
		for(j = i+1; j < N; j++)
		{
			if(count >= start_id && count < end_id)
				result[count] = correlation(matrix+i*M,matrix+j*M,M,mean[i],var[i],mean[j],var[j]);
			count++;
		}
	}
	
}

void * thread_work_even(void *thrd_arg)
{
	int i,j,k,iter;
	int thread_level_provided;
	//main thread,only this thread will make MPI call


	struct thrd_data *t_data;	
	/* Initialize my part of the global array and keep local sum */
	t_data = (struct thrd_data*)thrd_arg;
		
	int t_id = t_data -> t_id;
	int myid = t_data -> myid;
	double *block1 = t_data -> block1;
	double *block2 =  t_data -> block2;
	double *block1_mean = t_data -> block1_mean;
	double *block2_mv = t_data -> block2_mv + 0;
	double *block2_mean = t_data -> block2_mv + 0;
	double *block1_var = t_data -> block1_var;
	double *block2_var = t_data -> block2_mv + v_b+1 ; 
	double *result = t_data -> result;
	int *blocks_size = t_data -> blocks_size;
	int *blocks2_id = t_data ->blocks2_id;
	
	double *block2_buffer,*block2_mv_buffer; 	
	/**in hpc, non-blocking communication needs to waited until it is received by receiver, so need process 0 to receive it first **/
	if(myid == 0 && t_id ==0)
	{
		block2_buffer = malloc(sizeof(double)*(v_b+1) * M);
		block2_mv_buffer  = malloc(2* sizeof(double)*(v_b+1));		
	}
	
	int pairs_number  = 0, start_id = 0, end_id = 0,result_count = 0, work_count =0;
	
	MPI_Status stat,stat1;
	MPI_Request req,req1;
	for( iter = 0; iter < steps-1; iter++)
	{
		int receiver_id =  (myid == 0 ? numprocs-1:myid-1);
		if(t_id == 0)
		{		
			printf("process %d send block2 to process %d\n",myid,receiver_id);	
			MPI_Isend(block2_mv, 2*(v_b+1) , MPI_DOUBLE,receiver_id,TAG3,MPI_COMM_WORLD, &req1);			
						
			MPI_Isend(block2, M * blocks_size[blocks2_id[iter]], MPI_DOUBLE, receiver_id,TAG1,MPI_COMM_WORLD,&req);
		
		}

		int b2_id = blocks2_id[iter];
		//number of works in this iteration
		pairs_number = blocks_size[myid] * blocks_size[b2_id]; 
		get_start_end_id(&start_id,&end_id,pairs_number,n_threads,t_id);
		work_count = 0;
		for( i = 0; i < blocks_size[myid]; i++)
		{

			for(j = 0; j < blocks_size[b2_id]; j++)
			{
				if(work_count>= start_id && work_count < end_id)
					result[result_count] = correlation(block1+i*M,block2+j*M,M,block1_mean[i],block1_var[i],block2_mean[j],block2_var[j]);
		
				work_count++;
				result_count++;
			}
		}

		pthread_barrier_wait(&barrier1);		
//		wait_my_barrier(&barrier,n_threads);	
		//first thread responsible for exchanging data in block2
		if(t_id == 0)
		{
			if(myid == 0)
			{
				MPI_Recv(block2_buffer, M * blocks_size[blocks2_id[iter+1]],MPI_DOUBLE, (myid+1)%numprocs,TAG1,MPI_COMM_WORLD,&stat);
				MPI_Recv(block2_mv_buffer, 2*(v_b+1) , MPI_DOUBLE,1,TAG3,MPI_COMM_WORLD, &stat1);				
				MPI_Wait(&req,&stat);
				MPI_Wait(&req1,&stat1);
				double *buffer = block2_buffer;
				block2_buffer = block2;
				block2 = buffer;
				memcpy(block2_mv,block2_mv_buffer, sizeof(double) * 2 *(v_b+1));				
			}  
			else 
			{
				MPI_Wait(&req1,&stat1);
				MPI_Wait(&req,&stat); 
				MPI_Recv(block2_mv, 2*(v_b+1) , MPI_DOUBLE,(myid+1)%numprocs,TAG3,MPI_COMM_WORLD, &stat1);					
				MPI_Recv(block2, M * blocks_size[blocks2_id[iter+1]],MPI_DOUBLE, (myid+1)%numprocs,TAG1,MPI_COMM_WORLD,&stat);
			}			
		} 	
		pthread_barrier_wait(&barrier1);	
	}
		
	pthread_barrier_wait(&barrier1);	
	
	if(myid < numprocs/2)
	{
		int b2_id = blocks2_id[steps-1];
		//number of works in this iteration
		pairs_number = blocks_size[myid] * blocks_size[b2_id]; 
		get_start_end_id(&start_id,&end_id,pairs_number,n_threads,t_id);
		work_count = 0;
		for( i = 0; i < blocks_size[myid]; i++)
		{

			for(j = 0; j < blocks_size[b2_id]; j++)
			{
				if(work_count>= start_id && work_count < end_id)
					result[result_count] = correlation(block1+i*M,block2+j*M,M,block1_mean[i],block1_var[i],block2_mean[j],block2_var[j]);
		
				work_count++;
				result_count++;
			}
		}		
		
	}

	if(myid >= numprocs/2)
	{
		int b2_id = blocks2_id[steps-1];
		//number of works in this iteration
		pairs_number = (blocks_size[myid] * (blocks_size[myid]-1))/2;
		get_start_end_id(&start_id,&end_id,pairs_number,n_threads,t_id);
		work_count = 0;
		
		for(i =0; i < blocks_size[myid]-1; i++)
		{
			for(j = i+1; j < blocks_size[myid]; j++)
			{
				if(work_count >= start_id && work_count < end_id)
					result[result_count] = correlation(block1+i*M,block1+j*M,M,block1_mean[i],block1_var[i],block1_mean[j],block1_var[j]);
				
				work_count++;
				result_count++;
			}
		}
		
		pairs_number = (blocks_size[b2_id] * (blocks_size[b2_id]-1))/2;
		get_start_end_id(&start_id,&end_id,pairs_number,n_threads,t_id);
		work_count = 0;
		for(i =0; i < blocks_size[b2_id]-1; i++)
		{
			for(j = i+1; j < blocks_size[b2_id]; j++)
			{
				if(work_count >= start_id && work_count < end_id)
					result[result_count] = correlation(block2+i*M,block2+j*M,M,block2_mean[i],block2_var[i],block2_mean[j],block2_var[j]);
				
				work_count++;
				result_count++;
			}
		}				
	}			
	
	

}

void *thread_work_odd(void *thrd_arg)
{
	int i,j,k,iter;
	int thread_level_provided;
	//main thread,only this thread will make MPI call


	struct thrd_data *t_data;	
	/* Initialize my part of the global array and keep local sum */
	t_data = (struct thrd_data*)thrd_arg;
		
	int t_id = t_data -> t_id;
	int myid = t_data -> myid;
	double *block1 = t_data -> block1;
	double *block2 =  t_data -> block2;
	double *block1_mean = t_data -> block1_mean;
	double *block2_mv = t_data -> block2_mv + 0;
	double *block2_mean = t_data -> block2_mv + 0;
	double *block1_var = t_data -> block1_var;
	double *block2_var = t_data -> block2_mv + v_b+1 ; 
	double *result = t_data -> result;
	int *blocks_size = t_data -> blocks_size;
	int *blocks2_id = t_data ->blocks2_id;
	
	double *block2_buffer,*block2_mv_buffer; 	
	/**in hpc, non-blocking communication needs to waited until it is received by receiver, so need process 0 to receive it first **/
	if(myid == 0 && t_id ==0)
	{
		block2_buffer = malloc(sizeof(double)*(v_b+1) * M);
		block2_mv_buffer  = malloc(2* sizeof(double)*(v_b+1));		
	}
	int pairs_number = blocks_size[myid] * (blocks_size[myid]-1)/2;
	int start_id=0, end_id=0;
	
	get_start_end_id(&start_id,&end_id,pairs_number,n_threads,t_id);
	
	
	int work_count = 0,result_count = 0;
	//correaltion of block1
	for(i =0; i < blocks_size[myid]-1; i++)
	{
		for(j = i+1; j < blocks_size[myid]; j++)
		{
			if(work_count >= start_id && work_count < end_id)
				result[result_count] = correlation(block1+i*M,block1+j*M,M,block1_mean[i],block1_var[i],block1_mean[j],block1_var[j]);
			work_count++;
			result_count++;
		}
	}
//	printf(" a line 257 process %d,blocks2_id %d\n",myid,blocks2_id[0]);	
	//iterations from second to last step
	MPI_Status stat,stat1;
	MPI_Request req,req1;
	for( iter = 0; iter < steps; iter++)
	{
		int receiver_id =  (myid == 0 ? numprocs-1:myid-1);
		if(t_id == 0 && iter < steps -1)
		{		
			printf("process %d send block2 to process %d\n",myid,receiver_id);	
			MPI_Isend(block2_mv, 2*(v_b+1) , MPI_DOUBLE,receiver_id,TAG3,MPI_COMM_WORLD, &req1);			
						
			MPI_Isend(block2, M * blocks_size[blocks2_id[iter]], MPI_DOUBLE, receiver_id,TAG1,MPI_COMM_WORLD,&req);
			//printf("line 269 process %d\n",myid);		
		}

		int b2_id = blocks2_id[iter];
		//number of works in this iteration
		pairs_number = blocks_size[myid] * blocks_size[b2_id]; 
		get_start_end_id(&start_id,&end_id,pairs_number,n_threads,t_id);
		work_count = 0;
		for( i = 0; i < blocks_size[myid]; i++)
		{

			for(j = 0; j < blocks_size[b2_id]; j++)
			{
				if(work_count>= start_id && work_count < end_id)
				{
					result[result_count] = correlation(block1+i*M,block2+j*M,M,block1_mean[i],block1_var[i],block2_mean[j],block2_var[j]);
//					printf("process %d,work %d",myid,work_count);		
				}
				work_count++;
				result_count++;
			}
		}

		pthread_barrier_wait(&barrier1);		
//		wait_my_barrier(&barrier,n_threads);	
		//first thread responsible for exchanging data in block2
		if(t_id == 0 && iter < steps -1)
		{
			if(myid == 0)
			{
				MPI_Recv(block2_buffer, M * blocks_size[blocks2_id[iter+1]],MPI_DOUBLE, (myid+1)%numprocs,TAG1,MPI_COMM_WORLD,&stat);
				MPI_Recv(block2_mv_buffer, 2*(v_b+1) , MPI_DOUBLE,1,TAG3,MPI_COMM_WORLD, &stat1);				
				MPI_Wait(&req,&stat);
				MPI_Wait(&req1,&stat1);
				double *buffer = block2_buffer;
				block2_buffer = block2;
				block2 = buffer;
				memcpy(block2_mv,block2_mv_buffer, sizeof(double) * 2 *(v_b+1));				
			}  
			else 
			{
				MPI_Wait(&req1,&stat1);
				MPI_Wait(&req,&stat); 
				MPI_Recv(block2_mv, 2*(v_b+1) , MPI_DOUBLE,(myid+1)%numprocs,TAG3,MPI_COMM_WORLD, &stat1);					
				MPI_Recv(block2, M * blocks_size[blocks2_id[iter+1]],MPI_DOUBLE, (myid+1)%numprocs,TAG1,MPI_COMM_WORLD,&stat);
			}			
		} 
		pthread_barrier_wait(&barrier1);		
	}
	
	if(myid == 0 && t_id ==0)
	{
		free(block2_buffer) ;
		free(block2_mv_buffer);		
	}	
	
	
}


//sequential computing for self-checking
void sequential_process(double *matrix,int M, int N)
{
	int i,j,count=0;	
	int T_size = M/interval + 1;
	double *mean = malloc(sizeof(double) * N);
	double *variance = malloc(sizeof(double) * N);
	double *result = malloc(sizeof(double)*N *(N-1)/2);
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
}

//note, this algorithm only for condition that numporcs is odd 
int main(int argc, char **argv)
{
	int myid,length;
	//determine rank of each preocessor
	int thread_level_provided;	
//	MPI_Init(&argc,&argv);
    MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&thread_level_provided);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	
	int i,j,k,l,count = 0;

	n_node = atoi(argv[1]);
	core_per_node = atoi(argv[2]);
	N = atoi(argv[3]);	
	if(argc > 4)
		M = atoi(argv[4]);
	
	
	n_threads = core_per_node;
	mkdir(directory_name, 0777);

	
	if(numprocs == 1)
	{
		double *matrix = malloc(sizeof(double)* N * M);
		read_files(matrix,1,N,M);
		sequential_process(matrix,M,N);
		free(matrix);
		return;
	}	
	

	pthread_attr_t attr;
//	my_barrier_init (&barrier);
	pthread_barrier_init(&barrier,NULL,n_threads+1);
	pthread_barrier_init(&barrier1,NULL,n_threads);
	pthread_t *thread_id = (pthread_t *)malloc(sizeof(pthread_t)*n_threads);
	struct thrd_data *t_arg = (struct thrd_data *)malloc(sizeof(struct thrd_data)*n_threads);	



	if(numprocs%2 ==1)
		steps = (numprocs-1)/2;		
	else 
		steps = numprocs/2;
		
	MPI_Status stat,stat1;
	MPI_Request req,req1;
	v_b = N / numprocs;
	int remainder = N%numprocs;
	//each process contains two blocks,each block contain v_b or v_b+1 vectors
	double *block1 = malloc(sizeof(double) * (v_b+1) * M);
	double *block2 = malloc(sizeof(double) * (v_b+1) * M);	
	//mean and variance of block1,2
	double *block1_mv = malloc(sizeof(double)* 2*(v_b+1)); 
	double *block1_mean = (block1_mv+0);
	double *block1_variance = (block1_mv+ (v_b+1));	
	double *block2_mv = malloc(sizeof(double)* 2*(v_b+1)); 
	double *block2_mean = (block2_mv+0);
	double *block2_variance = (block2_mv+ (v_b+1));

	//the ids of second block each process keeps in each iteration, (frist block is fixed during computation)
	int *blocks2_id = malloc(sizeof(int) * steps);
	for(i=0;i < steps; i++ )
		blocks2_id[i] = (myid+i+1)%numprocs;
	
	//number of vectors each block contain
	int *blocks_size = malloc(sizeof(int)*numprocs);
	int *blocks_vector_id = malloc(sizeof(int)*numprocs);

	// id of the first vectors each block contain 
	blocks_vector_id[0] = 0;
	blocks_size[0] = (0 < remainder) ? v_b+1 : v_b;

	printf("id of the first vector in each process:");		
	for(i = 1; i < numprocs; i++)
	{
		blocks_size[i] = (i < remainder) ? v_b+1 : v_b;
		blocks_vector_id[i] = blocks_vector_id[i-1] + blocks_size[i-1];
		printf(" %d",blocks_vector_id[i]);
	}
	puts("");
	printf("blocks size:");
	for( i = 0; i < numprocs; i++)
		printf(" %d ", blocks_size[i]);	
	puts("");
	int v_b_buffer = v_b+1;
	//maximum number of size of results computed by each process
	int max_results_size = v_b_buffer*(v_b_buffer-1)/2 + v_b_buffer * v_b_buffer * steps;
	//store results computed by this process
	double *result_each_process = malloc( sizeof(double) * max_results_size );
	
	//each process read the a part of the whole matrix
 	read_files(block1,blocks_vector_id[myid] + 1,blocks_size[myid], M);
	
	int receiver_id = (myid == 0 ? numprocs -1 : myid -1);
	printf("send block1, process %d\n",myid);
	MPI_Isend(block1,M * blocks_size[myid],MPI_DOUBLE,receiver_id,TAG1,MPI_COMM_WORLD,&req);
	printf("receive block2, process %d\n",myid);
	MPI_Recv(block2,M * blocks_size[(myid+1)%numprocs],MPI_DOUBLE,(myid+1)%numprocs,TAG1,MPI_COMM_WORLD,&stat);		

	for(i = 0; i <blocks_size[myid]; i++)
		vector_mean_variance(block1+i*M,M,block1_mean+i,block1_variance+i);
				
	MPI_Wait(&req,&stat);

	MPI_Isend(block1_mv, 2*(v_b+1), MPI_DOUBLE, receiver_id,TAG3,MPI_COMM_WORLD, &req);
	MPI_Recv(block2_mv, 2*(v_b+1),MPI_DOUBLE,(myid+1)%numprocs ,TAG3,MPI_COMM_WORLD, &stat);	
	MPI_Wait(&req,&stat);	
	


	for(i = 0; i < n_threads; i++)
	{
		//define thread argument
		t_arg[i].t_id = i;
		t_arg[i].myid = myid;
		t_arg[i].block1 = block1;
		t_arg[i].block2 = block2;
		t_arg[i].block1_mean = block1_mean;
		t_arg[i].block1_var = block1_variance;
		t_arg[i].block2_mv = block2_mv;
		t_arg[i].result = result_each_process;
		t_arg[i].blocks_size = blocks_size;
		t_arg[i].blocks2_id = blocks2_id;
	}
	
	
	gettimeofday(&t0, NULL);
	/** thread work is different for numporcs is odd/even**/
	double *block2_buffer, *block2_mv_buffer;
	if (myid == 0)
	{
	  block2_buffer = malloc(sizeof(double) * (v_b+1) * M);	
	  block2_mv_buffer = malloc(sizeof(double) * (v_b+1) * 2);
	}
	
	int iter;
	printf("numprocs:%d n_threads:%d\n",numprocs,n_threads);
	
	if(numprocs%2 ==1)
	{
	
		for(i = 0; i < n_threads; i++)
		{
			printf("process %d create thread %d\n",myid,i);	
			int rc = pthread_create(thread_id+i,NULL, &thread_work_odd,t_arg+i);
			if (rc < 0)
				printf("create of thread %d in process %d failed with rc=%d\n",i,myid,rc);
		}
	}
	
	if(numprocs%2 ==0)
	{	
		for(i = 0; i < n_threads; i++)
			pthread_create(thread_id+i,NULL, &thread_work_even,t_arg+i);		
	}	
	
	for( i = 0; i < n_threads; i++)
	{
		pthread_join(thread_id[i],NULL);
		printf("process %d: thread  %d finished\n",myid,i);	
	}
	printf("finish thread work process %d\n",myid);
	
	
	gettimeofday(&t1, NULL);
	timersub(&t1, &t0, &dt);	
		
	if(myid == 0)
	{
		free(block2_buffer);
		free(block2_mv_buffer);
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
		puts("partial result is:");
		for( j = 0; j < max_results_size; j++)
			printf("%lf ",results_from_each_process[2][j]);
		puts("");			
		
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
		if(numprocs%2 ==1)
		{
			for( i = 0; i < numprocs; i++)
			{
				block1_size = blocks_size[i];
				count = 0;
				int result_size_each_iteration = block1_size * (block1_size - 1)/2;
				printf("copy results of block1\n");
				for( j = 0; j < block1_size; j++)
				{
					for(k = 0; k < block1_size-1-j; k++)
						correlation_matrix[blocks_vector_id[i]+j][k] = results_from_each_process[i][count++];
				}

				for( j = 0; j < steps; j++)
				{
					block2_id = (i+j+1)%numprocs;
					block2_size = blocks_size[block2_id];
					printf("size of block %d is %d\n",block2_id,block2_size);
					for( k = 0; k < block1_size; k++)
					{
					
						for ( l = 0; l < block2_size; l++)
						{
							x = min(blocks_vector_id[i] + k, blocks_vector_id[block2_id] + l);
							y = max(blocks_vector_id[i] + k, blocks_vector_id[block2_id] + l);
							y -= (x+1);
							correlation_matrix[x][y] = results_from_each_process[i][count++];	
							printf(" correaltion matrix[%d][%d] = %lf\n",x,y,correlation_matrix[x][y]);
						}
					}	
				}
			}		
		}
		
		if(numprocs%2 == 0)
		{
			for( i = 0; i < numprocs/2; i++)
			{
				printf(" process %d line 279\n ",i);
				block1_size = blocks_size[i];
				count = 0;

				for( j = 0; j < steps; j++)
				{
					block2_id = (i+j+1)%numprocs;
					block2_size = blocks_size[block2_id];
					printf("size of block %d is %d\n",block2_id,block2_size);
					for( k = 0; k < block1_size; k++)
					{
						for ( l = 0; l < block2_size; l++)
						{	
							x = min(blocks_vector_id[i] + k, blocks_vector_id[block2_id] + l);
							y = max(blocks_vector_id[i] + k, blocks_vector_id[block2_id] + l);
							y -= (x+1);
							correlation_matrix[x][y]  = results_from_each_process[i][count++];						
							
						}
					}	
				}
			}
		
			for( i = numprocs/2; i < numprocs; i++)
			{			
				printf(" process %d line 279\n ",i);
				block1_size = blocks_size[i];
				count = 0;
				for( j = 0; j < steps - 1; j++)
				{
					puts("get information of block2");
					block2_id = (i+j+1)%numprocs;
					block2_size = blocks_size[block2_id];
					printf("size of block %d is %d\n",block2_id,block2_size);
					for( k = 0; k < block1_size; k++)
					{	
						for ( l = 0; l < block2_size; l++)
						{	
							x = min(blocks_vector_id[i] + k, blocks_vector_id[block2_id] + l);
							y = max(blocks_vector_id[i] + k, blocks_vector_id[block2_id] + l);
							y -= (x+1);
							double a = results_from_each_process[i][count++];						
							correlation_matrix[x][y] = a;
					
						}
					}	
				}
				printf("get information of the second block, i:%d, steps:%d numprocs:%d \n",i,steps,numprocs);
				block2_id = (i+steps)%numprocs;
				//block2_size = blocks_size[block2_id];
				block2_size = blocks_size[block2_id];
				printf("got id:%d and size %d of block 2",block2_id,block2_size);
				for(k = 0; k < block1_size; k++)
				{
					for( l = 0; l < block1_size -1 -k; l++)
						correlation_matrix[blocks_vector_id[i]+k][l] = results_from_each_process[i][count++];
				}
				for(k = 0; k < block2_size; k++)
				{
					for( l = 0; l < block2_size-1-k; l++)
						correlation_matrix[blocks_vector_id[block2_id]+k ][l] = results_from_each_process[i][count++];
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

	if(myid > 0)
	{
	    printf("process %d: send results to main process\n",myid);
		MPI_Send(result_each_process,max_results_size,MPI_DOUBLE,0,TAG2,MPI_COMM_WORLD);
		
		printf("result of  process %d is:",myid);
		for( j = 0; j < max_results_size; j++)
			printf("%lf ",result_each_process[j]);
		puts("");
	}
	printf("process %d finished,waiting others\n",myid);
	MPI_Barrier(MPI_COMM_WORLD);	
	
	free(thread_id);
	free(t_arg);
	free(block1);
	free(block2);
	free(block1_mv);
	free(block2_mv);
	free(blocks2_id);
	free(blocks_size);
	free(blocks_vector_id);	
	pthread_barrier_destroy(&barrier);
	
//	my_barrier_destroy(&barrier);
	pthread_exit (NULL);



	MPI_Finalize();
	return 0;
	
}




