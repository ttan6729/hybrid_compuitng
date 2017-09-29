/**
 * pure MPI version, non-blocking communication, remove wait to observe effect of coummnucation time**/
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
char *directory_name = "/project/parallel/tao_tang/all_results/results_v15_29_Aug";
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
struct timeval t0, t1, dt;
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

struct work_data
{
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

void *distributed_work_even(struct work_data *w_data)
{
	int i,j,k,iter,tid;
	int thread_level_provided;
	//main thread,only this thread will make MPI call
	/* Initialize my part of the global array and keep local sum */		
	int myid = w_data -> myid;
	double *block1 = w_data -> block1;
	double *block2 =  w_data -> block2;
	double *block1_mean = w_data -> block1_mean;
	double *block2_mv = w_data -> block2_mv + 0;
	double *block2_mean = w_data -> block2_mv + 0;
	double *block1_var = w_data -> block1_var;
	double *block2_var = w_data -> block2_mv + v_b+1 ; 
	double *result = w_data -> result;
	int *blocks_size = w_data -> blocks_size;
	int *blocks2_id = w_data ->blocks2_id;
	
	double *block2_buffer,*block2_mv_buffer; 	
	int interval_number = M/interval;	
	/**in hpc, non-blocking communication needs to waited until it is received by receiver, so need process 0 to receive it first **/
	if(myid == 0)
	{
		block2_buffer = malloc(sizeof(double)*(v_b+1) * M);
		block2_mv_buffer  = malloc(2* sizeof(double)*(v_b+1));		
	}
	
	int pairs_number  = 0, start_id = 0, end_id = 0;
	
	MPI_Status stat,stat1;
	MPI_Request req,req1;
	int result_count = 0,work_count =0,result_count_buffer =0; 
	int b1_size = blocks_size[myid];
	int k_m;	
	printf("line 247 process %d\n",myid);	
	for( iter = 0; iter < steps-1; iter++)
	{
		int receiver_id =  (myid == 0 ? numprocs-1:myid-1);

		printf("process %d send block2 to process %d\n",myid,receiver_id);	
		MPI_Isend(block2_mv, 2*(v_b+1) , MPI_DOUBLE,receiver_id,TAG3,MPI_COMM_WORLD, &req1);			
		MPI_Isend(block2, M * blocks_size[blocks2_id[iter]], MPI_DOUBLE, receiver_id,TAG1,MPI_COMM_WORLD,&req);
		
		result_count_buffer = result_count;
		int b2_id = blocks2_id[iter];
		for(k=0; k< interval_number; k++)	
		{
			k_m = k * interval;	
			for(i = 0; i < b1_size; i++)
			{
				for(j = 0; j < blocks_size[b2_id]; j++)		
					result[result_count++] += update_covariance(block1+i*M+k_m, block2+j*M+k_m,interval, block1_mean[i],block2_mean[j]);
			}
			result_count = result_count_buffer;
		}
	
		printf("line 265 process %d\n",myid);		
		k_m = interval_number * interval;
		for(i = 0; i < b1_size; i++)
		{		
			for(j = 0; j< blocks_size[b2_id]; j++)
			{		
				result[result_count] += update_covariance(block1+i*M+k_m, block2+j*M+k_m,0, block1_mean[i],block2_mean[j]);
				double cov = result[result_count]/(M-1);
				result[result_count] = result[result_count]/(sqrt(block1_var[0]) * sqrt(block2_var[0]));
				result_count++;
			}
			
		}	
		printf("line 278 process %d\n",myid);			
		result_count_buffer = result_count;

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

	if(myid < numprocs/2)
	{
		int b2_id = blocks2_id[steps-1];
		result_count_buffer = result_count;	
		
		for(k=0; k< interval_number; k++)
		{
			k_m = k * interval;				
			for(i =0; i < blocks_size[myid]; i++)
			{
				for(j = 0; j < blocks_size[b2_id]; j++)
					result[result_count++] += update_covariance(block1+i*M+ k_m, block2+j*M+k_m, interval,block1_mean[i],block2_mean[j]);
			}	
			result_count = result_count_buffer;
		}

		k_m = interval_number*interval;
		for(i =0; i < blocks_size[myid]; i++)
		{
			
			for(j = 0; j < blocks_size[b2_id]; j++)
			{
				result[result_count] += update_covariance(block1+i*M+k_m, block2+j*M+k_m,M%interval, block1_mean[i],block2_mean[j]);
				result[result_count]= result[result_count]/(M-1);
				/** corr(x,y) = cov(s,y)/ sqrt(var(x) * var(y)**/
				result[result_count]= result[result_count]/(sqrt(block1_var[i]) * sqrt(block2_var[j]));
			}
			result_count++;
		}
	}

	if(myid >= numprocs/2)
	{
		int b2_id = blocks2_id[steps-1];
		//number of works in this iteration
		pairs_number = (blocks_size[myid] * (blocks_size[myid]-1))/2;		
		result_count_buffer = result_count;	
		for(k=0; k< interval_number; k++)
		{
			k_m = k*interval;
			for(i =0; i < blocks_size[myid]-1; i++)
			{
				for(j = i+1; j < blocks_size[myid]; j++)
					result[result_count++] += update_covariance(block1+i*M+ k_m, block1+j*M+k_m, interval,block1_mean[i],block1_mean[j]);
			}	
			result_count = result_count_buffer;
		}

		k_m = interval*interval_number;
		for(i =0; i < blocks_size[myid]-1; i++)
		{
			for(j = i+1; j < blocks_size[myid]; j++)
			{
				result[result_count] += update_covariance(block1+i*M+ k_m, block1+j*M+k_m, M%interval,block1_mean[i],block1_mean[j]);
				result[result_count]= result[result_count]/(M-1);
				/** corr(x,y) = cov(s,y)/ sqrt(var(x) * var(y)**/
				result[result_count]= result[result_count]/(sqrt(block1_var[i]) * sqrt(block1_var[j]));
				result_count++;
			}
		}
		
		result_count_buffer = result_count;	
				
		for(k=0; k< interval_number; k++)
		{
			k_m = k*interval;
			for(i =0; i < blocks_size[b2_id]-1; i++)
			{
				for(j = i+1; j < blocks_size[b2_id]; j++)
					result[result_count++] += update_covariance(block2+i*M+k_m, block2+j*M+k_m, interval,block2_mean[i],block2_mean[j]);
			}
			result_count = result_count_buffer;
		}
		k_m = interval*interval_number;				
		for(i =0; i < blocks_size[b2_id]-1; i++)
		{
			for(j = i+1; j < blocks_size[b2_id]; j++)
			{
				result[result_count] += update_covariance(block2+i*M+k_m, block2+j*M+k_m,M%interval,block2_mean[i],block2_mean[j]);
				result[result_count]= result[result_count]/(M-1);
				/** corr(x,y) = cov(s,y)/ sqrt(var(x) * var(y)**/
				result[result_count]= result[result_count]/(sqrt(block2_var[i]) * sqrt(block2_var[j]));

				result_count++;
			}
		}				
	}			
}

void *distributed_work_odd(struct work_data *w_data)
{
	int i,j,k,iter;
	//main thread,only this thread will make MPI call
	/* Initialize my part of the global array and keep local sum */
	int myid = w_data -> myid;
	double *block1 = w_data -> block1;
	double *block2 =  w_data -> block2;
	double *block1_mean = w_data -> block1_mean;
	double *block2_mv = w_data -> block2_mv + 0;
	double *block2_mean = w_data -> block2_mv + 0;
	double *block1_var = w_data -> block1_var;
	double *block2_var = w_data -> block2_mv + v_b+1 ; 
	double *result = w_data -> result;
	int *blocks_size = w_data -> blocks_size;
	int *blocks2_id = w_data ->blocks2_id;
	
	double *block2_buffer,*block2_mv_buffer; 	
	/**in hpc, non-blocking communication needs to waited until it is received by receiver, so need process 0 to receive it first **/
	if(myid == 0)
	{
		block2_buffer = malloc(sizeof(double)*(v_b+1) * M);
		block2_mv_buffer  = malloc(2* sizeof(double)*(v_b+1));		
	}
	int pairs_number = blocks_size[myid] * (blocks_size[myid]-1)/2;
	int work_count = 0,result_count = 0, result_count_buffer;
	//correaltion of block1
	int interval_number = M/interval;
	result_count_buffer = result_count;
	int k_m;
	for(k=0; k< interval_number; k++)
	{
		k_m = k*interval;
		for(i =0; i < blocks_size[myid]-1; i++)
		{
			for(j = i+1; j < blocks_size[myid]; j++)
				result[result_count++] += update_covariance(block1+i*M+ k_m, block1+j*M+k_m, interval,block1_mean[i],block1_mean[j]);
		}	
		result_count = result_count_buffer;
	}

	k_m = interval*interval_number;
	for(i =0; i < blocks_size[myid]-1; i++)
	{
		for(j = i+1; j < blocks_size[myid]; j++)
		{
			result[result_count] += update_covariance(block1+i*M+ k_m, block1+j*M+k_m, M%interval,block1_mean[i],block1_mean[j]);
			result[result_count]= result[result_count]/(M-1);
			/** corr(x,y) = cov(s,y)/ sqrt(var(x) * var(y)**/
			result[result_count]= result[result_count]/(sqrt(block1_var[i]) * sqrt(block1_var[j]));
			result_count++;
		}
	}
	result_count_buffer = result_count;
		
	int b1_size = blocks_size[myid];
//	printf(" a line 257 process %d,blocks2_id %d\n",myid,blocks2_id[0]);	
	//iterations from second to last step
	MPI_Status stat,stat1;
	MPI_Request req,req1;
	
	printf("line 454 process %d\n",myid);
	for( iter = 0; iter < steps; iter++)
	{
		int receiver_id =  (myid == 0 ? numprocs-1:myid-1);
		if(iter < steps -1)
		{		
			printf("process %d send block2 to process %d\n",myid,receiver_id);	
			MPI_Isend(block2_mv, 2*(v_b+1) , MPI_DOUBLE,receiver_id,TAG3,MPI_COMM_WORLD, &req1);								
			MPI_Isend(block2, M * blocks_size[blocks2_id[iter]], MPI_DOUBLE, receiver_id,TAG1,MPI_COMM_WORLD,&req);
			//printf("line 269 process %d\n",myid);		
		}
		
		//number of works in this iteration
		result_count_buffer = result_count;
		int b2_id = blocks2_id[iter];
		int k_m;
		for(k=0; k< interval_number; k++)	
		{
			k_m = k * interval;	
			for(i = 0; i < b1_size; i++)
			{
				for(j = 0; j < blocks_size[b2_id]; j++)		
					result[result_count++] += update_covariance(block1+i*M+k_m, block2+j*M+k_m,interval, block1_mean[i],block2_mean[j]);
			}
			result_count = result_count_buffer;
		}
	
		k_m = interval_number * interval;
		for(i = 0; i < b1_size; i++)
		{		
			for(j = 0; j<blocks_size[b2_id]; j++)
			{		
				result[result_count] += update_covariance(block1+i*M+k_m, block2+j*M+k_m,M%interval, block1_mean[i],block2_mean[j]);
				double cov = result[result_count]/(M-1);
				result[result_count] = cov/(sqrt(block1_var[i]) * sqrt(block2_var[j]));
				result_count++;
			}
			
		}
		result_count_buffer= result_count;
		if(iter < steps -1)
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
	}
	
	if(myid == 0)
	{
		free(block2_buffer) ;
		free(block2_mv_buffer);		
	}	
	
	
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
				/** corr(x,y) = cov(s,y)/ sqrt(var(x) * var(y)**/
				result[count]= result[count]/(sqrt(var[i]) * sqrt(var[j]));
			
			}
			count++;
		}
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
	if(argc > 5)
		interval = atoi(argv[5]);
	if(argc > 6)
		mark = argv[6];
	
	
	
	n_threads = core_per_node;
	mkdir(directory_name, 0777);

	
	if(numprocs == 1)
	{
		double *matrix = malloc(sizeof(double)* N * M);
		read_synthetic_data(matrix,1,N,M);
		sequential_process(matrix,M,N);
		free(matrix);
		return;
	}	

	pthread_attr_t attr;
//	my_barrier_init (&barrier);
	pthread_barrier_init(&barrier,NULL,n_threads+1);
	pthread_barrier_init(&barrier1,NULL,n_threads);
	pthread_t *thread_id = (pthread_t *)malloc(sizeof(pthread_t)*n_threads);
	struct work_data *w_arg = (struct work_data *)malloc(sizeof(struct work_data));	


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
	for(i = 0; i < max_results_size; i++)
		result_each_process[i] = 0;
	
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
	
	w_arg->myid = myid;
	w_arg->block1 = block1;
	w_arg->block2 = block2;
	w_arg->block1_mean = block1_mean;
	w_arg->block1_var = block1_variance;
	w_arg->block2_mv = block2_mv;
	w_arg->result = result_each_process;
	w_arg->blocks_size = blocks_size;
	w_arg->blocks2_id = blocks2_id;
	
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
		distributed_work_odd(w_arg);
	
	if(numprocs%2 ==0)
		distributed_work_even(w_arg);	
	
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
					
		
		printf("sort result to correlation matrix\n");
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
//							printf(" correaltion matrix[%d][%d] = %lf\n",x,y,correlation_matrix[x][y]);
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
