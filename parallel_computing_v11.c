/**
 * send mean of each sequence during shift of data,compare to v3**/
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

struct timeval t0, t1, dt;

//char *output_hpc = "/project/parallel/tao_tang/correlation_matrix.txt"; 
//char *input_hpc = "/project/parallel/tao_tang/data.txt";
char *output_hpc = "correlation_matrix.txt"; 
char *input_hpc = "data.txt";
char *directory_name = "/project/parallel/tao_tang/all_results/results_v11_test";
/** length of each vector**/
#define VECTOR_LENGTH 2725784
//#define VECTOR_LENGTH 35000
/** length of each interval in pairwise algorithm**/
int interval = 10000;
int n_node = 1;
int core_per_node = 1;
/**read data file into matrix **/
void read_data(double *matrix, int size, char *input_path)
{
	puts("loading data");
	FILE *input = fopen(input_path,"r");
	int i;
	for(i = 0; i < size; i++)
		fscanf(input,"%lf",matrix+i);
	fclose(input);
	puts("finish loading data");
	return;	
}


double normal_variance(double *x,int len)
{
	double x_bar = 0,s = 0;
	int i;
	for (i = 0; i < len; i++)
		x_bar += x[i];
	
	x_bar = x_bar/len;
	for (i = 0; i < len; i++)
		s += (x[i]-x_bar) * (x[i] - x_bar);
	return s;
		
}


void interval_variance(double *v,int len, double *s, double *T)
{
	double buffer;
	int i;
	(*T) = v[0];
	(*s) = 0;
	for (i = 1; i < len; i++)
	{
		(*T) += v[i];
		buffer = (i+1)*v[i]-(*T); 
		(*s) += buffer * buffer/((i+1)*i);
	}
	return;
}

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


double correlation(double *x, double *y, int len, double x_mean,double x_var, double y_mean, double y_var)
{
	double cov = 0;
	int i;
	for( i = 0; i < len ; i++)
		cov += x[i]*y[i];
		
	cov -= x_mean * y_mean * len;
//	if(x_var == y_var)
//		printf("xmean is %.2f, xvar is:%.2f, ymean is %.2f, yvar is %.2f, cov is %.2f\n",x_mean,x_var,y_mean,y_var,cov)
	return cov/(sqrt(x_var)*sqrt(y_var));
}

//sequential computing for self-checking
void sequential_process(double *matrix, double *result, int M, int N)
{
	int i,j,count=0;	
	int T_size = M/interval + 1;
	double *mean = malloc(sizeof(double) * N);
	double *variance = malloc(sizeof(double) * N);
	for ( i = 0; i < N; i++)
		vector_mean_variance(matrix + i * M, M, mean+i, variance+i);
	//perform operations between vectors in block1 and block2
	for( i = 0; i < N-1; i++)
	{
		for(j=i+1; j < N; j++)
			result[count++] = correlation(matrix+i*M,matrix+j*M,M,mean[i],variance[i],mean[j],variance[j]);
	}				
	free(mean);
	free(variance);
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



/**the main difference between odd version and even is, processes of even version does not calculate inner product of their block1 in the first step, in the last step, the
blocks contained in first half processes will be indentical to second half, first half calculate products between their block1 and block2, second half calculate inner product 
product of block1 and block2, which needs n^2 and n^2-n computations respectively, load balancing is still acheived **/

int even_version(int i_myid,int i_numprocs,int i_N,int i_M)
{
	int myid = i_myid;
	int numprocs = i_numprocs;
	int M = i_M;
	int N = i_N;
	int steps  = numprocs/2;
	int i,j,k,l,iter;
	int T_size = 1 + M/interval;
	MPI_Request *reqs;
	MPI_Status *stats;
	int v_b = N / numprocs;
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
	{
		blocks2_id[i] = (myid+i+1)%numprocs;
		printf(" %d ", blocks2_id[i]);	
	}
	puts("");
	//number of vectors each block contain
	int *blocks_size = malloc(sizeof(int)*numprocs);
	int *blocks_vector_id = malloc(sizeof(int)*numprocs);

	// id of the first vectors each block contain 
	blocks_vector_id[0] = 0;
	blocks_size[0] = (0 < remainder) ? v_b+1 : v_b;

	int count = 0;
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
	//maxmimu number of size of results computed by each process
	int max_results_size = v_b_buffer*(v_b_buffer-1)/2 + v_b_buffer * v_b_buffer * steps;
	//store results computeed by this process
	double *result = malloc( sizeof(double) * max_results_size );
	for( i = 0; i < max_results_size; i++)
		result[i] = 0.0;
 
		
	if(myid == 0)
	{
		//file name start from ...1.txt, so id needs to plus 1
		read_files(block1,blocks_vector_id[myid] + 1,blocks_size[myid], M);
		
		gettimeofday(&t0, NULL);	
		
		//used to store results sent from each process		
		double **results_from_each_process = malloc(sizeof(double *) * numprocs);
		results_from_each_process[0] = result;
		for(i =1; i <numprocs; i++)
			results_from_each_process[i] = malloc(sizeof(double) * max_results_size);
 
		puts("");
		//correations coefficients(format of ouput results) is a N-1 d array, 1st row has N-1 elements,second N-2,...
		double *correlations = malloc(sizeof(double) * N*(N-1)/2);
		double **correlation_matrix = malloc(sizeof(double *) * (N-1));		
		count = 0;
		for(i=0; i < N-1; i++)
		{
			correlation_matrix[i] = correlations + count;
			count += (N-i-1);
		}

		printf("process %d line 205\n",myid);
		reqs = malloc(sizeof(MPI_Request)*numprocs);
		stats = malloc(sizeof(MPI_Status)*numprocs);
		//second to last step
		MPI_Request req,req1;		
		MPI_Status stat,stat1;	

		//buffer to recevie block2 during iteration as MPI_Wait needs to wait Isend totally completed in hpc 
		double *block2_buffer = malloc(sizeof(double)*(v_b+1) * M);
		double *block2_mv_buffer = malloc(2* sizeof(double)*(v_b+1));		
		
		MPI_Isend(block1,M * blocks_size[0],MPI_DOUBLE, numprocs-1,TAG1, MPI_COMM_WORLD, &req);
		MPI_Recv(block2,M * blocks_size[(myid+1)%numprocs],MPI_DOUBLE,(myid+1)%numprocs,TAG1,MPI_COMM_WORLD,&stat);
		
		//calculate variance of the first block
		for(i = 0; i <blocks_size[myid]; i++)
			vector_mean_variance(block1+i*M, M, block1_mean+i, block1_variance + i);
		
		MPI_Wait(&req, &stat);			


		//receive mean and variance of block 2 from proecess 1
		MPI_Isend(block1_mv, 2*(v_b+1), MPI_DOUBLE, numprocs-1,TAG3,MPI_COMM_WORLD, &req);
		MPI_Recv(block2_mv, 2*(v_b+1),MPI_DOUBLE, 1,TAG3,MPI_COMM_WORLD, &stat);
		int result_count = 0;	
		
		for(iter = 0; iter < steps; iter++)
		{
			//after calculation, use non-blocking communcation to send block2 to the last process, last step doesn't need to exchange block
			if(iter < steps -1)
			{
				MPI_Isend(block2_mv, 2*(v_b+1) , MPI_DOUBLE,numprocs-1,TAG3,MPI_COMM_WORLD, &req1);
				MPI_Isend(block2, M*blocks_size[blocks2_id[iter]] , MPI_DOUBLE,numprocs-1,TAG1,MPI_COMM_WORLD, &req);			
			}
			//perform operations between vectors in block1 and block2
			for( i = 0; i < blocks_size[myid]; i++)
			{
				for(j=0; j < blocks_size[ blocks2_id[iter]]; j++)
					result[result_count++] = correlation(block1+i*M,block2+j*M,M,block1_mean[i],block1_variance[i],block2_mean[j],block2_variance[j]);
			}				
				
			//wait sending blocks to other process completed, then recive block(to ensure the sending block is unchanged)
			if( iter < steps - 1 )
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
			
		}
		//receive results from other processes
		for (i = 1; i < numprocs; i++)
			MPI_Irecv(results_from_each_process[i],max_results_size,MPI_DOUBLE,i,TAG2,MPI_COMM_WORLD,reqs+i-1);
		
		//sort results
		int result_size_each_iteration = 0;             
		int block1_size,block2_id,block2_size,x,y;
		block2_id = 0;
		MPI_Waitall(numprocs-1,reqs, stats);
		
		//measure running time
		gettimeofday(&t1, NULL);
		timersub(&t1, &t0, &dt);	
		
		for( i = 0; i < numprocs; i++)
		{
			printf("results from process %d:",i);
			for(j = 0; j < max_results_size; j++)
				printf(" %f", results_from_each_process[i][j]);
			puts("");
		}
		
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
						double a = results_from_each_process[i][count++];						
						correlation_matrix[x][y] = a;
				
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
		
		char *fp = malloc(sizeof(char) * 100);
		sprintf(fp,"%s/result_%dvectors_%dnodes_%dcores.txt",directory_name,N,n_node,numprocs);
		FILE *f = fopen(fp, "w");
		fprintf(f,"runningtime: %d.%06d seconds\n",(int)dt.tv_sec,(int)dt.tv_usec);
		fprintf(f,"complexity: %d\n ", N*(N-1)/2);
	    fprintf(f,"nodenumber: %d\n ",n_node);
	    fprintf(f,"core_per_node: %d\n ",core_per_node);
	    fprintf(f,"process number: %d\n ",numprocs);
	    free(fp);		
		for(i=0; i<N-1; i++) 
		{
			for(j= 0;j < N-i-1; j++)
				fprintf(f,"%lf ",correlation_matrix[i][j]);
			fprintf(f,"\n");
		}
		fclose(f);
		puts("final correaltion matrix is:");
		for(i = 0; i < N-1; i++)
		{
			puts("");
			for(j=0; j < N-1-i; j++)
				printf("%lf ",correlation_matrix[i][j]);
		} 
		puts("");
		free(block2_buffer);
		free(block2_mv_buffer);
		for(i =1; i <numprocs; i++)
			free(results_from_each_process[i]);
		free(results_from_each_process);	
		free(stats);
		free(reqs);
		free(correlations);
		free(correlation_matrix);
	}		

	
	else 
	{	
		double *block1 = malloc(sizeof(double) * (v_b+1) * M);
		double *block2 = malloc(sizeof(double) * (v_b+1) * M);
	
		read_files(block1,blocks_vector_id[myid] + 1,blocks_size[myid], M);
	
		MPI_Status stat,stat1;
		MPI_Request req,req1;

		MPI_Isend(block1,M * blocks_size[myid],MPI_DOUBLE,myid-1,TAG1,MPI_COMM_WORLD,&req);
		MPI_Recv(block2,M * blocks_size[(myid+1)%numprocs],MPI_DOUBLE,(myid+1)%numprocs,TAG1,MPI_COMM_WORLD,&stat);
		MPI_Wait(&req,&stat);
			
		for(i = 0; i <blocks_size[myid]; i++)
			vector_mean_variance(block1+i*M, M , block1_mean + i, block1_variance+i);

		printf("variance of block %d is\n",myid);
		for(i = 0; i < blocks_size[myid]; i++)
			printf(" %f",block1_variance[i]);
		puts("");

		
		//count number of results already generated
		int result_count = 0;		
	    //second to last step;
		MPI_Isend(block1_mv, 2*(v_b+1), MPI_DOUBLE, myid-1,TAG3,MPI_COMM_WORLD, &req1);
		MPI_Recv(block2_mv, 2*(v_b+1),MPI_DOUBLE, (myid+1)%numprocs,TAG3,MPI_COMM_WORLD, &stat1);	
		for(iter = 0; iter < steps-1; iter++)
		{
			//Isend block2 to other processes
			MPI_Isend(block2_mv, 2*(v_b+1) , MPI_DOUBLE,myid-1,TAG3,MPI_COMM_WORLD, &req1);					
			MPI_Isend(block2, M * blocks_size[blocks2_id[iter]], MPI_DOUBLE, myid-1,TAG1,MPI_COMM_WORLD,&req);

			//perform operations between vectors in block1 and block2
			for( i = 0; i < blocks_size[myid]; i++)
			{
				for(j=0; j < blocks_size[ blocks2_id[iter]]; j++)
					result[result_count++] = correlation(block1+i*M,block2+j*M,M,block1_mean[i],block1_variance[i],block2_mean[j],block2_variance[j]);
			}
			
			MPI_Wait(&req1,&stat1);
			MPI_Wait(&req,&stat); 
			MPI_Recv(block2_mv, 2*(v_b+1) , MPI_DOUBLE,(myid+1)%numprocs,TAG3,MPI_COMM_WORLD, &stat1);					
			MPI_Recv(block2, M * blocks_size[blocks2_id[iter+1]],MPI_DOUBLE, (myid+1)%numprocs,TAG1,MPI_COMM_WORLD,&stat);
		}
		
		if(myid < numprocs/2)
		{
			//perform operations between vectors in block1 and block2
			for( i = 0; i < blocks_size[myid]; i++)
			{
				for(j=0; j < blocks_size[ blocks2_id[iter]]; j++)
					result[result_count++] = correlation(block1+i*M,block2+j*M,M,block1_mean[i],block1_variance[i],block2_mean[j],block2_variance[j]);
			}
		}
		
		else
		{
			for(i = 0; i < blocks_size[myid] - 1; i++)
			{	
				for(j = i + 1; j < blocks_size[myid]; j++)
					result[result_count++] = correlation(block1+i*M,block1+j*M,M,block1_mean[i],block1_variance[i],block1_mean[j],block1_variance[j]);
			}
				
			int block2_id = blocks2_id[steps-1];
			for(i = 0; i < blocks_size[block2_id] - 1; i++)
			{	
				for(j = i + 1; j < blocks_size[block2_id]; j++)
					result[result_count++] = correlation(block2+i*M,block2+j*M,M,block2_mean[i],block2_variance[i],block2_mean[j],block2_variance[j]);
			}
			
		}
		MPI_Send(result,max_results_size,MPI_DOUBLE,0,TAG2,MPI_COMM_WORLD);	
		printf("results of process %d: ",myid);
		for( i = 0; i < max_results_size; i++)
			printf(" %lf", result[i]);
			
		puts("");
	}

	free(block1);
	free(block2);
	free(block1_mv);
	free(block2_mv);
	free(blocks2_id);
	free(blocks_size);
	free(blocks_vector_id);
	free(result);

	return 0;		
}


//when the numprocs is odd
int odd_version(int i_myid,int i_numprocs,int i_N,int i_M)
{
	int myid = i_myid;
	int numprocs = i_numprocs;
	int M = i_M;
	int N = i_N;
	int steps  = (numprocs-1)/2;
	int i,j,k,l,iter;
	int T_size = 1 + M/interval;
	MPI_Request *reqs;
	MPI_Status *stats;

	if(numprocs == 1)
	{
		double *matrix = malloc(sizeof(double) * M *N);
		printf("numprocs equals 1, only apply sequential computing\n");
		read_files(matrix,1,N, M);
		double *results = malloc(sizeof(double)  * N *(N-1)/2);

		gettimeofday(&t0, NULL);
	
		sequential_process(matrix,results,M,N);
		
		gettimeofday(&t1, NULL);
		timersub(&t1, &t0, &dt);			
		/**measure running time**/

		puts("create file");
		char *fp = malloc(sizeof(char) * 100);
		sprintf(fp,"%s/result_%dvetcors_%dnodes_%dcores.txt",directory_name,N,n_node,numprocs);
		printf("file path is:%s\n",fp);
		FILE *f = fopen(fp, "w");
		fprintf(f,"runningtime: %d.%06d seconds\n",(int)dt.tv_sec,(int)dt.tv_usec);
		fprintf(f,"complexity: %d\n ", N*(N-1)/2);
	    fprintf(f,"nodenumber: %d\n ",n_node);
	    fprintf(f,"core_pernode: %d\n ",core_per_node);
	    fprintf(f,"processnumberr: %d\n ",numprocs);
	    free(fp);
	    puts("writing result to file");
	    int count = 0;
		for(i=0; i<N-1; i++) 
		{
			for(j= 0;j < N-i-1; j++)
				fprintf(f,"%f ", results[count++]);
			fprintf(f,"\n");
		}
		fclose(f);
		puts("gonna free memory");
		free(results);
		free(matrix);
		return;
	}
	
	int v_b = N / numprocs;
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
	{
		blocks2_id[i] = (myid+i+1)%numprocs;
		printf(" %d ", blocks2_id[i]);	
	}
	puts("");
	//number of vectors each block contain
	int *blocks_size = malloc(sizeof(int)*numprocs);
	int *blocks_vector_id = malloc(sizeof(int)*numprocs);

	// id of the first vectors each block contain 
	blocks_vector_id[0] = 0;
	blocks_size[0] = (0 < remainder) ? v_b+1 : v_b;

	int count = 0;
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
	//maxmimu number of size of results computed by each process
	int max_results_size = v_b_buffer*(v_b_buffer-1)/2 + v_b_buffer * v_b_buffer * steps;
	//store results computeed by this process
	double *result = malloc( sizeof(double) * max_results_size );
	for( i = 0; i < max_results_size; i++)
		result[i] = 0.0;
 
	if(myid == 0)
	{
		//file name start from ...1.txt, so id needs to plus 1
		read_files(block1,blocks_vector_id[myid] + 1,blocks_size[myid], M);
		printf("process %d line 175\n",myid);
		
		reqs = malloc(sizeof(MPI_Request)*numprocs);
		stats = malloc(sizeof(MPI_Status)*numprocs);
		//buffer to recevie block2 during iteration as MPI_Wait needs to wait Isend totally completed in hpc 
		double *block2_buffer = malloc(sizeof(double)*(v_b+1) * M);
		double *block2_mv_buffer = malloc(2* sizeof(double)*(v_b+1));
		MPI_Request req,req1;		
		MPI_Status stat,stat1;	
		//used to store results recevived from othher processes		
		double **results_from_each_process = malloc(sizeof(double *) * numprocs);
		results_from_each_process[0] = result;
		for(i =1; i <numprocs; i++)
			results_from_each_process[i] = malloc(sizeof(double) * max_results_size);
		//correations coefficients(format of ouput results) is a N-1 d array, 1st row has N-1 elements,second N-2,...
		double *correlations = malloc(sizeof(double) * N*(N-1)/2);
		double **correlation_matrix = malloc(sizeof(double *) * (N-1));		
		count = 0;
		for(i=0; i < N-1; i++)
		{
			correlation_matrix[i] = correlations + count;
			count += (N-i-1);
		}
						
		gettimeofday(&t0, NULL);
						
		//send block 1 to last process
		MPI_Isend(block1,M * blocks_size[0],MPI_DOUBLE, numprocs-1,TAG1, MPI_COMM_WORLD, &req);
		MPI_Recv(block2,M * blocks_size[(myid+1)%numprocs],MPI_DOUBLE,(myid+1)%numprocs,TAG1,MPI_COMM_WORLD,&stat);
		for(i = 0; i <blocks_size[myid]; i++)
		    vector_mean_variance(block1+i*M,M,block1_mean+i,block1_variance+i);
	
		printf("variance of block %d is\n",myid);
		for(i = 0; i < blocks_size[myid]; i++)
			printf(" %f",block1_variance[i]);
		puts("");	
			
		MPI_Wait(&req, &stat);
		
		int result_count = 0;	
		/** perform operation on vecotr pairs in block1 **/
		for(i = 0; i < blocks_size[myid] - 1; i++)
		{	
			for(j = i + 1; j < blocks_size[myid]; j++)
				result[result_count++] =correlation(block1+i*M,block1+j*M,M,block1_mean[i],block1_variance[i],block1_mean[j],block1_variance[j]);
		}
		//second to last step
		//receive mean and variance of block 2 from proecess 1
		MPI_Isend(block1_mv, 2*(v_b+1), MPI_DOUBLE, numprocs-1,TAG3,MPI_COMM_WORLD, &req);
		MPI_Recv(block2_mv, 2*(v_b+1),MPI_DOUBLE, 1,TAG3,MPI_COMM_WORLD, &stat);			

		printf("process %d line 830\n",myid);
		for(iter = 0; iter < steps; iter++)
		{
			printf("iteration %d process %d\n",iter,myid);
			//use non-blocking communcation to send block2 to the last process, last step doesn't need to exchange block
			if(iter < steps -1)
			{
				MPI_Isend(block2_mv, 2*(v_b+1) , MPI_DOUBLE,numprocs-1,TAG3,MPI_COMM_WORLD, &req1);
				MPI_Isend(block2, M*blocks_size[blocks2_id[iter]] , MPI_DOUBLE,numprocs-1,TAG1,MPI_COMM_WORLD, &req);

			}
			//perform operations between vectors in block1 and block2
			for( i = 0; i < blocks_size[myid]; i++)
			{
				for(j=0; j < blocks_size[ blocks2_id[iter]]; j++)
					result[result_count++] = correlation(block1+i*M,block2+j*M,M,block1_mean[i],block1_variance[i],block2_mean[j],block2_variance[j]);
			}
				
	
			//wait sending blocks to other process completed, then recive block(to ensure the sending block is unchanged)
			if( iter < steps - 1 )
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
		}
		printf("process %d line 824\n",myid);	
		//receive results from other processes
		for (i = 1; i < numprocs; i++)
		{
			printf("get results from process %d\n",i);
			MPI_Irecv(results_from_each_process[i],max_results_size,MPI_DOUBLE,i,TAG2,MPI_COMM_WORLD,reqs+i-1);
		}
		printf("process %d line 829\n",myid);	
		int result_size_each_iteration = 0;             
		int block1_size,block2_id,block2_size,x,y;
		MPI_Waitall(numprocs-1,reqs, stats);
		
		/**measure running time**/
		gettimeofday(&t1, NULL);
		timersub(&t1, &t0, &dt);
		
		for( i = 0; i < numprocs; i++)
		{
			printf("results from process %d:",i);
			for(j = 0; j < max_results_size; j++)
				printf(" %f", results_from_each_process[i][j]);
			puts("");
		}
		//sort results to correlation matrix
		for( i = 0; i < numprocs; i++)
		{
			printf(" process %d line 279\n ",i);
			block1_size = blocks_size[i];
			count = 0;
			result_size_each_iteration = block1_size * (block1_size - 1)/2;
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
//						printf("x: %d, y: %d \n", x, y);
						double a = results_from_each_process[i][count++];						
//						printf("insert result %.1f\n",a);
						correlation_matrix[x][y] = a;
					}
				}	
			}
		}
		printf("write result:\n");
		char *fp = malloc(sizeof(char) * 150);
		sprintf(fp,"%s/result_%dvetcors_%dnodes_%dcores.txt",directory_name,N,n_node,numprocs);
		FILE *f = fopen(fp, "w");
		fprintf(f,"runningtime: %d.%06d seconds\n",(int)dt.tv_sec,(int)dt.tv_usec);
		fprintf(f,"complexity: %d\n ", N*(N-1)/2);
	    fprintf(f,"nodenumber: %d\n ",n_node);
	    fprintf(f,"core_pernode: %d\n ",core_per_node);
	    fprintf(f,"processnumberr: %d\n ",numprocs);
	    free(fp);
	    
		for(i=0; i<N-1; i++) 
		{
			for(j= 0;j < N-i-1; j++)
				fprintf(f,"%lf ",correlation_matrix[i][j]);
			fprintf(f,"\n");
		}
		fclose(f);

		puts("final correaltion matrix is:");
		for(i = 0; i < N-1; i++)
		{
			puts("");
			for(j=0; j < N-1-i; j++)
				printf("%lf ",correlation_matrix[i][j]);
		} 
		puts("");
		for(i =1; i <numprocs; i++)
			free(results_from_each_process[i]);
		free(results_from_each_process);	
		free(stats);
		free(reqs);
		free(correlations);
		free(correlation_matrix);
		free(block2_buffer);
		free(block2_mv_buffer);
	}
	
	else 
	{	
		MPI_Status stat,stat1;
		MPI_Request req,req1;

		//file name start from ...1.txt, so id needs to plus 1
		read_files(block1,blocks_vector_id[myid] + 1,blocks_size[myid], M);		
			
		MPI_Isend(block1,M * blocks_size[myid],MPI_DOUBLE,myid-1,TAG1,MPI_COMM_WORLD,&req);
		MPI_Recv(block2,M * blocks_size[(myid+1)%numprocs],MPI_DOUBLE,(myid+1)%numprocs,TAG1,MPI_COMM_WORLD,&stat);
		
		for(i = 0; i <blocks_size[myid]; i++)
		    vector_mean_variance(block1+i*M,M,block1_mean+i,block1_variance+i);
			
		printf("variance of block %d is\n",myid);
		for(i = 0; i < blocks_size[myid]; i++)
			printf(" %f",block1_variance[i]);
		puts("");	
		
		MPI_Wait(&req,&stat);
		//count number of results already generated
		int result_count = 0;		
		// perform operation on vecotr pairs in block1
		for(i = 0; i < blocks_size[myid] - 1; i++)
		{	
			for(j = i + 1; j < blocks_size[myid]; j++)
				result[result_count++] =correlation(block1+i*M,block1+j*M,M,block1_mean[i],block1_variance[i],block1_mean[j],block1_variance[j]);
		}	
		printf("line 979 process %d\n",myid);
	    //second to last step;
		MPI_Isend(block1_mv, 2*(v_b+1), MPI_DOUBLE, myid-1,TAG3,MPI_COMM_WORLD, &req1);
		MPI_Recv(block2_mv, 2*(v_b+1),MPI_DOUBLE, (myid+1)%numprocs,TAG3,MPI_COMM_WORLD, &stat1);	
		printf("line 983 process %d\n",myid);
		for(iter = 0; iter < steps; iter++)
		{
			//Isend block2 to other processes
			if(iter < steps -1)
			{
				MPI_Isend(block2_mv, 2*(v_b+1) , MPI_DOUBLE,myid-1,TAG3,MPI_COMM_WORLD, &req1);				
				MPI_Isend(block2, M * blocks_size[blocks2_id[iter]], MPI_DOUBLE, myid-1,TAG1,MPI_COMM_WORLD,&req);
			}
	     //perform operations between vectors in block1 and block2
			for( i = 0; i < blocks_size[myid]; i++)
			{
				for(j=0; j < blocks_size[ blocks2_id[iter]]; j++)
					result[result_count++] = correlation(block1+i*M,block2+j*M,M,block1_mean[i],block1_variance[i],block2_mean[j],block2_variance[j]);
			}			
			
			if (iter < steps - 1)
			{
				MPI_Wait(&req1,&stat1);
				MPI_Wait(&req,&stat); 
				MPI_Recv(block2_mv, 2*(v_b+1) , MPI_DOUBLE,(myid+1)%numprocs,TAG3,MPI_COMM_WORLD, &stat1);					
				MPI_Recv(block2, M * blocks_size[blocks2_id[iter+1]],MPI_DOUBLE, (myid+1)%numprocs,TAG1,MPI_COMM_WORLD,&stat);
			}
		}
		
		printf("line 936 process %d\n ",myid);		
		//send results to the first process
		MPI_Send(result,max_results_size,MPI_DOUBLE,0,TAG2,MPI_COMM_WORLD);	
		printf("results of process %d\n ",myid);
		for( i = 0; i < max_results_size; i++)
			printf(" %f", result[i]);
			
		puts("");
	}

	//free allocated memory
	free(block1);
	free(block2);
	free(block1_mv);
	free(block2_mv);
	free(blocks2_id);
	free(blocks_size);
	free(blocks_vector_id);
	free(result);

	return 0;	
	
}
//note, this algorithm only for condition that numporcs is odd 
int main(int argc, char **argv)
{
	int myid,numprocs,length;
	//determine rank of each preocessor
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	
	char name[50];
	MPI_Get_processor_name(name, &length);
	
	int N,M;
	n_node = atoi(argv[1]);
	core_per_node = atoi(argv[2]);
	N = atoi(argv[3]);
	mkdir(directory_name, 0777);
	
	M = 49500;
//	M = VECTOR_LENGTH;
	if(numprocs%2 == 1)
		odd_version(myid,numprocs,N,M);
		
	else
		even_version(myid,numprocs,N,M);

	//record which node this process belongs to
	char *fp = malloc(sizeof(char) * 200);
	sprintf(fp,"/project/parallel/tao_tang/all_results/core_record/%dnodes_%dcores.txt",n_node,numprocs);
	FILE *f = fopen(fp, "a");
	fprintf(f,"record from process %s,id %d of %d\n",name,myid,numprocs);
	fclose(f);
	free(fp);

	MPI_Finalize();
	return 0;

}
