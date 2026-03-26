#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define EPSILON 1e-9

double rand_double() {
    return (double)rand() / RAND_MAX;
}

void sequential_gemm(double *A, double *B, double *C, int M, int N, int K) {
    for(int i=0;i<M;i++)
        for(int j=0;j<K;j++) {
            C[i*K+j]=0;
            for(int k=0;k<N;k++)
                C[i*K+j]+=A[i*N+k]*B[k*K+j];
        }
}

int verify(double *C1,double *C2,int size){
    for(int i=0;i<size;i++){
        if(fabs(C1[i]-C2[i])>EPSILON) return 0;
    }
    return 1;
}

int main(int argc,char *argv[]) {

    MPI_Init(&argc,&argv);

    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int M,N,K;

    if(rank==0){
        printf("Enter M N K: \n");
        fflush(stdout);
        (void)scanf("%d %d %d",&M,&N,&K);
    }

    MPI_Bcast(&M,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&K,1,MPI_INT,0,MPI_COMM_WORLD);

    int rows = M/size;

    double *A=NULL,*B=NULL,*C_pt=NULL,*C_col=NULL,*C_seq=NULL;

    if(rank==0){
        A=(double*)malloc(M*N*sizeof(double));
        B=(double*)malloc(N*K*sizeof(double));
        C_pt=(double*)malloc(M*K*sizeof(double));
        C_col=(double*)malloc(M*K*sizeof(double));
        C_seq=(double*)malloc(M*K*sizeof(double));

        srand(time(NULL));

        for(int i=0;i<M*N;i++) A[i]=rand_double();
        for(int i=0;i<N*K;i++) B[i]=rand_double();
    }

    double *localA=(double*)malloc(rows*N*sizeof(double));
    double *localC=(double*)malloc(rows*K*sizeof(double));

    if(rank!=0)
        B=(double*)malloc(N*K*sizeof(double));

/* =====================================================
   (a) MPI POINT-TO-POINT IMPLEMENTATION
   ===================================================== */

    if(rank==0){

        for(int p=1;p<size;p++){
            MPI_Send(A + p*rows*N, rows*N, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
        }

        for(int p=1;p<size;p++){
            MPI_Send(B, N*K, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
        }

        for(int i=0;i<rows*N;i++)
            localA[i]=A[i];

        for(int i=0;i<N*K;i++)
            B[i]=B[i];

    }
    else{

        MPI_Recv(localA, rows*N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B, N*K, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_pt = MPI_Wtime();

    for(int i=0;i<rows;i++)
        for(int j=0;j<K;j++){
            localC[i*K+j]=0;
            for(int k=0;k<N;k++)
                localC[i*K+j]+=localA[i*N+k]*B[k*K+j];
        }

    if(rank==0){

        for(int i=0;i<rows*K;i++)
            C_pt[i]=localC[i];

        for(int p=1;p<size;p++){
            MPI_Recv(C_pt + p*rows*K, rows*K, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

    }
    else{
        MPI_Send(localC, rows*K, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    double end_pt = MPI_Wtime();


/* =====================================================
   (b) MPI COLLECTIVE IMPLEMENTATION
   ===================================================== */

    MPI_Bcast(B,N*K,MPI_DOUBLE,0,MPI_COMM_WORLD);

    MPI_Scatter(A,rows*N,MPI_DOUBLE,localA,rows*N,MPI_DOUBLE,0,MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_col = MPI_Wtime();

    for(int i=0;i<rows;i++)
        for(int j=0;j<K;j++){
            localC[i*K+j]=0;
            for(int k=0;k<N;k++)
                localC[i*K+j]+=localA[i*N+k]*B[k*K+j];
        }

    MPI_Gather(localC,rows*K,MPI_DOUBLE,C_col,rows*K,MPI_DOUBLE,0,MPI_COMM_WORLD);

    double end_col = MPI_Wtime();


/* =====================================================
   VERIFICATION (NOT TIMED)
   ===================================================== */

    if(rank==0){

        sequential_gemm(A,B,C_seq,M,N,K);

        printf("\nPoint-to-Point Time: %f seconds\n", end_pt-start_pt);
        printf("Collective Time: %f seconds\n", end_col-start_col);

        if(verify(C_pt,C_seq,M*K))
            printf("Point-to-Point Verification PASSED\n");
        else
            printf("Point-to-Point Verification FAILED\n");

        if(verify(C_col,C_seq,M*K))
            printf("Collective Verification PASSED\n");
        else
            printf("Collective Verification FAILED\n");
    }

    MPI_Finalize();
}