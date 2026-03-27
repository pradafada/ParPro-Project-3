#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SIZE 1000000

int data[SIZE];
int seq_data[SIZE];

/* swap */
void swap(int* a, int* b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

/* partition */
int partition(int arr[], int low, int high)
{
    int pivot = arr[high];
    int i = low - 1;

    for(int j = low; j <= high - 1; j++)
    {
        if(arr[j] < pivot)
        {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }

    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}

/* sequential quicksort */
void quickSort(int arr[], int low, int high)
{
    if(low < high)
    {
        int pi = partition(arr, low, high);

        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

/* merge two sorted arrays */
void merge(int *a, int sizeA, int *b, int sizeB, int *result)
{
    int i=0,j=0,k=0;

    while(i<sizeA && j<sizeB)
    {
        if(a[i] < b[j])
            result[k++] = a[i++];
        else
            result[k++] = b[j++];
    }

    while(i<sizeA)
        result[k++] = a[i++];

    while(j<sizeB)
        result[k++] = b[j++];
}

/* verification */
int verify(int *parallel, int *sequential, int n)
{
    for(int i=0;i<n;i++)
    {
        if(parallel[i] != sequential[i])
            return 0;
    }
    return 1;
}

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int chunk = SIZE / size;
    int *local = malloc(chunk * sizeof(int));

    struct timespec start,end;
    MPI_Status status;

    /* ROOT reads data */
    if(rank == 0)
    {
        FILE *fp = fopen("data.txt","r");

        for(int i=0;i<SIZE;i++)
        {
            fscanf(fp,"%d",&data[i]);
            seq_data[i] = data[i];
        }

        fclose(fp);

        /* sequential sort for verification */
        quickSort(seq_data,0,SIZE-1);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    clock_gettime(CLOCK_MONOTONIC,&start);

    /* MANUAL DATA DISTRIBUTION USING MPI_Send */
    if(rank == 0)
    {
        for(int i=1;i<size;i++)
        {
            MPI_Send(data + i*chunk, chunk, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        memcpy(local, data, chunk*sizeof(int));
    }
    else
    {
        MPI_Recv(local, chunk, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }

    /* LOCAL QUICKSORT */
    quickSort(local,0,chunk-1);

    /* SEND SORTED CHUNKS BACK */
    if(rank == 0)
    {
        memcpy(data, local, chunk*sizeof(int));

        for(int i=1;i<size;i++)
        {
            MPI_Recv(data + i*chunk, chunk, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
        }
    }
    else
    {
        MPI_Send(local, chunk, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    /* ROOT MERGES SORTED CHUNKS */
    if(rank == 0)
    {
        int *temp = malloc(SIZE*sizeof(int));
        int current_size = chunk;

        memcpy(temp,data,chunk*sizeof(int));

        for(int i=1;i<size;i++)
        {
            merge(temp,current_size,
                  data + i*chunk,chunk,
                  seq_data);

            memcpy(temp,seq_data,(current_size+chunk)*sizeof(int));
            current_size += chunk;
        }

        memcpy(data,temp,SIZE*sizeof(int));

        free(temp);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    clock_gettime(CLOCK_MONOTONIC,&end);

    if(rank == 0)
    {
        unsigned long long diff =
        1000000000L*(end.tv_sec-start.tv_sec)
        + end.tv_nsec-start.tv_nsec;

        printf("MPI Parallel QuickSort Time = %llu ns\n",diff);

        if(verify(data,seq_data,SIZE))
            printf("Verification: SUCCESS\n");
        else
            printf("Verification: FAILED\n");
    }

    free(local);

    MPI_Finalize();
    return 0;
}