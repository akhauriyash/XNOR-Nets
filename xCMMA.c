///////////////////////////////////////////////////////////////////////////////
// xCMMA.c - Tests for Intel(R) Xeon Phi(TM) Processor.
//				Implemented by Yash Akhauri.
// Notes:
//		- Performance tests matrix multiply algorithms on a Intel Xeon Phi 7210 Processor.
//		- To compile, make sure the directory of echo ~/_director_/xconv.out | qsub matches.
// To Compile:
//		icpc -xMIC-AVX512 -qopenmp -mkl -fp-model fast=2 -fma -unroll=4 xCMMA.c -o xcmma.out && echo ~/xcmma.out | qsub 
//
///////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <omp.h>
#include <math.h>
#include <mkl.h>
#include <iostream>

#define FPUTYPE				float
#define BINTYPE				unsigned int

// #define MX_SIZE				16384
// #define MX_SIZE				 8192
#define MX_SIZE					 4096
// #define MX_SIZE				 2048
// #define MX_SIZE				 1024
// #define MX_SIZE				  512
// #define MX_SIZE				  256


#define NUM_OF_THREADS			64
#define TEST_LOOP				100

// printBits prints the binary format of the unsigned int passed to it.
void printBits(size_t const size, void const * const ptr){
    unsigned char *b = (unsigned char*) ptr;
    unsigned char byte;
    int i, j;
    printf("\n");
    for (i=size-1;i>=0;i--)
        for (j=7;j>=0;j--)
        {
            byte = (b[i] >> j) & 1;
            printf("%u", byte);
        }
    puts("");    printf("\n");             
    }


int main( void )
{
	size_t m, n, p;
	size_t r, i, j, k, sm;
	double dTimeS, dTimeE;
	m = p = n = MX_SIZE;
	printf("Matrix size: %d x %d\n", m, p);
	putenv("KMP_AFFINITY=scatter");
	// putenv("KMP_AFFINITY=balanced, granularity=fine");
	// putenv("KMP_AFFINITY=compact");
	omp_set_num_threads(NUM_OF_THREADS);
	printf("Number of OpenMP threads: %3d\n", NUM_OF_THREADS);


////////////////////////  Allocate full precision matrices 	///////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

	__attribute__( ( aligned( 64 ) ) ) FPUTYPE **pA = NULL;			// Allocating memory 
	__attribute__( ( aligned( 64 ) ) ) FPUTYPE **pB = NULL;			// for matrices aligned
	__attribute__( ( aligned( 64 ) ) ) FPUTYPE **pC = NULL;			// on 64-byte boundary


	pA = ( FPUTYPE ** )_mm_malloc(m*sizeof(FPUTYPE *), 64);			// These loops can 
	for(int i = 0; i < m; i++){										// be collapsed
		pA[i] = ( FPUTYPE * )_mm_malloc(p*sizeof(FPUTYPE), 64);		// as m = n = p = MX_SIZE
	}
	pB = ( FPUTYPE ** )_mm_malloc(p*sizeof(FPUTYPE *), 64);
	for(int i = 0; i < p; i++){
		pB[i] = ( FPUTYPE * )_mm_malloc(n*sizeof(FPUTYPE), 64);
	}
	pC = ( FPUTYPE ** )_mm_malloc(m*sizeof(FPUTYPE *), 64);
	for(int i = 0; i < m; i++){
		pC[i] = ( FPUTYPE * )_mm_malloc(n*sizeof(FPUTYPE), 64);
	}
	if( pA == NULL || pB == NULL || pC == NULL )					// Error handling 
	{																// if any array is
		printf( "\nERROR: Can't allocate memory for matrices\n" );	// not allocated
		_mm_free( pA );
		_mm_free( pB );
		_mm_free( pC );
		return ( int )0;
	}
	for(int j = 0; j < m; j++){
		for( i = 0; i < p; i++)
		{
			FPUTYPE x = (FPUTYPE) rand()/RAND_MAX;					// Create random
			pA[j][i] = ( x < 0.5 ) ? -1 : 1;						// +1/-1 matrices
		}
	}
	for(int j = 0; j < p; j++){
		for( i = 0; i < n; i++)
		{
			FPUTYPE x = (FPUTYPE) rand()/RAND_MAX;					// Create random
			pB[j][i] = ( x > 0.5 ) ? -1 : 1;						// +1/-1 matrices
		}
	}
	for(int j = 0; j < m; j++){
		for( i = 0; i < n; i++)
		{
			pC[j][i] = 0;
		}
	}
////////////////////////	Allocate binary matrices 	///////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

	__attribute__( ( aligned( 64 ) ) ) BINTYPE **bA = NULL;		// Allocated binary
	__attribute__( ( aligned( 64 ) ) ) BINTYPE **bB = NULL;		// matrices A and B

	bA = ( BINTYPE ** )_mm_malloc(m*sizeof(BINTYPE *), 64);
	bB = ( BINTYPE ** )_mm_malloc(n*sizeof(BINTYPE *), 64);
		if( bA == NULL || bB == NULL )					// Error handling 
		{																// if any array is
			printf( "\nERROR: Can't allocate memory for  matrices\n" );	// not allocated
			_mm_free( bA );
			_mm_free( bB );
			return ( int )0;
		}
	for(int i = 0; i<m; i++){
		bA[i] = (BINTYPE *)_mm_malloc((p/32)*sizeof(BINTYPE), 64);
	}
	for(int i = 0; i<n; i++){
		bB[i] = (BINTYPE *)_mm_malloc((p/32)*sizeof(BINTYPE), 64);
	}
////////////////////////	FP Matrix multiplication 	///////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
	float sum = 0;
	dTimeS = dsecnd();
	for(int jj = 0; jj < TEST_LOOP; jj++){
		#pragma omp parallel for private(i, j, k, sum) num_threads(NUM_OF_THREADS)
		for(int i = 0; i < m; i++){
			for(int j = 0; j < n; j++){
				sum = 0.0;
				for(int k = 0; k < p; k++){
					sum += pA[i][k]*pB[k][j];
				}
				pC[i][j] = sum;
			}
		}
	}
	dTimeE = dsecnd();
	printf( "\nFull precision CMMA - Completed in: %.7f seconds\n", ( dTimeE - dTimeS ) / (double) TEST_LOOP);	
	printf("\nFull precision multiplication result:\n");			
	for(int i = 0; i<4; i++){
		for(j = 0; j<5; j++){
			printf("%f\t", pC[i][j]);
		}
		printf("\n");
	}	printf("\n");



////////////////////////	  Binarization of A&B   	///////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

	int sign; BINTYPE tbA; BINTYPE tbB;

	dTimeS = dsecnd();
	for(int jj = 0; jj < TEST_LOOP; jj++){
		#pragma omp parallel for
		for (int i = 0; i < MX_SIZE; i++)
		{
			for(int seg = 0; seg < MX_SIZE/32; seg++)
			{
				tbA = 0;
				for(int j = 0; j < 32; j++)
				{//					[i*n + seg*32 + j]		For flattened matrices
					sign = (int) (pA[i][seg*32 + j] >= 0);
					tbA = tbA|(sign<<j);
				}
			bA[i][seg] = tbA;
			}
		}
	}
	dTimeE = dsecnd();
	printf( "\nBinarization A - Completed in: %.7f seconds\n", ( dTimeE - dTimeS ) / TEST_LOOP);
	

	dTimeS = dsecnd();
	for(int jj = 0; jj < TEST_LOOP; jj++){
		#pragma omp parallel for
		for (int i = 0; i < MX_SIZE; i++)
		{
			for(int seg = 0; seg < MX_SIZE/32; seg++)
			{
				tbB = 0;
				for(int j = 0; j < 32; j++)
				{//					[i+seg*32*n + j*n]		For flattened matrices
					sign = (int) (pB[seg*32 + j][i] >= 0);
					tbB = tbB|(sign<<j);
				}
			bB[i][seg] = tbB;
			}
		}
	}
	dTimeE = dsecnd();
	printf( "\nBinarization B - Completed in: %.7f seconds\n\n\n", ( dTimeE - dTimeS ) / TEST_LOOP );

//////////////////////// 	Binarized Multiplication  	///////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
	int temp;
	int storeto;
	dTimeS = dsecnd();
	for(int jj = 0; jj < TEST_LOOP; jj++){
		#pragma omp parallel for private(i, j, sm, temp) num_threads(NUM_OF_THREADS)
		for(int i = 0; i < m; i++){
			for(int j = 0; j < n; j++){
					temp = 0;
					for(int sm = 0; sm < p/32; sm++){
						temp += 2*(__builtin_popcount(~(bA[i][sm]^bB[j][sm]))) - 32;
					}
				pC[i][j] = temp;
			}
		}
	}
	dTimeE = dsecnd();
	printf( "\nBinarized Multiplication - Completed in: %.7f seconds\n\n\n", ( dTimeE - dTimeS ) / (double) TEST_LOOP);
	printf("\nBinarized multiplication result:\n");
	for(int i = 0; i<4; i++){
		for(j = 0; j<5; j++){
			printf("%f\t", pC[i][j]);
		}
		printf("\n");
	}
}
