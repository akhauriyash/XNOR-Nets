///////////////////////////////////////////////////////////////////////////////
// xCONV.c - Tests for Intel(R) Xeon Phi(TM) Processor.
//				Implemented by Yash Akhauri.
// Notes:
//		- Performance tests matrix multiply algorithms on a Intel Xeon Phi 7210 Processor.
//		- To compile, make sure the directory of echo ~/_director_/xconv.out | qsub matches.
// To Compile:
//	Xeon Phi KNL 7210
//		icpc -xMIC-AVX512 -qopenmp -mkl -fp-model fast=2 -fma -unroll=4 xCONVbench.c -o xconvbench.out && echo ~/xconvbench.out | qsub 
//	Xeon Gold 6128
//		icpc -xCORE-AVX512 -qopenmp -mkl -fp-model fast=2 -fma -unroll=4 xCONVbench.c -o xconvbench.out && echo ~/xconvbench.out | qsub 
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

#define K_SIZE 				4					//	Choose kernel size : Currently only supports 4x4
#define NUM_K_ELEM			K_SIZE*K_SIZE

#define FPUTYPE				float
#define BINTYPE				unsigned short

#define NUM_OF_THREADS		256
#define TEST_LOOP			10

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
	int MX_SIZE;
	for(int size_setter=0; size_setter<=7; size_setter++){
		float bintime_tot = 0;
		MX_SIZE = 64*(pow(2, size_setter));								// Takes matrix size  
		size_t r, c, rk, ck;
		size_t i, j, k, ii, jj, kk, sm;
		double dTimeS, dTimeE;
		r = c = MX_SIZE;
		rk = ck = K_SIZE;
		printf("Number of test loops %d\n", TEST_LOOP);
		printf("Matrix size: %d x %d \n Kernel Size %d x %d \n", r, c, rk, ck);
		putenv("KMP_AFFINITY=scatter");								// OK
		// putenv("KMP_AFFINITY=balanced, granularity=fine");		// OK
		// putenv("KMP_AFFINITY=compact");							// BAD
		omp_set_num_threads(NUM_OF_THREADS);
		printf("Number of OpenMP threads: %3d\n", NUM_OF_THREADS);


	////////////////////////  Allocate full precision matrix 	///////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////

		__attribute__( ( aligned( 64 ) ) ) FPUTYPE **pA = NULL;
		pA = ( FPUTYPE ** )_mm_malloc(r*sizeof(FPUTYPE *), 64);
		for(int i = 0; i < r; i++){
			pA[i] = ( FPUTYPE * )_mm_malloc(c*sizeof(FPUTYPE), 64);
		}
		__attribute__( ( aligned( 64 ) ) ) FPUTYPE **kerA = NULL;
		kerA = ( FPUTYPE ** )_mm_malloc(rk*sizeof(FPUTYPE *), 64);
		for(int i = 0; i < rk; i++){
			kerA[i] = ( FPUTYPE *)_mm_malloc(ck*sizeof(FPUTYPE), 64);
		}
		__attribute__( ( aligned( 64 ) ) ) FPUTYPE **pC = NULL;
		pC = ( FPUTYPE ** )_mm_malloc((r-rk+1)*sizeof(FPUTYPE *), 64);
		for(int i = 0; i < (r-rk+1); i++){
			pC[i] = ( FPUTYPE * )_mm_malloc((c-ck+1)*sizeof(FPUTYPE), 64);
		}

		if(pA == NULL || kerA == NULL){
			printf( "\nERROR: Can't allocate memory for matrices\n" );
			_mm_free( pA );
			_mm_free( kerA );
			return ( int )0;
		}
		for(int i = 0; i < r; i++){
			for( j = 0; j < c; j++)
			{
				FPUTYPE x = (FPUTYPE) rand()/RAND_MAX;				// Create random
				pA[i][j] = ( x < 0.5 ) ? -1 : 1;					// +1/-1 matrices
			}
		}
		for(int i = 0; i < rk; i++){
			for( j = 0; j < ck; j++)
			{
				FPUTYPE x = (FPUTYPE) rand()/RAND_MAX;				// Create random
				kerA[i][j] = ( x < 0.5 ) ? -1 : 1;					// +1/-1 matrices
			}
		}

	////////////////////////	FP Matrix Convolution   	///////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////
		int accumulator;
	//	THIS LOOP ACTS AS WARMUP FOR BENCHMARKS. VERY CRITICAL FOR CORRECT RESULTS.
	//	This needs to be done only once, but just to be sure, why not. 
		for(int tloop = 0; tloop<4; tloop++){						// WARMUP
			#pragma omp parallel for private(i, j, ii, jj, accumulator) num_threads(NUM_OF_THREADS)
			for(int i = 0; i < r-rk+1; i++){						// WARMUP
				for(int j = 0; j < c-ck+1; j++){					// WARMUP
					accumulator = 0;								// WARMUP
					for(int ii = 0; ii < rk; ii++){					// WARMUP
						for(int jj = 0; jj < ck; jj++){				// WARMUP
							accumulator += pA[i+ii][j+jj]*kerA[ii][jj];
						}											// WARMUP
					}												// WARMUP
					pC[i][j] = accumulator;							// WARMUP
				}													// WARMUP
			}														// WARMUP
		}															// WARMUP

		dTimeS = dsecnd();
		for(int tloop = 0; tloop<TEST_LOOP; tloop++){
			#pragma omp parallel for private(i, j, ii, jj, accumulator) num_threads(NUM_OF_THREADS)
			for(int i = 0; i < r-rk+1; i++){
				for(int j = 0; j < c-ck+1; j++){
					accumulator = 0;
					for(int ii = 0; ii < rk; ii++){
						for(int jj = 0; jj < ck; jj++){
							accumulator += pA[i+ii][j+jj]*kerA[ii][jj];
						}
					}
					pC[i][j] = accumulator;
				}
			}
		}
		dTimeE = dsecnd();
		printf( "\n FP CONV - Completed in: %.7f seconds\n", ( dTimeE - dTimeS ) / TEST_LOOP);
		// //	Printing full precision convolution result
		// for(int i = 0; i < 5; i++){	
		// 	for(int j = 0; j < 5; j++){
		// 		printf("%.1f\t", pC[i][j]);
		// 	}
		// 	printf("\n");
		// }
		// printf("\n");

	////////////////////////	Allocate binary matrices   	///////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////
	// If data type is unsigned int, then the comment below is valid. If unsigned short, then no problem.
	// Post binarization: kerA: bkerA ==> 1 unsigned int 32 bit (high lane: 1s) [mxm] (assume m*m = 16)
	//					  pA:   bA    ==> Matrix: shape (n-m+1 x n-m+1) (Every 16 bit high lane 0s)

		__attribute__( ( aligned( 64 ) ) ) BINTYPE **bA = NULL;

		bA = ( BINTYPE ** )_mm_malloc((r-rk+1)*sizeof(BINTYPE *), 64);

		if(bA == NULL){
			printf( "\nERROR: Can't allocate memory for matrices\n" );
			_mm_free( bA );
			return ( int )0;
		}

		for(int i = 0; i < (r-rk+1); i++){
			bA[i] = ( BINTYPE * )_mm_malloc((c-ck+1)*sizeof(BINTYPE), 64);
		}


	////////////////////////  kerA pA binarized convolution ///////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////
		BINTYPE bkerA = 0;	int sign;	BINTYPE tbA = 0;
		for(int i = 0; i < rk; i++){
			for(int j = 0; j < ck; j++){
				// printf("%.1f\t", kerA[i][j]);						// For verification of binarization
				sign = (int) (kerA[i][j] >= 0);
				bkerA = bkerA|(sign<<(i*ck + j));
			}
		}
			// printf("\n");											// For verification of binarization
			// printBits(sizeof(bkerA), &bkerA);						// For verification of binarization
			// printf("\n");printf("\n");								// For verification of binarization

		dTimeS = dsecnd();
		// Time taken to binarize 16384 matrix: 1.089s (with pragma), 8.772s (without pragma)
		for(int tloop = 0; tloop<TEST_LOOP; tloop++){
			#pragma omp parallel for private(i, j, ii, jj, tbA, sign) num_threads(NUM_OF_THREADS)
			for(int i = 0; i < r-rk+1; i++){
				for(int j = 0; j < c-ck+1; j++){
					tbA = 0;
					for(int ii = 0; ii < rk; ii++){
						for(int jj = 0; jj < ck; jj++){
							sign = (int) (pA[i+ii][j+jj] >= 0);
							tbA = tbA|(sign<<(ii*ck + jj));
						}
					}
					pC[i][j] = 2*(__builtin_popcount(~(tbA^bkerA))) - 48;
				}
			}
		}	
		dTimeE = dsecnd();
		printf( "\n BIN+xCONV A - Completed in: %.7f seconds\n", ( dTimeE - dTimeS ) / TEST_LOOP);

			// for(int i = 0; i < 3; i++){							// For verification of binarization
			// 	for(int j = 0; j < 4; j++){							// For verification of binarization
			// 		tbA = 0;										// For verification of binarization
			// 		for(int ii = 0; ii < rk; ii++){					// For verification of binarization
			// 			for(int jj = 0; jj < ck; jj++){				// For verification of binarization
			// 				printf("%0.1f\t", pA[i+ii][j+jj]);		// For verification of binarization
			// 			}											// For verification of binarization
			// 		}												// For verification of binarization
			// 		printBits(sizeof(bA[i][j]), &bA[i][j]);			// For verification of binarization
			// 	}													// For verification of binarization
			// }													// For verification of binarization

		//	Printing binary convolution result
		// for(int i = 0; i < 5; i++){
		// 	for(int j = 0; j < 5; j++){
		// 		printf("%.1f\t", pC[i][j]);
		// 	}
		// 	printf("\n");
		// }
	}
}
