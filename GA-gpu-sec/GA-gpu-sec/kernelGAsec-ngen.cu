
using namespace std;  // permite usar el "cout"
#include <iostream>
#include <algorithm>
#include <stdlib.h>/* srand, rand */
#include "time.h"       /* time */
#include <stdio.h>      /* printf */
#include <math.h> 
#include <fstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<Cuda.h>
#include<curand.h>
#include<curand_kernel.h>
#include "stdlib.h"
#include<windows.h>

// Variables globales ****************************************************************************************
const int nvdec = 16 ;
const int nvbin = 4 ; // MULTIPLO nvbin*nvdec DE 2^n !!!
const int nvars = nvbin*nvdec ;
const int psize = 1920 ; // MULTIPLO DE 32!!!
const int ngen  = 1000000; //%10; 25; 10
const int nelit = 1 ; //cantidad de individuos del ELIT 
float mutp = 0.01 ; // Probabiludad de mutación
float tol = 1e-4 ;

// *************** ESTRUCTURAS *******************************************************************************
struct indiv {
  float Sol;
  int Ind;
} ;

bool lessthan(const indiv &a, const indiv &b) {
  return (b.Sol < a.Sol);
}

// *************** DEVICE FUNCTIONS **************************************************************************

__global__ void setup_rand ( curandState * state, unsigned long seed ) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init ( seed, id, 0, &state[id] );   }

__device__ float generate( curandState* globalState, int ind ) {
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState;
    return RANDOM;
}

__device__ float BinDec(int bin[nvbin], int n){
// convierte el vector binario ind a entero
	int sum = 0, two = 2;
	for(int i = 0; i<n; i++){  sum = sum + bin[i]*powf(two,i) ;  }
	return sum;
}

__global__ void InitPop(int *d_Pop, int nvars, curandState* globalState){   
// Genera la poblacion inicial
	int it = blockDim.x * blockIdx.x + threadIdx.x;
// PRUEBA	
	//for(int i=0; i<nvars; i++){ d_Pop[it*nvars + i] = it ; 	}
	//for(int i=0; i<nvars; i++){ d_Pop[it*nvars + i] = 1 ; 	}
	//for(int i=it; i<nvars+it; i++){ d_Pop[i] = 1 ; 	}
	//if(it < 1 ){d_Pop[ 3 ] = 0 ; d_Pop[ 6+it] = 0 ; d_Pop[8+it] = 0 ;d_Pop[ 9+it] = 0 ; d_Pop[22+it] = 0 ; d_Pop[29+it] = 0 ; }	
	//if(it = 1 ){d_Pop[ 2+it*64] = 0 ; d_Pop[5+it*64] = 0 ; d_Pop[7+it*64] = 0 ;d_Pop[ 8+it*64] = 0 ; d_Pop[21+it*64] = 0 ; d_Pop[30+it*64] = 0 ; }	
// FIN PRUEBA
		
	for(int i=0; i<nvars ; i++){
		float k = generate(globalState, i+it)*1 ; //i+it
		d_Pop[it*nvars + i] = lroundf(k) ; 
	}
}

__global__ void InitPop_s(int *d_Pop, int ipop, int nvars, curandState* globalState){   
// Genera la poblacion inicial
	int it = blockDim.x * blockIdx.x + threadIdx.x;
// PRUEBA	
	//for(int i=0; i<nvars; i++){ d_Pop[it*nvars + i] = it ; 	}
	//for(int i=0; i<nvars; i++){ d_Pop[it*nvars + i] = 1 ; 	}
	//for(int i=it; i<nvars+it; i++){ d_Pop[i] = 1 ; 	}
	//if(it < 1 ){d_Pop[ 3 ] = 0 ; d_Pop[ 6+it] = 0 ; d_Pop[8+it] = 0 ;d_Pop[ 9+it] = 0 ; d_Pop[22+it] = 0 ; d_Pop[29+it] = 0 ; }	
	//if(it = 1 ){d_Pop[ 2+it*64] = 0 ; d_Pop[5+it*64] = 0 ; d_Pop[7+it*64] = 0 ;d_Pop[ 8+it*64] = 0 ; d_Pop[21+it*64] = 0 ; d_Pop[30+it*64] = 0 ; }	
// FIN PRUEBA
		
	for(int i=0; i<nvars ; i++){
		float k = generate(globalState, i+ipop)*1 ; //i+it
		d_Pop[ipop*nvars + i] = lroundf(k) ; 
	}
}

__global__ void Func(float *d_Sol, int *d_Pop, int nvbin, int nvdec){   
// Genera la poblacion inicial
	int it = blockDim.x * blockIdx.x + threadIdx.x;
	
	int ipop=0, two=2, ten=10, bin[4];
	float x=0, y=0, X=0, Y=0, sum = 0, PI=3.141592653;
	float lim = 9/(powf(2,nvbin)-1) ;

	for (int inum=1; inum<nvdec/two; inum++){
		for(int i = 0; i<nvbin; i++){  bin[i] = d_Pop[it*nvbin*nvdec + nvbin*inum + i] ; }
		X = lroundf(BinDec(bin, nvbin)*lim) ;
		for(int i = 0; i<nvbin; i++){  bin[i] = d_Pop[it*nvbin*nvdec + nvbin*nvdec/two + nvbin*inum + i] ; }
		Y = lroundf(BinDec(bin, nvbin)*lim) ;
		x = x + powf(ten,(two-inum))*X ;
        y = y + powf(ten,(two-inum))*Y ;
	}
	
	for(int i = 0; i<nvbin; i++){  bin[i] = d_Pop[it*nvbin*nvdec + i] ; }
	if(BinDec(bin,nvbin) < (powf(two,nvbin)-1)*0.5 ){ x = (-1)*x ; }	
	for(int i = 0; i<nvbin; i++){  bin[i] = d_Pop[it*nvbin*nvdec + nvbin*nvdec/two + i] ; }
	if(BinDec(bin,nvbin) < (powf(two,nvbin)-1)*0.5 ){ y = (-1)*y ; }
	
	d_Sol[it] = x*x + y*y ;  
	//d_Sol[it] = X ;
	// d_Pop[it*nvbin*nvdec]; // Paraboloide eliptico ;//d_Pop[it+3];//[it*nvbin*nvdec];//X;//BinDec(bin, nvbin) ; //sum ; // 
	//return 
	// return 0.01*(x*x + y*y) + pow(sin( x*x + y*y), 2) ;  
	// Rastrigin's function ***/
	// return 10*two + (x*x-10*cos(two*PI*x)) + (y*y-10*cos(two*PI*y)) ;
}

__global__ void Func_s(float *d_Sol, int *d_Pop, int ipop, int nvbin, int nvdec){   
// Genera la poblacion inicial
	int it = ipop;
	
	int two=2, ten=10, bin[4];
	float x=0, y=0, X=0, Y=0, sum = 0, PI=3.141592653;
	float lim = 9/(powf(2,nvbin)-1) ;

	for (int inum=1; inum<nvdec/two; inum++){
		for(int i = 0; i<nvbin; i++){  bin[i] = d_Pop[it*nvbin*nvdec + nvbin*inum + i] ; }
		X = lroundf(BinDec(bin, nvbin)*lim) ;
		for(int i = 0; i<nvbin; i++){  bin[i] = d_Pop[it*nvbin*nvdec + nvbin*nvdec/two + nvbin*inum + i] ; }
		Y = lroundf(BinDec(bin, nvbin)*lim) ;
		x = x + powf(ten,(two-inum))*X ;
        y = y + powf(ten,(two-inum))*Y ;
	}
	
	for(int i = 0; i<nvbin; i++){  bin[i] = d_Pop[it*nvbin*nvdec + i] ; }
	if(BinDec(bin,nvbin) < (powf(two,nvbin)-1)*0.5 ){ x = (-1)*x ; }	
	for(int i = 0; i<nvbin; i++){  bin[i] = d_Pop[it*nvbin*nvdec + nvbin*nvdec/two + i] ; }
	if(BinDec(bin,nvbin) < (powf(two,nvbin)-1)*0.5 ){ y = (-1)*y ; }
	
	d_Sol[it] = x*x + y*y ;  
	//d_Sol[it] = X ;
	// d_Pop[it*nvbin*nvdec]; // Paraboloide eliptico ;//d_Pop[it+3];//[it*nvbin*nvdec];//X;//BinDec(bin, nvbin) ; //sum ; // 
	//return 
	// return 0.01*(x*x + y*y) + pow(sin( x*x + y*y), 2) ;  
	// Rastrigin's function ***/
	// return 10*two + (x*x-10*cos(two*PI*x)) + (y*y-10*cos(two*PI*y)) ;
}

__global__ void Rearrange(int *vec2, int *vec1, int *ind, int nvars, int nvec) {
    int it = blockDim.x * blockIdx.x + threadIdx.x;
	//blockDim = cantidad de hilos por bloque, es una constante [nvbin*nvdec] 64
	//blockIdx = nombre del bloque [identifica a la pareja] de 1 a psize
	//threadIdx = enumeracion del hilo dentro del bloque [cada variable de un ind.] de 1 a 64

	if (it < nvec) { vec2[it] = vec1[ind[blockIdx.x]*nvars + threadIdx.x] ; }  
		//NPop[it*2]   = Pop[  Male[blockIdx.x]*nvars + threadIdx.x*2    ] ;
	//vec2[it] = it ;
}

__global__ void Rearrange_s(int *vec2, int *vec1, int *ind, int nvars, int nvec, int it, int iblo, int ithr) {
    //int it = id; blockDim.x * blockIdx.x + threadIdx.x;
	//blockDim = cantidad de hilos por bloque, es una constante [nvbin*nvdec] 64
	//blockIdx = nombre del bloque [identifica a la pareja] de 1 a psize
	//threadIdx = enumeracion del hilo dentro del bloque [cada variable de un ind.] de 1 a 64

	if (it < nvec) { vec2[it] = vec1[ind[iblo]*nvars + ithr] ; }  
}

__global__ void Probability(float *vec2, float *vec1, int nvec, float alpha) {
    int it = blockDim.x * blockIdx.x + threadIdx.x;
	if (it < nvec) {
		vec2[it]=0 ;	
		for(int ivec=0; ivec<=it; ivec++){ 
			vec2[it]=vec2[it]+vec1[ivec]/alpha ; 
		}
	}
}

__global__ void CumSumVec(float *Sum, float *vec, int nvec) {
    int it = blockDim.x * blockIdx.x + threadIdx.x;
	if (it < nvec) { 
		__syncthreads();
		atomicAdd(Sum, vec[it]) ;
	}
}

__global__ void EqualityINT(int *vec2, int *vec1, int ivec, int nvec) {
    int it = blockDim.x * blockIdx.x + threadIdx.x;
	if (it < nvec) { vec2[it] = vec1[it+ivec] ; }  
	//vec2[it] = it ;
}

__global__ void EqualityINT_s(int *vec2, int *vec1, int ivec, int ipop, int nvec) {
    int it = ipop;
	if (it < nvec) { vec2[it] = vec1[it+ivec] ; }  
	//vec2[it] = it ;
}

__global__ void RandomINT(int *vec, int nvec, int psize, curandState* globalState){
// Genera un vector con "n" valores aleatorios enteros entre 0 y psize	
	int it = blockDim.x * blockIdx.x + threadIdx.x;
	if (it < nvec) {
		float k = generate(globalState, it)*(psize-0.5) ;
		vec[it] = lroundf(k) ; 
	}
}

__global__ void RandomINT_s(int *vec, int nvec, int psize, int ivar, curandState* globalState){
// Genera un vector con "n" valores aleatorios enteros entre 0 y psize	
	int it = ivar;
	if (it < nvec) {
		float k = generate(globalState, it)*(psize-0.5) ;
		vec[it] = lroundf(k) ; 
	}
}

__global__ void GroupSelection(int *sel, int *group, int gsize, int ngroup){
// Genera un vector con "n" valores aleatorios enteros entre 0 y psize	
	int it = blockDim.x * blockIdx.x + threadIdx.x;

	if(it<ngroup){
		sel[it] = group[it*gsize] ;
		for(int i=0; i<gsize; i++){
			if (sel[it] < group[it*gsize + i] ) { sel[it] = group[it*gsize + i] ; }
		}
	}
}

__global__ void GroupSelection_s(int *sel, int *group, int gsize, int ngroup, int ivar){
// Genera un vector con "n" valores aleatorios enteros entre 0 y psize	
	int it = ivar;

	if(it<ngroup){
		sel[it] = group[it*gsize] ;
		for(int i=0; i<gsize; i++){
			if (sel[it] < group[it*gsize + i] ) { sel[it] = group[it*gsize + i] ; }
		}
	}
}

__global__ void CrossoverSingle(int *NPop, int *Male, int *Female, int *Pop, int psize, int nvars){
// se llama una vez por cada 2  individuos y genera el cruzamiento entre el padre y la madre
	int it = blockDim.x * blockIdx.x + threadIdx.x ;
	//blockDim = cantidad de hilos por bloque, es una constante [nvbin*nvdec] 64
	//blockIdx = nombre del bloque [identifica a la pareja] de 1 a psize/2
	//threadIdx = enumeracion del hilo dentro del bloque [cada variable de un ind.] de 1 a 64
	
	if(threadIdx.x < nvars/2){
		NPop[it*2]   = Pop[  Male[blockIdx.x]*nvars + threadIdx.x*2    ] ;
		NPop[it*2+1] = Pop[Female[blockIdx.x]*nvars + threadIdx.x*2 + 1] ; 
	}else{
		NPop[it*2]   = Pop[Female[blockIdx.x]*nvars + (threadIdx.x - nvars/2 )*2    ] ;
		NPop[it*2+1] = Pop[  Male[blockIdx.x]*nvars + (threadIdx.x - nvars/2 )*2 + 1] ;
	}	
}

__global__ void CrossoverSingle_s(int *NPop, int *Male, int *Female, int *Pop, int psize, int nvars, int it, int iblo, int ithr){
// se llama una vez por cada 2  individuos y genera el cruzamiento entre el padre y la madre
	//int it = blockDim.x * blockIdx.x + threadIdx.x ;
	//blockDim = cantidad de hilos por bloque, es una constante [nvbin*nvdec] 64
	//blockIdx = nombre del bloque [identifica a la pareja] de 1 a psize/2
	//threadIdx = enumeracion del hilo dentro del bloque [cada variable de un ind.] de 1 a 64
	
	if(ithr < nvars/2){
		NPop[it*2]   = Pop[  Male[iblo]*nvars + ithr*2    ] ;
		NPop[it*2+1] = Pop[Female[iblo]*nvars + ithr*2 + 1] ; 
	}else{
		NPop[it*2]   = Pop[Female[iblo]*nvars + (ithr - nvars/2 )*2    ] ;
		NPop[it*2+1] = Pop[  Male[iblo]*nvars + (ithr - nvars/2 )*2 + 1] ;
	}	
}

__global__ void Mutation(int *Pop, float mutp, int nvec, int nvars, curandState* globalState){
// Genera un vector con "n" valores aleatorios enteros entre 0 y psize	
	int it = blockDim.x * blockIdx.x + threadIdx.x;

	if (it < nvec) {
		float ran = generate(globalState, it)*(1.0) ;
		if(ran<mutp){
			int ivar = lroundf(generate(globalState, it)*(nvars-0.5)) ;
			if(Pop[it*nvars+ivar]==0){
				Pop[it*nvars+ivar] = 1 ;
			}else{
				Pop[it*nvars+ivar] = 0 ;
			}
		} 
	}
}

__global__ void Mutation_s(int *Pop, float mutp, int nvec, int nvars, curandState* globalState, int ivar){
// Genera un vector con "n" valores aleatorios enteros entre 0 y psize	
	int it = ivar;

	if (it < nvec) {
		float ran = generate(globalState, it)*(1.0) ;
		if(ran<mutp){
			int ivar = lroundf(generate(globalState, it)*(nvars-0.5)) ;
			if(Pop[it*nvars+ivar]==0){
				Pop[it*nvars+ivar] = 1 ;
			}else{
				Pop[it*nvars+ivar] = 0 ;
			}
		} 
	}
}

//__global__ void EqualityINTprueba(int *vec2, int *vec1, int ivec, int nvec) {
//    int it = blockDim.x * blockIdx.x + threadIdx.x;
//	//vec2[it] = it;
//	if (it < nvec) { vec2[it] = it+ivec ; }  
//}
//	
//__global__ void EqualityElite(int *vec, int nvec) {
//    int it = blockDim.x * blockIdx.x + threadIdx.x;
//	vec[it] = it+nvec ;
//}

// *************** HOST FUNCTIONS ****************************************************************************

//SortArray2(Sol, Pop, nvars2, psize);
//void SortArray2(vector<float> &Sol, int Pop[psize][nvars2], int nvars2, int psize){
//  indiv POP1[psize];
//  for (int j=0; j<psize; j++) 
//    {
//    POP1[j].Solu = Sol[j];
//    //      printf("POP1[%d].Solu = %g \n", j, POP1[j].Solu);
//    POP1[j].vars.resize(nvars2);
//      for (int k=0; k<nvars2; k++)
//	{
//        POP1[j].vars[k] = Pop[j][k];
//	}
//        POP1[j].index = j;
//	//        printf("POP1[%i].index = %d \n", j, POP1[j].index);
//    }
//    sort(POP1,POP1+psize,lessthan);
//  for (int l=0; l<psize; l++) 
//    {
//    Sol[l]= POP1[l].Solu;
//    //      printf("Sol[%d] = %g \n", l, Sol[l]);
//    //      POP1[j].vars.resize(nvars);
//    for (int m=0; m<nvars2; m++)
//      {
//      Pop[l][m]=POP1[l].vars[m];
//      }
//      //        POP1[j].index = j;
//    }
//  /*  for (int j=0; j<psize; j++) 
//    {
//    printf("POP1[%d].Solu = %g \n", j, POP1[j].Solu);
//    printf("POP1[%i].index = %d \n", j, POP1[j].index);
//    }*/
//}
//void SortArray(float Sol[psize], int Pop[psize][nvdec][nvbin], int nvdec, int nvbin, int psize){ 
//	int i,j;                //Variables contadoras del ciclo.
//	//int lista[Nelementos]={6,9,3,1}; //Declaracion e inicializacion de un arreglo de 4 elementos.
//	float tempS = 0.0 ;    //Variable temporal.
//	float tempP = 0.0 ;    //Variable temporal.
//	 
//	for (i=1; i<psize; i++){
//       for (j=0; j <= psize-2; j++){
//          if (Sol[j] < Sol[j+1]){     //de Mayor a Menor: < ; de Menor a Mayor: >
//            tempS = Sol[j] ;
//			Sol[j] = Sol[j+1];
//            Sol[j+1] = tempS;
//			for(int idec=0; idec<nvdec; idec++){
//				for(int ibin=0; ibin<nvbin; ibin++){
//					tempP                = Pop[j][idec][ibin]   ;
//					Pop[j][idec][ibin]   = Pop[j+1][idec][ibin] ;
//					Pop[j+1][idec][ibin] = tempP                ;
//				}
//			}
//          }
//       }
//	}
//}

//********************** MAIN **********************************************************************************
int main()
{	
	clock_t start = clock();  
    cudaError_t err = cudaSuccess; // Error code to check return values for CUDA calls
 
	int igen=0,ipop; 
	float BestSol=1.0, x=0.0, y=0.0  ;
	
// random number on device
    curandState* devStates;
    cudaMalloc ( &devStates, psize*nvars*sizeof( curandState ) );
	setup_rand <<< psize*nvars/32, 32 >>> ( devStates,unsigned(time(NULL)) ); // setup seeds
 
	size_t size = psize*nvars*sizeof(int);
    int *d_Pop = NULL;
    cudaMalloc((void **)&d_Pop, size);
	InitPop<<< psize/32, 32 >>>(d_Pop, nvars, devStates);
	//for(ipop=0; ipop<psize; ipop++){ InitPop_s<<< 1, 1 >>>(d_Pop, ipop, nvars, devStates);	}  // SECUENCIAL !!!

	//float media = 100000 ;	
// Calcula la media de la función
	//while(fabs(media) > 1000){
	//	igen++ ;
	//start = clock();  
	//InitPop<<< psize/32, 32 >>>(d_Pop, nvars, devStates);
	//for(ipop=0; ipop<psize; ipop++){ InitPop_s<<< 1, 1 >>>(d_Pop, ipop, nvars, devStates);	}  // SECUENCIAL !!!		
	//	float *d_Sol = NULL;
	//	float *h_Sol = (float *)malloc(psize*sizeof(float));
	//	cudaMalloc((void **)&d_Sol, psize*sizeof(float));
	//	Func<<<psize/32, 32>>>(d_Sol, d_Pop, nvbin, nvdec) ;	
	//	float *h_Sum = (float *)malloc(sizeof(float)) ; 
	//	float *d_Sum = 0 ;
	//	cudaMalloc((void **)&d_Sum, sizeof(float)); 
	//	*h_Sum= 0 ;
	//	cudaMemcpy(d_Sum, h_Sum, sizeof(float), cudaMemcpyHostToDevice);
	//	CumSumVec<<<psize/32, 32>>>(d_Sum, d_Sol, psize);
	//	cudaMemcpy(h_Sum, d_Sum, sizeof(float), cudaMemcpyDeviceToHost);
	//	media = h_Sum[0]/psize ;
	//} ;
// fin de la media de la función


	
	err = cudaFree(devStates);
	if (err != cudaSuccess) {fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
				
// Bucle de las generaciones  ************************************************************
	while(BestSol > tol){
	//while(igen < ngen){
		if(igen > ngen+1){ exit(EXIT_FAILURE); }
		igen++ ;

		size = psize*sizeof(float); 
		float *d_Sol = NULL;
		float *h_Sol = (float *)malloc(size);
		cudaMalloc((void **)&d_Sol, size);

// Evaluacion funcion objetivo	
		Func<<<psize/32, 32>>>(d_Sol, d_Pop, nvbin, nvdec) ;		
		//for(ipop=0; ipop<psize; ipop++){ Func_s<<< 1, 1 >>>(d_Sol, d_Pop, ipop, nvbin, nvdec);	} // SECUENCIAL !!!
		cudaMemcpy(h_Sol, d_Sol, size, cudaMemcpyDeviceToHost);		

// Ordena los individuos segun el valor de la solucion
		indiv Pop[psize];
		for (int j=0; j<psize; j++) { Pop[j].Sol = h_Sol[j] ; Pop[j].Ind = j ; }
		sort(Pop, Pop+psize, lessthan);
		BestSol = Pop[psize-1].Sol ;

		err = cudaFree(d_Sol);
		if (err != cudaSuccess) {fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
		
		int h_Ind[psize] ;
		for(int i=0;i<psize;i++) { h_Ind[i] = Pop[i].Ind ;  }
		int *d_NPop = NULL;
		int *d_Pind = NULL;
		cudaMalloc((void **)&d_Pind, psize*sizeof(int)) ;
		cudaMalloc((void **)&d_NPop, psize*nvars*sizeof(int)) ;

		EqualityINT<<<psize, nvars>>>(d_NPop, d_Pop, 0, psize*nvars) ; // hace una copia de d_Pop
		//for(ipop=0; ipop<psize; ipop++){ EqualityINT_s<<< 1, 1 >>>(d_NPop, d_Pop, 0, ipop, psize*nvars);	} // SECUENCIAL !!!
				
		cudaMemcpy(d_Pind, h_Ind, psize*sizeof(int), cudaMemcpyHostToDevice);	
		Rearrange<<<psize, nvars>>>(d_Pop, d_NPop, d_Pind, nvars, psize*nvars) ; // ordena d_Pop
// INICIO SECUENCIAL !!!
		//int it=0;
		//for(int iblo=0; iblo<psize; iblo++){ 
		//	for(int ithr=0; ithr<nvars; ithr++){ 
		//		it++ ;
		//		Rearrange_s<<< 1, 1 >>>(d_Pop, d_NPop, d_Pind, nvars, psize*nvars, it, iblo, ithr);	
		//	}
		//} 
// FIN SECUENCIAL !!!

		err = cudaFree(d_Pind);
		if (err != cudaSuccess) {fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
	    err = cudaFree(d_NPop);
		if (err != cudaSuccess) {fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
		
		x=0.0 ; y=0.0 ;
		//XY(x, y, Best, nvdec, nvbin) ;				
		printf("igen %i ,Minimo %f ,X: %f ,Y: %f \n",igen,BestSol,x,y);
	
// Elige la población ELITE
		size = nelit*nvars*sizeof(int); 
		int *d_Eli = NULL ;
		cudaMalloc((void **)&d_Eli, size) ;

		for(ipop=psize-nelit; ipop<psize; ipop++){
			EqualityINT<<<1, nvars>>>(d_Eli, d_Pop, ipop*nvars, nvars) ;
			//for(int ivar=0; ivar<nvars; ivar++){ EqualityINT_s<<< 1, 1 >>>(d_Eli, d_Pop, ipop*nvars, ivar, nvars);	} // SECUENCIAL !!!
		}

// SELECCION (Simple Roulet) 
		int ngroup = psize ;	// cantidad de grupos (de cada grupo sale un individuo)
		int gsize = 3 ;			// 5 tamaño de cada grupo (nro. de ind. en cada grupo)
		curandState* devStates;
		cudaMalloc ( &devStates, gsize*ngroup*sizeof( curandState ) );
		setup_rand <<< ngroup, gsize >>> ( devStates,unsigned(time(NULL)) ); 

		size = gsize*ngroup*sizeof(int); 
		int *d_Tiro = NULL ;
		cudaMalloc((void **)&d_Tiro, size) ;
		RandomINT<<< ngroup, gsize >>>(d_Tiro, gsize*ngroup, psize, devStates) ;
		//for(int ivar=0; ivar<ngroup*gsize; ivar++){ RandomINT_s<<< 1, 1 >>>(d_Tiro, gsize*ngroup, psize, ivar, devStates);	} // SECUENCIAL !!!

		err = cudaFree(devStates);
		if (err != cudaSuccess) {fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
				
		size = psize*sizeof(int); 
		int *d_Sel = NULL ;
		cudaMalloc((void **)&d_Sel, size) ;	
		
		GroupSelection<<< psize/32, 32 >>>(d_Sel, d_Tiro, gsize, ngroup) ;
		//for(int ivar=0; ivar<psize; ivar++){ GroupSelection_s<<< 1, 1 >>>(d_Sel, d_Tiro, gsize, ngroup, ivar);	} // SECUENCIAL !!!
		
		err = cudaFree(d_Tiro);
		if (err != cudaSuccess) {fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }

// CRUZAMIENTO (Crossover Simple)
		size = (psize/2)*sizeof(int) ; 
		int *d_Male = NULL ;
		int *d_Female = NULL ;
		cudaMalloc((void **)&d_Male, size) ;
		cudaMalloc((void **)&d_Female, size) ;

		EqualityINT<<< psize/2/32, 32>>>(d_Male, d_Sel, 0, psize/2) ;
		EqualityINT<<< psize/2/32, 32>>>(d_Female, d_Sel, psize/2, psize/2) ;
// INICIO SECUENCIAL !!!
		//for(int ivar=0; ivar<psize/2; ivar++){ 
		//	EqualityINT_s<<< 1, 1 >>>(d_Male, d_Sel, 0, ivar, psize/2);
		//	EqualityINT_s<<< 1, 1 >>>(d_Female, d_Sel, psize/2, ivar, psize/2);
		//} 
// FIN SECUENCIAL !!!		
		
	    err = cudaFree(d_Sel);
		if (err != cudaSuccess) {fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
						
		size = psize*nvars*sizeof(int);
		cudaMalloc((void **)&d_NPop, size) ;
		
		CrossoverSingle<<<psize/2, nvars>>>(d_NPop, d_Male, d_Female, d_Pop, psize, nvars) ;
// INICIO SECUENCIAL !!!
		//int itt=0;
		//for(int iblo=0; iblo<psize/2; iblo++){ 
		//	for(int ithr=0; ithr<nvars; ithr++){ 
		//		itt++ ;
		//		CrossoverSingle_s<<< 1, 1 >>>(d_NPop, d_Male, d_Female, d_Pop, psize, nvars, itt, iblo, ithr);	
		//	}
		//} 
// FIN SECUENCIAL !!!
		
		err = cudaFree(d_Male);
		if (err != cudaSuccess) {fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
	    err = cudaFree(d_Female);
		if (err != cudaSuccess) {fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }

		EqualityINT<<<psize, nvbin*nvdec>>>(d_Pop, d_NPop, 0, psize*nvars) ;  
		//for(int ivar=0; ivar<psize*nvbin*nvdec; ivar++){ EqualityINT_s<<< 1, 1 >>>(d_Pop, d_NPop, 0, ivar, psize*nvars); } // SECUENCIAL !!!

	    err = cudaFree(d_NPop);
		if (err != cudaSuccess) {fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
		
// MUTACION (Step mutation)
		cudaMalloc ( &devStates, psize*sizeof( curandState ) );
		setup_rand <<< psize/32, 32 >>> ( devStates,unsigned(time(NULL)) ); 
		Mutation<<< psize/32, 32 >>>(d_Pop, mutp, psize, nvars, devStates) ;
		//for(int ivar=0; ivar<psize; ivar++){ Mutation_s<<< 1, 1 >>>(d_Pop, mutp, psize, nvars, devStates, ivar); } // SECUENCIAL !!!

	    err = cudaFree(devStates);
		if (err != cudaSuccess) {fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
				
// Copia la población ELITE	
		for(ipop=psize-nelit; ipop<psize; ipop++){
			EqualityINT<<<1, nvars>>>(d_Pop, d_Eli, ipop*nvars, nvars) ;
			//for(int ivar=0; ivar<nvars; ivar++){ EqualityINT_s<<< 1, 1 >>>(d_Pop, d_Eli, ipop*nvars, ivar, nvars);	} // SECUENCIAL !!!
		}

		err = cudaFree(d_Eli);
		if (err != cudaSuccess) {fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
		
	}

	// Free device global memory
    err = cudaFree(d_Pop);
	if (err != cudaSuccess) {fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
	
	printf("Poblacion: %i, Generaciones: %i, Tiempo transcurrido: %f",psize , igen, ((double)clock() - start) / CLOCKS_PER_SEC) ;
    printf("Done\n");
}


