// This program executes a typical convolutional layer in regular CNNs.Neuron sparsity(zero ratio) is 50% and Weight sparsity is 70%.
#include <iostream>
#include "CNNConvLayer.h"
using namespace std;

// This is the CPU version, please don't modify it
void convLayerCPU()
{
	// declarations for bunch of indexing parameters
	int fn, sli, fmy, fmx, y, x;
	int ifmy, ifmx, ofmy, ofmx;
	int filtIdx, inNeuIdx, outNeuIdx, outIdx;
	int filtVol  = FMDEPTH  * FILTSIZE * FILTSIZE;
	int fmArea   = FMSIZE   * FMSIZE;
	int filtArea = FILTSIZE * FILTSIZE;
	int outArea  = FMSIZE/3 * FMSIZE/3;
	int sum;

	// Convolution
	for(fn = 0; fn < FILTNUM; fn++){					//iterate through each filters
		for(fmy = 0; fmy < FMSIZE; fmy += STRIDE){		//Stride through
			for(fmx = 0; fmx < FMSIZE; fmx += STRIDE){	//Stride through
				sum = 0;
				for(sli = 0; sli < FMDEPTH; sli++){		//Iterate through depth
					//Convolution
					for(y = 0; y < FILTSIZE; y++){
						for(x = 0; x < FILTSIZE; x++){
							ifmy = fmy - FILTSIZE / 2 + y;	
							ifmx = fmx - FILTSIZE / 2 + x;
							filtIdx = fn*filtVol + sli*filtArea + y*FILTSIZE + x;
							inNeuIdx = sli*fmArea + ifmy*FMSIZE + ifmx;
							if(ifmy >= 0 && ifmy < FMSIZE && ifmx >= 0 && ifmx < FMSIZE)
								sum += filt[filtIdx] * inNeu[inNeuIdx];
							//"filt" is a giant array that stores all of the parameters of all the filters
							//size = 307200
							//inNeu size = 69984
							//What's tricky here is that filter weighting and input neurons are all int
						}
					}
				}
				// Activation - ReLU <- Don't pronounce it wrong
				outNeuIdx = fn*fmArea + fmy*FMSIZE + fmx;
				if(sum <= 0)
					outNeu[outNeuIdx] = 0;
				else
					outNeu[outNeuIdx] = sum;
			}
		}
	}

	// Max Pooling with Window Size 3x3 and stride 3
	int max, tmpVal;
	for(sli = 0; sli < FILTNUM; sli++){
		for(fmy = 0; fmy < FMSIZE/3 ; fmy += 1){
			for(fmx = 0; fmx < FMSIZE/3 ; fmx += 1){
				outNeuIdx = sli*fmArea + fmy*3*FMSIZE + fmx*3;
				max = outNeu[outNeuIdx];
				for(y = 0; y < 3; y++){
					for(x = 0; x < 3; x++){
						ofmy = fmy*3 + y;
						ofmx = fmx*3 + x;
						outNeuIdx = sli*fmArea + ofmy*FMSIZE + ofmx;
						tmpVal = outNeu[outNeuIdx];	
						if(tmpVal > max)
							max = tmpVal;
					}
				}
				outIdx = sli*outArea + fmy*FMSIZE/3 + fmx;
				outCPU[outIdx] = max;
				
			}
		}
	}
}

/***	Implement your CUDA Kernel here	***/
__global__
void convLayerGPU(int* filt_GPU, int* inNeu_GPU, int* out_GPU_kernel, int* out_Neu_kernel)
{
	
	// declarations for bunch of indexing parameters
	int fn, sli, fmy, fmx, y, x;
	int ifmy, ifmx, ofmy, ofmx;
	int filtIdx, inNeuIdx, outNeuIdx, outIdx;
	int filtVol  = FMDEPTH  * FILTSIZE * FILTSIZE;
	int fmArea   = FMSIZE   * FMSIZE;
	int filtArea = FILTSIZE * FILTSIZE;
	int outArea  = FMSIZE/3 * FMSIZE/3;
	int sum;

	int i = blockIdx.x*blockDim.x + threadIdx.x;


	if(i < FILTNUM*FMSIZE*FMSIZE){
		//if(i<1000)
		//	printf("Hi ^^%d\n", i);
		sum = 0;
		fn = i/FMSIZE/FMSIZE;
		for(sli = 0; sli < FMDEPTH; sli++){
			for(y = 0; y < FILTSIZE; y++){
				for(x = 0; x < FILTSIZE; x++){
					fmy = (i/FMSIZE)%FMSIZE;//Checked
					fmx = i%FMSIZE;	//Checked						
					ifmy = fmy - FILTSIZE / 2 + y;	
					ifmx = fmx - FILTSIZE / 2 + x;
					filtIdx = fn*filtVol + sli*filtArea + y*FILTSIZE + x;
					inNeuIdx = sli*fmArea + ifmy*FMSIZE + ifmx;
					if(ifmy >= 0 && ifmy < FMSIZE && ifmx >= 0 && ifmx < FMSIZE)
						sum += filt_GPU[filtIdx] * inNeu_GPU[inNeuIdx];
				}
			}
		}
		
		outNeuIdx = fn*fmArea + fmy*FMSIZE + fmx;
		if(sum <= 0)
			out_Neu_kernel[outNeuIdx] = 0;
		else
			out_Neu_kernel[outNeuIdx] = sum;
	}
	__syncthreads();
		// Max Pooling with Window Size 3x3 and stride 3
	/*
	if(i == 0){
		int max, tmpVal;
		for(sli = 0; sli < FILTNUM; sli++){
			for(fmy = 0; fmy < FMSIZE/3 ; fmy += 1){
				for(fmx = 0; fmx < FMSIZE/3 ; fmx += 1){
					outNeuIdx = sli*fmArea + fmy*3*FMSIZE + fmx*3;
					max = out_Neu_kernel[outNeuIdx];
					for(y = 0; y < 3; y++){
						for(x = 0; x < 3; x++){
							ofmy = fmy*3 + y;
							ofmx = fmx*3 + x;
							outNeuIdx = sli*fmArea + ofmy*FMSIZE + ofmx;
							tmpVal = out_Neu_kernel[outNeuIdx];	
							if(tmpVal > max)
								max = tmpVal;
						}
					}
					outIdx = sli*outArea + fmy*FMSIZE/3 + fmx;
					out_GPU_kernel[outIdx] = max;
				}
			}
		}
	}
	*/


	if(i < FILTNUM * (FMSIZE/3) * (FMSIZE/3)){
		sli = i/(FMSIZE/3)/(FMSIZE/3);
		fmy = (i/(FMSIZE/3))%(FMSIZE/3);
		fmx = i%(FMSIZE/3);
		outNeuIdx = sli*fmArea + fmy*3*FMSIZE + fmx*3;
		max = out_Neu_kernel[outNeuIdx];
		for(y = 0; y < 3; y++){
			for(x = 0; x < 3; x++){
				ofmy = fmy*3 + y;
				ofmx = fmx*3 + x;
				outNeuIdx = sli*fmArea + ofmy*FMSIZE + ofmx;
				tmpVal = out_Neu_kernel[outNeuIdx];	
				if(tmpVal > max)
					max = tmpVal;
			}
		}
		outIdx = sli*outArea + fmy*FMSIZE/3 + fmx;
		out_GPU_kernel[outIdx] = max;
	}
}
/***	Implement your CUDA Kernel here	***/

int main()
{
	//variables setting and loading input data
	timespec time_begin, time_end; 
	int convLayerCPUExecTime, convLayerGPUExecTime;
	init();
	
	/******** Added ********/
	int* filt_GPU;
	int* inNeu_GPU;
	int* out_GPU_kernel;
	int* out_Neu_kernel;

	cudaMalloc(&filt_GPU, FILTSIZE*FILTSIZE*FMDEPTH*FILTNUM*sizeof(int)); 
	cudaMalloc(&inNeu_GPU, FMSIZE*FMSIZE*FMDEPTH*sizeof(int));
	cudaMalloc(&out_GPU_kernel, FILTNUM * FMSIZE/3 * FMSIZE/3*sizeof(int));
	cudaMalloc(&out_Neu_kernel, FILTNUM * FMSIZE * FMSIZE*sizeof(int));

	cudaMemcpy(filt_GPU, filt, FILTSIZE*FILTSIZE*FMDEPTH*FILTNUM*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(inNeu_GPU, inNeu, FMSIZE*FMSIZE*FMDEPTH*sizeof(int), cudaMemcpyHostToDevice);

	/******** Added ********/


	//Convolution by CPU                                                
	clock_gettime(CLOCK_REALTIME, &time_begin);
	convLayerCPU();
	clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerCPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "CPU time for executing a typical convolutional layer = "  <<  ((float)convLayerCPUExecTime)/1000 << "ms" << endl;

  
	//Convolution by GPU   
	clock_gettime(CLOCK_REALTIME, &time_begin);

	/***	Lunch your CUDA Kernel here	***/
	convLayerGPU<<<(FILTNUM*FMSIZE*FMSIZE+1023)/1024,1024>>>(filt_GPU, inNeu_GPU, out_GPU_kernel,  out_Neu_kernel); // Lunch the kernel
	cudaDeviceSynchronize(); // Do synchronization before clock_gettime()
	cudaMemcpy(outGPU, out_GPU_kernel, FILTNUM * FMSIZE/3 * FMSIZE/3*sizeof(int), cudaMemcpyDeviceToHost);
	/***	Lunch your CUDA Kernel here	***/

	clock_gettime(CLOCK_REALTIME, &time_end);
	convLayerGPUExecTime = timespec_diff_us(time_begin, time_end);
	cout << "GPU time for executing a typical convolutional layer = "  << ((float)convLayerGPUExecTime)/1000 << "ms" << endl;

	
	//check the anser from CPU and from GPU
	if(checker()){
		cout << "Congratulations! You pass the check." << endl;
		cout << "Speedup: " << (float)convLayerCPUExecTime / convLayerGPUExecTime << endl;
	}
	else
		cout << "Hummm there's something wrong" << endl;

	/******** Added ********/
	cudaFree(filt_GPU);
	cudaFree(inNeu_GPU);
	/******** Added ********/




	//release memory space
	ending();
	
	return 0;
}
