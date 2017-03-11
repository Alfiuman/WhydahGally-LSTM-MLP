#pragma once

#define PRINT(x) std::cout << x

//OPTIONS
#define DEBUG	1
#define CUDA	1

//DEFINITIONS
//Algorithm
#define LSTM	1
#define MLP		2
#define MLPFst	3

//Task
#define TRAIN		1
#define TEST		2
#define CLASSIFY	3

//History length
#define TOTAL_HISTORY	0

//Loss function
#define LOSSFUNCTSIMPLE			0
#define LOSSFUNCTLOG			1
#define LOSSFUNCTLOGPOW3		2
#define LOSSFUNCTPOW3			3
#define LOSSFUNCTPOW3PLOGPOW3	4

//Random distribution
#define UNIFORM_DISTRIBUTION	0
#define NORMAL_DISTRIBUTION		1	

//CUDA block size
#define BLOCK_SIZE 16

//CPU or GPU
#define CPU		0
#define GLOBAL	1
#define SHARED	2

