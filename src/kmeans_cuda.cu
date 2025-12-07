/**
 * K-Means 1D - Versão CUDA (GPU)
 * Projeto PCD - Programação Concorrente e Distribuída
 * 
 * Este arquivo implementa o algoritmo K-Means paralelizado com CUDA.
 * Cada thread CUDA processa um ponto no passo de Assignment.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

#define MAX_ITER 1000
#define EPS 1e-6
#define BLOCK_SIZE 256

// Estrutura para armazenar os resultados
typedef struct {
    int iterations;
    double sse;
    double time_ms;
} KMeansResult;

// Macro para verificar erros CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================
// KERNELS CUDA
// ============================================

/**
 * Kernel de Assignment: cada thread processa um ponto
 * Encontra o centróide mais próximo e acumula SSE parcial
 */
__global__ void assignment_kernel(const double* X, const double* C, 
                                   int* assign, double* partial_sse,
                                   int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        double min_dist = DBL_MAX;
        int best_cluster = 0;
        
        // Encontrar centróide mais próximo
        for (int c = 0; c < K; c++) {
            double diff = X[i] - C[c];
            double dist = diff * diff;
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = c;
            }
        }
        
        assign[i] = best_cluster;
        partial_sse[i] = min_dist;
    }
}

/**
 * Kernel de redução para SSE
 * Soma todos os valores de SSE parciais
 */
__global__ void reduce_sse_kernel(double* partial_sse, double* result, int N) {
    extern __shared__ double sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Carregar dados para memória compartilhada
    sdata[tid] = (i < N) ? partial_sse[i] : 0.0;
    __syncthreads();
    
    // Redução na memória compartilhada
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Escrever resultado deste bloco
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

/**
 * Kernel para acumular somas e contadores por cluster
 * Usa atomicAdd para evitar race conditions
 */
__global__ void accumulate_kernel(const double* X, const int* assign,
                                   double* sum, int* count, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N) {
        int c = assign[i];
        atomicAdd(&sum[c], X[i]);
        atomicAdd(&count[c], 1);
    }
}

/**
 * Kernel para atualizar centróides
 * Calcula a média de cada cluster
 */
__global__ void update_centroids_kernel(double* C, const double* sum, 
                                         const int* count, const double* X,
                                         int K) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c < K) {
        if (count[c] > 0) {
            C[c] = sum[c] / count[c];
        } else {
            // Cluster vazio: copiar X[0]
            C[c] = X[0];
        }
    }
}

// ============================================
// FUNÇÕES HOST
// ============================================

double* read_csv(const char* filename, int* count) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Erro ao abrir arquivo: %s\n", filename);
        exit(1);
    }
    
    int n = 0;
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        n++;
    }
    
    double* data = (double*)malloc(n * sizeof(double));
    
    rewind(file);
    int i = 0;
    while (fgets(line, sizeof(line), file) && i < n) {
        data[i++] = atof(line);
    }
    
    fclose(file);
    *count = n;
    return data;
}

void save_csv_int(const char* filename, int* data, int n) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Erro ao criar arquivo: %s\n", filename);
        exit(1);
    }
    for (int i = 0; i < n; i++) {
        fprintf(file, "%d\n", data[i]);
    }
    fclose(file);
}

void save_csv_double(const char* filename, double* data, int n) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Erro ao criar arquivo: %s\n", filename);
        exit(1);
    }
    for (int i = 0; i < n; i++) {
        fprintf(file, "%.6f\n", data[i]);
    }
    fclose(file);
}

/**
 * Algoritmo K-Means com CUDA
 */
KMeansResult kmeans_cuda(double* h_X, int N, double* h_C, int K, int* h_assign, 
                         int max_iter, double eps) {
    KMeansResult result;
    
    // Configuração de blocos e threads
    int blocks_N = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_K = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Alocar memória na GPU
    double *d_X, *d_C, *d_partial_sse, *d_sse_result, *d_sum;
    int *d_assign, *d_count;
    
    CUDA_CHECK(cudaMalloc(&d_X, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_assign, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_partial_sse, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sse_result, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sum, K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_count, K * sizeof(int)));
    
    // Copiar dados para GPU
    CUDA_CHECK(cudaMemcpy(d_X, h_X, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, K * sizeof(double), cudaMemcpyHostToDevice));
    
    // Criar eventos para medir tempo
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    double prev_sse = DBL_MAX;
    double sse = 0.0;
    int iter;
    
    for (iter = 0; iter < max_iter; iter++) {
        // ========================================
        // PASSO 1: ASSIGNMENT (GPU)
        // ========================================
        assignment_kernel<<<blocks_N, BLOCK_SIZE>>>(d_X, d_C, d_assign, 
                                                     d_partial_sse, N, K);
        CUDA_CHECK(cudaGetLastError());
        
        // Redução para calcular SSE total
        CUDA_CHECK(cudaMemset(d_sse_result, 0, sizeof(double)));
        reduce_sse_kernel<<<blocks_N, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(
            d_partial_sse, d_sse_result, N);
        CUDA_CHECK(cudaGetLastError());
        
        // Copiar SSE para host
        CUDA_CHECK(cudaMemcpy(&sse, d_sse_result, sizeof(double), cudaMemcpyDeviceToHost));
        
        // Verificar convergência
        if (fabs(prev_sse - sse) < eps) {
            iter++;
            break;
        }
        prev_sse = sse;
        
        // ========================================
        // PASSO 2: UPDATE (GPU)
        // ========================================
        
        // Zerar acumuladores
        CUDA_CHECK(cudaMemset(d_sum, 0, K * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_count, 0, K * sizeof(int)));
        
        // Acumular somas
        accumulate_kernel<<<blocks_N, BLOCK_SIZE>>>(d_X, d_assign, d_sum, d_count, N);
        CUDA_CHECK(cudaGetLastError());
        
        // Atualizar centróides
        update_centroids_kernel<<<blocks_K, BLOCK_SIZE>>>(d_C, d_sum, d_count, d_X, K);
        CUDA_CHECK(cudaGetLastError());
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // Copiar resultados de volta para host
    CUDA_CHECK(cudaMemcpy(h_assign, d_assign, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, K * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Liberar memória GPU
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_assign));
    CUDA_CHECK(cudaFree(d_partial_sse));
    CUDA_CHECK(cudaFree(d_sse_result));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_count));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    result.iterations = iter;
    result.sse = sse;
    result.time_ms = milliseconds;
    
    return result;
}

void print_gpu_info() {
    int device;
    cudaDeviceProp prop;
    
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Max Threads/Block: %d\n", prop.maxThreadsPerBlock);
    printf("Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
}

int main(int argc, char* argv[]) {
    const char* data_file = "dados.csv";
    const char* centroids_file = "centroides_iniciais.csv";
    
    if (argc >= 3) {
        data_file = argv[1];
        centroids_file = argv[2];
    }
    
    printf("===========================================\n");
    printf("  K-Means 1D - Versão CUDA (GPU)\n");
    printf("===========================================\n\n");
    
    // Exibir informações da GPU
    print_gpu_info();
    printf("\n");
    
    // Ler dados
    int N, K;
    double* X = read_csv(data_file, &N);
    double* C = read_csv(centroids_file, &K);
    
    printf("Pontos (N): %d\n", N);
    printf("Clusters (K): %d\n", K);
    printf("Block Size: %d\n", BLOCK_SIZE);
    printf("Max iterações: %d\n", MAX_ITER);
    printf("Epsilon (eps): %.2e\n\n", EPS);
    
    // Alocar array de atribuições
    int* assign = (int*)malloc(N * sizeof(int));
    
    // Executar K-Means
    KMeansResult result = kmeans_cuda(X, N, C, K, assign, MAX_ITER, EPS);
    
    // Exibir resultados
    printf("-------------------------------------------\n");
    printf("RESULTADOS:\n");
    printf("-------------------------------------------\n");
    printf("Iterações: %d\n", result.iterations);
    printf("SSE final: %.6f\n", result.sse);
    printf("Tempo total: %.3f ms\n", result.time_ms);
    printf("-------------------------------------------\n\n");
    
    // Salvar arquivos de saída
    save_csv_int("assign_cuda.csv", assign, N);
    save_csv_double("centroids_cuda.csv", C, K);
    
    printf("Arquivos gerados:\n");
    printf("  - assign_cuda.csv (atribuição de clusters)\n");
    printf("  - centroids_cuda.csv (centróides finais)\n\n");
    
    // Exibir centróides finais
    printf("Centróides finais:\n");
    for (int c = 0; c < K; c++) {
        printf("  C[%d] = %.6f\n", c, C[c]);
    }
    
    // Liberar memória
    free(X);
    free(C);
    free(assign);
    
    return 0;
}
