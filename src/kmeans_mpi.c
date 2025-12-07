/**
 * K-Means 1D - Versão MPI (Memória Distribuída)
 * Projeto PCD - Programação Concorrente e Distribuída
 * 
 * Este arquivo implementa o algoritmo K-Means paralelizado com MPI.
 * Os dados são distribuídos entre os processos.
 * Cada processo calcula assignment para sua porção dos dados.
 * Centróides são sincronizados via MPI_Allreduce.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <mpi.h>

#define MAX_ITER 1000
#define EPS 1e-6

// Estrutura para armazenar os resultados
typedef struct {
    int iterations;
    double sse;
    double time_ms;
    int num_procs;
} KMeansResult;

// Função para ler CSV (1 coluna, sem cabeçalho) - apenas processo 0
double* read_csv(const char* filename, int* count) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Erro ao abrir arquivo: %s\n", filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
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
        return;
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
        return;
    }
    for (int i = 0; i < n; i++) {
        fprintf(file, "%.6f\n", data[i]);
    }
    fclose(file);
}

// Calcula distância Euclidiana 1D (ao quadrado)
double distance_squared(double x, double c) {
    double diff = x - c;
    return diff * diff;
}

/**
 * Algoritmo K-Means com MPI
 * 
 * Estratégia de distribuição:
 * 1. Processo 0 lê todos os dados
 * 2. Dados são distribuídos entre processos via MPI_Scatterv
 * 3. Cada processo calcula assignment local
 * 4. SSE e somas são combinadas via MPI_Allreduce
 * 5. Centróides são atualizados em todos os processos
 * 6. Resultados são coletados via MPI_Gatherv
 */
KMeansResult kmeans_mpi(const char* data_file, const char* centroids_file,
                        int max_iter, double eps) {
    KMeansResult result;
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int N = 0, K = 0;
    double* X_all = NULL;       // Todos os dados (apenas rank 0)
    double* C = NULL;           // Centróides (todos os processos)
    int* assign_all = NULL;     // Todas as atribuições (apenas rank 0)
    
    // Arrays locais
    double* X_local = NULL;     // Porção local dos dados
    int* assign_local = NULL;   // Atribuições locais
    int local_N = 0;            // Quantidade local de pontos
    
    // Arrays para distribuição de dados
    int* sendcounts = NULL;
    int* displs = NULL;
    
    double start_time = MPI_Wtime();
    
    // ========================================
    // PASSO 1: Processo 0 lê os dados
    // ========================================
    if (rank == 0) {
        X_all = read_csv(data_file, &N);
        C = read_csv(centroids_file, &K);
        assign_all = (int*)malloc(N * sizeof(int));
        
        printf("===========================================\n");
        printf("  K-Means 1D - Versão MPI (Distribuído)\n");
        printf("===========================================\n\n");
        printf("Pontos (N): %d\n", N);
        printf("Clusters (K): %d\n", K);
        printf("Processos MPI: %d\n", size);
        printf("Max iterações: %d\n", MAX_ITER);
        printf("Epsilon (eps): %.2e\n\n", EPS);
    }
    
    // Broadcast de N e K para todos os processos
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Alocar centróides em todos os processos
    if (rank != 0) {
        C = (double*)malloc(K * sizeof(double));
    }
    MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // ========================================
    // PASSO 2: Calcular distribuição de dados
    // ========================================
    sendcounts = (int*)malloc(size * sizeof(int));
    displs = (int*)malloc(size * sizeof(int));
    
    // Dividir dados entre processos
    int base_count = N / size;
    int remainder = N % size;
    
    int offset = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = base_count + (i < remainder ? 1 : 0);
        displs[i] = offset;
        offset += sendcounts[i];
    }
    
    local_N = sendcounts[rank];
    X_local = (double*)malloc(local_N * sizeof(double));
    assign_local = (int*)malloc(local_N * sizeof(int));
    
    // Distribuir dados
    MPI_Scatterv(X_all, sendcounts, displs, MPI_DOUBLE,
                 X_local, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // ========================================
    // PASSO 3: Loop principal do K-Means
    // ========================================
    double* local_sum = (double*)malloc(K * sizeof(double));
    int* local_count = (int*)malloc(K * sizeof(int));
    double* global_sum = (double*)malloc(K * sizeof(double));
    int* global_count = (int*)malloc(K * sizeof(int));
    
    double prev_sse = DBL_MAX;
    double sse = 0.0;
    int iter;
    
    for (iter = 0; iter < max_iter; iter++) {
        double local_sse = 0.0;
        
        // ========================================
        // PASSO 3a: ASSIGNMENT (local)
        // ========================================
        for (int i = 0; i < local_N; i++) {
            double min_dist = DBL_MAX;
            int best_cluster = 0;
            
            for (int c = 0; c < K; c++) {
                double dist = distance_squared(X_local[i], C[c]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            
            assign_local[i] = best_cluster;
            local_sse += min_dist;
        }
        
        // Redução global do SSE
        MPI_Allreduce(&local_sse, &sse, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // Verificar convergência
        if (fabs(prev_sse - sse) < eps) {
            iter++;
            break;
        }
        prev_sse = sse;
        
        // ========================================
        // PASSO 3b: UPDATE (distribuído)
        // ========================================
        
        // Inicializar acumuladores locais
        for (int c = 0; c < K; c++) {
            local_sum[c] = 0.0;
            local_count[c] = 0;
        }
        
        // Acumular somas locais
        for (int i = 0; i < local_N; i++) {
            int c = assign_local[i];
            local_sum[c] += X_local[i];
            local_count[c]++;
        }
        
        // Redução global de somas e contadores
        MPI_Allreduce(local_sum, global_sum, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_count, global_count, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        // Atualizar centróides (todos os processos)
        for (int c = 0; c < K; c++) {
            if (global_count[c] > 0) {
                C[c] = global_sum[c] / global_count[c];
            } else {
                // Cluster vazio: precisamos do X[0] global
                double x0;
                if (rank == 0) {
                    x0 = X_all[0];
                }
                MPI_Bcast(&x0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                C[c] = x0;
            }
        }
    }
    
    double end_time = MPI_Wtime();
    
    // ========================================
    // PASSO 4: Coletar resultados
    // ========================================
    MPI_Gatherv(assign_local, local_N, MPI_INT,
                assign_all, sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    
    // ========================================
    // PASSO 5: Processo 0 salva resultados
    // ========================================
    if (rank == 0) {
        result.iterations = iter;
        result.sse = sse;
        result.time_ms = (end_time - start_time) * 1000.0;
        result.num_procs = size;
        
        printf("-------------------------------------------\n");
        printf("RESULTADOS:\n");
        printf("-------------------------------------------\n");
        printf("Iterações: %d\n", result.iterations);
        printf("SSE final: %.6f\n", result.sse);
        printf("Tempo total: %.3f ms\n", result.time_ms);
        printf("Processos MPI: %d\n", result.num_procs);
        printf("-------------------------------------------\n\n");
        
        // Salvar arquivos de saída
        save_csv_int("assign_mpi.csv", assign_all, N);
        save_csv_double("centroids_mpi.csv", C, K);
        
        printf("Arquivos gerados:\n");
        printf("  - assign_mpi.csv (atribuição de clusters)\n");
        printf("  - centroids_mpi.csv (centróides finais)\n\n");
        
        // Exibir centróides finais
        printf("Centróides finais:\n");
        for (int c = 0; c < K; c++) {
            printf("  C[%d] = %.6f\n", c, C[c]);
        }
        
        free(X_all);
        free(assign_all);
    }
    
    // Liberar memória
    free(X_local);
    free(assign_local);
    free(C);
    free(local_sum);
    free(local_count);
    free(global_sum);
    free(global_count);
    free(sendcounts);
    free(displs);
    
    return result;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    const char* data_file = "dados.csv";
    const char* centroids_file = "centroides_iniciais.csv";
    
    if (argc >= 3) {
        data_file = argv[1];
        centroids_file = argv[2];
    }
    
    kmeans_mpi(data_file, centroids_file, MAX_ITER, EPS);
    
    MPI_Finalize();
    return 0;
}
