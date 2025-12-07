/**
 * K-Means 1D - Versão Naive (Sequencial)
 * Projeto PCD - Programação Concorrente e Distribuída
 * 
 * Este arquivo implementa o algoritmo K-Means básico sem paralelização.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define MAX_ITER 1000
#define EPS 1e-6

// Estrutura para armazenar os resultados
typedef struct {
    int iterations;
    double sse;
    double time_ms;
} KMeansResult;

// Função para ler CSV (1 coluna, sem cabeçalho)
double* read_csv(const char* filename, int* count) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Erro ao abrir arquivo: %s\n", filename);
        exit(1);
    }
    
    // Contar linhas
    int n = 0;
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        n++;
    }
    
    // Alocar memória
    double* data = (double*)malloc(n * sizeof(double));
    if (!data) {
        fprintf(stderr, "Erro de alocação de memória\n");
        exit(1);
    }
    
    // Ler valores
    rewind(file);
    int i = 0;
    while (fgets(line, sizeof(line), file) && i < n) {
        data[i++] = atof(line);
    }
    
    fclose(file);
    *count = n;
    return data;
}

// Função para salvar CSV
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

// Calcula distância Euclidiana 1D (ao quadrado)
double distance_squared(double x, double c) {
    double diff = x - c;
    return diff * diff;
}

/**
 * Algoritmo K-Means Naive (Sequencial)
 * 
 * @param X         Array de pontos (N valores)
 * @param N         Número de pontos
 * @param C         Array de centróides (K valores) - será modificado
 * @param K         Número de clusters
 * @param assign    Array de atribuições (N valores) - output
 * @param max_iter  Número máximo de iterações
 * @param eps       Tolerância para convergência do SSE
 * @return          Estrutura com resultados (iterações, SSE, tempo)
 */
KMeansResult kmeans_naive(double* X, int N, double* C, int K, int* assign, 
                          int max_iter, double eps) {
    KMeansResult result;
    
    clock_t start = clock();
    
    double prev_sse = DBL_MAX;
    double sse = 0.0;
    int iter;
    
    // Arrays auxiliares para cálculo dos novos centróides
    double* sum = (double*)malloc(K * sizeof(double));
    int* count = (int*)malloc(K * sizeof(int));
    
    for (iter = 0; iter < max_iter; iter++) {
        sse = 0.0;
        
        // ========================================
        // PASSO 1: ASSIGNMENT
        // Para cada ponto, encontrar o centróide mais próximo
        // ========================================
        for (int i = 0; i < N; i++) {
            double min_dist = DBL_MAX;
            int best_cluster = 0;
            
            for (int c = 0; c < K; c++) {
                double dist = distance_squared(X[i], C[c]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            
            assign[i] = best_cluster;
            sse += min_dist;  // Acumular SSE
        }
        
        // Verificar convergência
        if (fabs(prev_sse - sse) < eps) {
            iter++;  // Contar esta iteração
            break;
        }
        prev_sse = sse;
        
        // ========================================
        // PASSO 2: UPDATE
        // Recalcular centróides como média dos pontos atribuídos
        // ========================================
        
        // Inicializar somas e contadores
        for (int c = 0; c < K; c++) {
            sum[c] = 0.0;
            count[c] = 0;
        }
        
        // Acumular somas
        for (int i = 0; i < N; i++) {
            int c = assign[i];
            sum[c] += X[i];
            count[c]++;
        }
        
        // Calcular novas médias
        for (int c = 0; c < K; c++) {
            if (count[c] > 0) {
                C[c] = sum[c] / count[c];
            } else {
                // Cluster vazio: copiar X[0] (estratégia simples)
                C[c] = X[0];
            }
        }
    }
    
    clock_t end = clock();
    
    free(sum);
    free(count);
    
    result.iterations = iter;
    result.sse = sse;
    result.time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    
    return result;
}

int main(int argc, char* argv[]) {
    // Arquivos de entrada padrão
    const char* data_file = "dados.csv";
    const char* centroids_file = "centroides_iniciais.csv";
    
    // Permitir arquivos personalizados via argumentos
    if (argc >= 3) {
        data_file = argv[1];
        centroids_file = argv[2];
    }
    
    printf("===========================================\n");
    printf("  K-Means 1D - Versão NAIVE (Sequencial)\n");
    printf("===========================================\n\n");
    
    // Ler dados
    int N, K;
    double* X = read_csv(data_file, &N);
    double* C = read_csv(centroids_file, &K);
    
    printf("Pontos (N): %d\n", N);
    printf("Clusters (K): %d\n", K);
    printf("Max iterações: %d\n", MAX_ITER);
    printf("Epsilon (eps): %.2e\n\n", EPS);
    
    // Alocar array de atribuições
    int* assign = (int*)malloc(N * sizeof(int));
    
    // Executar K-Means
    KMeansResult result = kmeans_naive(X, N, C, K, assign, MAX_ITER, EPS);
    
    // Exibir resultados
    printf("-------------------------------------------\n");
    printf("RESULTADOS:\n");
    printf("-------------------------------------------\n");
    printf("Iterações: %d\n", result.iterations);
    printf("SSE final: %.6f\n", result.sse);
    printf("Tempo total: %.3f ms\n", result.time_ms);
    printf("-------------------------------------------\n\n");
    
    // Salvar arquivos de saída
    save_csv_int("assign.csv", assign, N);
    save_csv_double("centroids.csv", C, K);
    
    printf("Arquivos gerados:\n");
    printf("  - assign.csv (atribuição de clusters)\n");
    printf("  - centroids.csv (centróides finais)\n\n");
    
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
