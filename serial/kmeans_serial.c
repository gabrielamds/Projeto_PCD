/**
 * ============================================================================
 * K-MEANS 1D - VERSÃO SERIAL (SEQUENCIAL)
 * ============================================================================
 * Projeto PCD - Programação Concorrente e Distribuída
 * 
 * Esta é a versão baseline (referência) para comparação de desempenho.
 * Implementação sequencial sem nenhuma paralelização.
 * 
 * Métricas coletadas:
 *   - Tempo total de execução (ms)
 *   - SSE (Sum of Squared Errors) por iteração
 *   - Número de iterações até convergência
 *   - Throughput (pontos processados por segundo)
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

/* ============================================================================
 * PARÂMETROS PADRONIZADOS (iguais em todas as versões)
 * ============================================================================ */
#define MAX_ITER 100        // Número máximo de iterações
#define EPS 1e-6            // Tolerância para convergência do SSE
#define DEFAULT_N 100000    // Número padrão de pontos
#define DEFAULT_K 5         // Número padrão de clusters
#define SEED 42             // Semente para reprodutibilidade

/* ============================================================================
 * ESTRUTURAS DE DADOS
 * ============================================================================ */

// Estrutura para armazenar métricas de execução
typedef struct {
    int iterations;           // Número de iterações executadas
    double sse_final;         // SSE final
    double time_total_ms;     // Tempo total em milissegundos
    double time_assignment_ms;// Tempo gasto em assignment
    double time_update_ms;    // Tempo gasto em update
    double throughput;        // Pontos por segundo
    double* sse_history;      // SSE de cada iteração (para validação)
} KMeansMetrics;

/* ============================================================================
 * FUNÇÕES UTILITÁRIAS
 * ============================================================================ */

// Retorna tempo atual em segundos (alta precisão)
double get_time_sec() {
    struct timespec ts;
    #ifdef _WIN32
        return (double)clock() / CLOCKS_PER_SEC;
    #else
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return ts.tv_sec + ts.tv_nsec / 1e9;
    #endif
}

// Lê arquivo CSV (1 coluna, sem cabeçalho)
double* read_csv(const char* filename, int* count) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "ERRO: Não foi possível abrir '%s'\n", filename);
        exit(EXIT_FAILURE);
    }
    
    // Contar linhas
    int n = 0;
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (strlen(line) > 1) n++;  // Ignora linhas vazias
    }
    
    // Alocar memória
    double* data = (double*)malloc(n * sizeof(double));
    if (!data) {
        fprintf(stderr, "ERRO: Falha na alocação de memória (%d doubles)\n", n);
        exit(EXIT_FAILURE);
    }
    
    // Ler valores
    rewind(file);
    int i = 0;
    while (fgets(line, sizeof(line), file) && i < n) {
        if (strlen(line) > 1) {
            data[i++] = atof(line);
        }
    }
    
    fclose(file);
    *count = i;
    return data;
}

// Salva array de inteiros em CSV
void save_csv_int(const char* filename, int* data, int n) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "ERRO: Não foi possível criar '%s'\n", filename);
        return;
    }
    for (int i = 0; i < n; i++) {
        fprintf(file, "%d\n", data[i]);
    }
    fclose(file);
}

// Salva array de doubles em CSV
void save_csv_double(const char* filename, double* data, int n) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "ERRO: Não foi possível criar '%s'\n", filename);
        return;
    }
    for (int i = 0; i < n; i++) {
        fprintf(file, "%.10f\n", data[i]);
    }
    fclose(file);
}

// Salva histórico de SSE para análise de convergência
void save_sse_history(const char* filename, double* sse_history, int iterations) {
    FILE* file = fopen(filename, "w");
    if (!file) return;
    
    fprintf(file, "iteracao,sse\n");  // Cabeçalho CSV
    for (int i = 0; i < iterations; i++) {
        fprintf(file, "%d,%.10f\n", i + 1, sse_history[i]);
    }
    fclose(file);
}

/* ============================================================================
 * ALGORITMO K-MEANS SERIAL
 * ============================================================================ */

/**
 * Executa o algoritmo K-Means sequencial
 * 
 * @param X         Array de N pontos (valores 1D)
 * @param N         Número de pontos
 * @param C         Array de K centróides (entrada/saída)
 * @param K         Número de clusters
 * @param assign    Array de atribuições de cluster (saída)
 * @param max_iter  Número máximo de iterações
 * @param eps       Tolerância para convergência
 * @return          Estrutura com métricas de execução
 */
KMeansMetrics kmeans_serial(double* X, int N, double* C, int K, int* assign,
                            int max_iter, double eps) {
    KMeansMetrics metrics;
    metrics.sse_history = (double*)malloc(max_iter * sizeof(double));
    metrics.time_assignment_ms = 0;
    metrics.time_update_ms = 0;
    
    // Arrays auxiliares
    double* sum = (double*)malloc(K * sizeof(double));
    int* count = (int*)malloc(K * sizeof(int));
    
    double start_total = get_time_sec();
    double prev_sse = DBL_MAX;
    double sse = 0.0;
    int iter;
    
    for (iter = 0; iter < max_iter; iter++) {
        sse = 0.0;
        
        /* ================================================================
         * PASSO 1: ASSIGNMENT
         * Para cada ponto, encontrar o centróide mais próximo
         * Complexidade: O(N * K)
         * ================================================================ */
        double start_assign = get_time_sec();
        
        for (int i = 0; i < N; i++) {
            double min_dist = DBL_MAX;
            int best_cluster = 0;
            
            for (int c = 0; c < K; c++) {
                double diff = X[i] - C[c];
                double dist = diff * diff;  // Distância Euclidiana ao quadrado
                
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            
            assign[i] = best_cluster;
            sse += min_dist;
        }
        
        double end_assign = get_time_sec();
        metrics.time_assignment_ms += (end_assign - start_assign) * 1000.0;
        
        // Registrar SSE desta iteração
        metrics.sse_history[iter] = sse;
        
        // Verificar convergência
        if (fabs(prev_sse - sse) < eps) {
            iter++;
            break;
        }
        prev_sse = sse;
        
        /* ================================================================
         * PASSO 2: UPDATE
         * Recalcular centróides como média dos pontos atribuídos
         * Complexidade: O(N + K)
         * ================================================================ */
        double start_update = get_time_sec();
        
        // Inicializar acumuladores
        for (int c = 0; c < K; c++) {
            sum[c] = 0.0;
            count[c] = 0;
        }
        
        // Acumular somas e contagens
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
                // Cluster vazio: usar primeiro ponto (estratégia simples)
                C[c] = X[0];
            }
        }
        
        double end_update = get_time_sec();
        metrics.time_update_ms += (end_update - start_update) * 1000.0;
    }
    
    double end_total = get_time_sec();
    
    // Preencher métricas
    metrics.iterations = iter;
    metrics.sse_final = sse;
    metrics.time_total_ms = (end_total - start_total) * 1000.0;
    metrics.throughput = (double)(N * iter) / (metrics.time_total_ms / 1000.0);
    
    free(sum);
    free(count);
    
    return metrics;
}

/* ============================================================================
 * FUNÇÃO PRINCIPAL
 * ============================================================================ */

void print_header() {
    printf("\n");
    printf("================================================================\n");
    printf("          K-MEANS 1D - VERSAO SERIAL (SEQUENCIAL)              \n");
    printf("                  Projeto PCD - Baseline                        \n");
    printf("================================================================\n\n");
}

void print_config(int N, int K, int max_iter, double eps) {
    printf("----------------------------------------------------------------\n");
    printf(" CONFIGURACAO                                                  \n");
    printf("----------------------------------------------------------------\n");
    printf(" Pontos (N):           %10d                                \n", N);
    printf(" Clusters (K):         %10d                                \n", K);
    printf(" Max Iteracoes:        %10d                                \n", max_iter);
    printf(" Epsilon (eps):        %14.2e                            \n", eps);
    printf(" Semente (seed):       %10d                                \n", SEED);
    printf("----------------------------------------------------------------\n\n");
}

void print_results(KMeansMetrics* metrics, int N) {
    printf("----------------------------------------------------------------\n");
    printf(" RESULTADOS                                                    \n");
    printf("----------------------------------------------------------------\n");
    printf(" Iteracoes:            %10d                                \n", metrics->iterations);
    printf(" SSE Final:            %14.6f                        \n", metrics->sse_final);
    printf("----------------------------------------------------------------\n");
    printf(" TEMPO DE EXECUCAO                                             \n");
    printf("----------------------------------------------------------------\n");
    printf(" Tempo Total:          %10.3f ms                            \n", metrics->time_total_ms);
    printf(" Tempo Assignment:     %10.3f ms (%5.1f%%)                  \n", 
           metrics->time_assignment_ms,
           100.0 * metrics->time_assignment_ms / metrics->time_total_ms);
    printf(" Tempo Update:         %10.3f ms (%5.1f%%)                  \n", 
           metrics->time_update_ms,
           100.0 * metrics->time_update_ms / metrics->time_total_ms);
    printf("----------------------------------------------------------------\n");
    printf(" DESEMPENHO                                                    \n");
    printf("----------------------------------------------------------------\n");
    printf(" Throughput:           %10.0f pontos/segundo                \n", metrics->throughput);
    printf(" Tempo por iteracao:   %10.3f ms                            \n", 
           metrics->time_total_ms / metrics->iterations);
    printf("----------------------------------------------------------------\n\n");
}

int main(int argc, char* argv[]) {
    // Arquivos de entrada (padrão ou argumentos)
    const char* data_file = "../data/dados.csv";
    const char* centroids_file = "../data/centroides_iniciais.csv";
    
    // Parse de argumentos (compatível com outras versões)
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            data_file = argv[++i];
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            centroids_file = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Uso: %s [opcoes]\n", argv[0]);
            printf("Opcoes:\n");
            printf("  -d <arquivo>   Arquivo de dados (CSV)\n");
            printf("  -c <arquivo>   Arquivo de centroides iniciais (CSV)\n");
            printf("  -h, --help     Exibe esta ajuda\n");
            return 0;
        }
    }
    
    // Fallback para argumentos posicionais (compatibilidade)
    if (argc >= 3 && argv[1][0] != '-') {
        data_file = argv[1];
        centroids_file = argv[2];
    }
    
    print_header();
    
    // Ler dados
    int N, K;
    double* X = read_csv(data_file, &N);
    double* C = read_csv(centroids_file, &K);
    
    // Fazer cópia dos centróides iniciais (para reprodutibilidade)
    double* C_initial = (double*)malloc(K * sizeof(double));
    memcpy(C_initial, C, K * sizeof(double));
    
    print_config(N, K, MAX_ITER, EPS);
    
    // Exibir centróides iniciais
    printf("Centróides iniciais:\n");
    for (int c = 0; c < K; c++) {
        printf("  C[%d] = %.6f\n", c, C[c]);
    }
    printf("\n");
    
    // Alocar array de atribuições
    int* assign = (int*)malloc(N * sizeof(int));
    
    // Executar K-Means
    KMeansMetrics metrics = kmeans_serial(X, N, C, K, assign, MAX_ITER, EPS);
    
    // Exibir resultados
    print_results(&metrics, N);
    
    // Exibir centróides finais
    printf("Centróides finais:\n");
    for (int c = 0; c < K; c++) {
        printf("  C[%d] = %.6f\n", c, C[c]);
    }
    printf("\n");
    
    // Salvar arquivos de saída
    save_csv_int("assign_serial.csv", assign, N);
    save_csv_double("centroids_serial.csv", C, K);
    save_sse_history("sse_history_serial.csv", metrics.sse_history, metrics.iterations);
    
    printf("Arquivos gerados:\n");
    printf("  [OK] assign_serial.csv       (atribuicao de clusters)\n");
    printf("  [OK] centroids_serial.csv    (centroides finais)\n");
    printf("  [OK] sse_history_serial.csv  (convergencia do SSE)\n\n");
    
    // Liberar memória
    free(X);
    free(C);
    free(C_initial);
    free(assign);
    free(metrics.sse_history);
    
    return 0;
}
