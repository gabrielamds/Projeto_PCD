/**
 * ============================================================================
 * K-MEANS 1D - VERSÃO OPENMP (CPU - MEMÓRIA COMPARTILHADA)
 * ============================================================================
 * Projeto PCD - Programação Concorrente e Distribuída
 * 
 * Paralelização usando OpenMP para CPUs multi-core.
 * 
 * Análises implementadas:
 *   - Variação do número de threads (1, 2, 4, 8, ...)
 *   - Comparação de estratégias de schedule (static, dynamic, guided)
 *   - Speedup e eficiência vs versão serial
 *   - Overhead de sincronização
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>

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

typedef struct {
    int iterations;
    double sse_final;
    double time_total_ms;
    double time_assignment_ms;
    double time_update_ms;
    double throughput;
    double* sse_history;
    int num_threads;
    char schedule_type[32];
    double speedup;
    double efficiency;
} KMeansMetrics;

/* ============================================================================
 * FUNÇÕES UTILITÁRIAS
 * ============================================================================ */

double* read_csv(const char* filename, int* count) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "ERRO: Não foi possível abrir '%s'\n", filename);
        exit(EXIT_FAILURE);
    }
    
    int n = 0;
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (strlen(line) > 1) n++;
    }
    
    double* data = (double*)malloc(n * sizeof(double));
    if (!data) {
        fprintf(stderr, "ERRO: Falha na alocação de memória\n");
        exit(EXIT_FAILURE);
    }
    
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

void save_csv_int(const char* filename, int* data, int n) {
    FILE* file = fopen(filename, "w");
    if (!file) return;
    for (int i = 0; i < n; i++) {
        fprintf(file, "%d\n", data[i]);
    }
    fclose(file);
}

void save_csv_double(const char* filename, double* data, int n) {
    FILE* file = fopen(filename, "w");
    if (!file) return;
    for (int i = 0; i < n; i++) {
        fprintf(file, "%.10f\n", data[i]);
    }
    fclose(file);
}

void save_sse_history(const char* filename, double* sse_history, int iterations) {
    FILE* file = fopen(filename, "w");
    if (!file) return;
    fprintf(file, "iteracao,sse\n");
    for (int i = 0; i < iterations; i++) {
        fprintf(file, "%d,%.10f\n", i + 1, sse_history[i]);
    }
    fclose(file);
}

/* ============================================================================
 * ALGORITMO K-MEANS COM OPENMP
 * ============================================================================ */

/**
 * K-Means paralelizado com OpenMP
 * 
 * Estratégias de paralelização:
 * 1. Loop de Assignment: #pragma omp parallel for reduction(+:sse)
 * 2. Loop de Update: arrays locais por thread + seção crítica
 * 
 * @param schedule_type: "static", "dynamic", ou "guided"
 */
KMeansMetrics kmeans_openmp(double* X, int N, double* C, int K, int* assign,
                            int max_iter, double eps, int num_threads,
                            const char* schedule_type) {
    KMeansMetrics metrics;
    metrics.sse_history = (double*)malloc(max_iter * sizeof(double));
    metrics.time_assignment_ms = 0;
    metrics.time_update_ms = 0;
    metrics.num_threads = num_threads;
    strncpy(metrics.schedule_type, schedule_type, sizeof(metrics.schedule_type) - 1);
    
    // Configurar número de threads
    omp_set_num_threads(num_threads);
    
    // Arrays auxiliares globais
    double* sum = (double*)calloc(K, sizeof(double));
    int* count = (int*)calloc(K, sizeof(int));
    
    double start_total = omp_get_wtime();
    double prev_sse = DBL_MAX;
    double sse = 0.0;
    int iter;
    
    for (iter = 0; iter < max_iter; iter++) {
        sse = 0.0;
        
        /* ================================================================
         * PASSO 1: ASSIGNMENT (Paralelizado)
         * ================================================================ */
        double start_assign = omp_get_wtime();
        
        // Escolher schedule baseado no parâmetro
        if (strcmp(schedule_type, "static") == 0) {
            #pragma omp parallel for reduction(+:sse) schedule(static)
            for (int i = 0; i < N; i++) {
                double min_dist = DBL_MAX;
                int best_cluster = 0;
                
                for (int c = 0; c < K; c++) {
                    double diff = X[i] - C[c];
                    double dist = diff * diff;
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_cluster = c;
                    }
                }
                
                assign[i] = best_cluster;
                sse += min_dist;
            }
        } else if (strcmp(schedule_type, "dynamic") == 0) {
            #pragma omp parallel for reduction(+:sse) schedule(dynamic, 1000)
            for (int i = 0; i < N; i++) {
                double min_dist = DBL_MAX;
                int best_cluster = 0;
                
                for (int c = 0; c < K; c++) {
                    double diff = X[i] - C[c];
                    double dist = diff * diff;
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_cluster = c;
                    }
                }
                
                assign[i] = best_cluster;
                sse += min_dist;
            }
        } else {  // guided
            #pragma omp parallel for reduction(+:sse) schedule(guided)
            for (int i = 0; i < N; i++) {
                double min_dist = DBL_MAX;
                int best_cluster = 0;
                
                for (int c = 0; c < K; c++) {
                    double diff = X[i] - C[c];
                    double dist = diff * diff;
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_cluster = c;
                    }
                }
                
                assign[i] = best_cluster;
                sse += min_dist;
            }
        }
        
        double end_assign = omp_get_wtime();
        metrics.time_assignment_ms += (end_assign - start_assign) * 1000.0;
        
        metrics.sse_history[iter] = sse;
        
        if (fabs(prev_sse - sse) < eps) {
            iter++;
            break;
        }
        prev_sse = sse;
        
        /* ================================================================
         * PASSO 2: UPDATE (Paralelizado)
         * ================================================================ */
        double start_update = omp_get_wtime();
        
        // Zerar acumuladores globais
        for (int c = 0; c < K; c++) {
            sum[c] = 0.0;
            count[c] = 0;
        }
        
        // Acumular com arrays locais por thread (evita contenção)
        #pragma omp parallel
        {
            double* local_sum = (double*)calloc(K, sizeof(double));
            int* local_count = (int*)calloc(K, sizeof(int));
            
            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) {
                int c = assign[i];
                local_sum[c] += X[i];
                local_count[c]++;
            }
            
            // Combinar resultados (seção crítica)
            #pragma omp critical
            {
                for (int c = 0; c < K; c++) {
                    sum[c] += local_sum[c];
                    count[c] += local_count[c];
                }
            }
            
            free(local_sum);
            free(local_count);
        }
        
        // Atualizar centróides
        for (int c = 0; c < K; c++) {
            if (count[c] > 0) {
                C[c] = sum[c] / count[c];
            } else {
                C[c] = X[0];
            }
        }
        
        double end_update = omp_get_wtime();
        metrics.time_update_ms += (end_update - start_update) * 1000.0;
    }
    
    double end_total = omp_get_wtime();
    
    metrics.iterations = iter;
    metrics.sse_final = sse;
    metrics.time_total_ms = (end_total - start_total) * 1000.0;
    metrics.throughput = (double)(N * iter) / (metrics.time_total_ms / 1000.0);
    
    free(sum);
    free(count);
    
    return metrics;
}

/* ============================================================================
 * ANÁLISE DE ESCALABILIDADE
 * ============================================================================ */

void run_scalability_analysis(double* X, int N, double* C_original, int K,
                              int max_iter, double eps, double serial_time) {
    printf("\n");
    printf("================================================================\n");
    printf("              ANALISE DE ESCALABILIDADE                        \n");
    printf("================================================================\n\n");
    
    int thread_counts[] = {1, 2, 4, 8, 16};
    int num_tests = 5;
    int max_threads = omp_get_max_threads();
    
    // Arquivo para resultados
    FILE* results = fopen("scalability_openmp.csv", "w");
    fprintf(results, "threads,schedule,time_ms,speedup,efficiency,throughput\n");
    
    const char* schedules[] = {"static", "dynamic", "guided"};
    
    printf("---------+----------+-----------+---------+------------+-------------\n");
    printf(" Threads | Schedule | Tempo(ms) | Speedup | Eficiencia | Throughput  \n");
    printf("---------+----------+-----------+---------+------------+-------------\n");
    
    for (int s = 0; s < 3; s++) {
        for (int t = 0; t < num_tests; t++) {
            int num_threads = thread_counts[t];
            if (num_threads > max_threads) continue;
            
            // Copiar centróides originais
            double* C = (double*)malloc(K * sizeof(double));
            memcpy(C, C_original, K * sizeof(double));
            int* assign = (int*)malloc(N * sizeof(int));
            
            KMeansMetrics metrics = kmeans_openmp(X, N, C, K, assign,
                                                   max_iter, eps, num_threads,
                                                   schedules[s]);
            
            double speedup = serial_time / metrics.time_total_ms;
            double efficiency = speedup / num_threads * 100.0;
            
            printf("   %2d    | %-8s | %9.3f |  %5.2fx |   %5.1f%%   | %9.0f/s \n",
                   num_threads, schedules[s], metrics.time_total_ms,
                   speedup, efficiency, metrics.throughput);
            
            fprintf(results, "%d,%s,%.3f,%.4f,%.4f,%.0f\n",
                    num_threads, schedules[s], metrics.time_total_ms,
                    speedup, efficiency / 100.0, metrics.throughput);
            
            free(C);
            free(assign);
            free(metrics.sse_history);
        }
        if (s < 2) {
            printf("---------+----------+-----------+---------+------------+-------------\n");
        }
    }
    
    printf("---------+----------+-----------+---------+------------+-------------\n");
    
    fclose(results);
    printf("\n[OK] Resultados salvos em: scalability_openmp.csv\n");
}

/* ============================================================================
 * FUNÇÃO PRINCIPAL
 * ============================================================================ */

void print_header() {
    printf("\n");
    printf("================================================================\n");
    printf("         K-MEANS 1D - VERSAO OPENMP (CPU PARALELO)            \n");
    printf("              Memoria Compartilhada - Multi-thread             \n");
    printf("================================================================\n\n");
}

void print_config(int N, int K, int max_iter, double eps, int num_threads) {
    printf("----------------------------------------------------------------\n");
    printf(" CONFIGURACAO                                                  \n");
    printf("----------------------------------------------------------------\n");
    printf(" Pontos (N):           %10d                                \n", N);
    printf(" Clusters (K):         %10d                                \n", K);
    printf(" Max Iteracoes:        %10d                                \n", max_iter);
    printf(" Epsilon (eps):        %14.2e                            \n", eps);
    printf(" Threads OpenMP:       %10d                                \n", num_threads);
    printf(" Max Threads Disp.:    %10d                                \n", omp_get_max_threads());
    printf("----------------------------------------------------------------\n\n");
}

void print_results(KMeansMetrics* metrics) {
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
    printf(" CONFIGURACAO PARALELA                                         \n");
    printf("----------------------------------------------------------------\n");
    printf(" Threads utilizadas:   %10d                                \n", metrics->num_threads);
    printf(" Schedule:             %-10s                              \n", metrics->schedule_type);
    printf(" Throughput:           %10.0f pontos/s                     \n", metrics->throughput);
    printf("----------------------------------------------------------------\n\n");
}

int main(int argc, char* argv[]) {
    const char* data_file = "../data/dados.csv";
    const char* centroids_file = "../data/centroides_iniciais.csv";
    int num_threads = omp_get_max_threads();
    const char* schedule_type = "static";
    int run_analysis = 0;
    double serial_time = 0.0;
    
    // Parsing de argumentos
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            schedule_type = argv[++i];
        } else if (strcmp(argv[i], "-a") == 0) {
            run_analysis = 1;
        } else if (strcmp(argv[i], "--serial-time") == 0 && i + 1 < argc) {
            serial_time = atof(argv[++i]);
        } else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            data_file = argv[++i];
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            centroids_file = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Uso: %s [opções]\n", argv[0]);
            printf("Opções:\n");
            printf("  -t <num>      Número de threads (padrão: max disponível)\n");
            printf("  -s <tipo>     Schedule: static, dynamic, guided (padrão: static)\n");
            printf("  -a            Executar análise de escalabilidade\n");
            printf("  --serial-time <ms>  Tempo da versão serial (para speedup)\n");
            printf("  -d <arquivo>  Arquivo de dados\n");
            printf("  -c <arquivo>  Arquivo de centróides\n");
            return 0;
        }
    }
    
    print_header();
    
    // Ler dados
    int N, K;
    double* X = read_csv(data_file, &N);
    double* C = read_csv(centroids_file, &K);
    double* C_original = (double*)malloc(K * sizeof(double));
    memcpy(C_original, C, K * sizeof(double));
    
    print_config(N, K, MAX_ITER, EPS, num_threads);
    
    printf("Centróides iniciais:\n");
    for (int c = 0; c < K; c++) {
        printf("  C[%d] = %.6f\n", c, C[c]);
    }
    printf("\n");
    
    int* assign = (int*)malloc(N * sizeof(int));
    
    // Executar K-Means
    KMeansMetrics metrics = kmeans_openmp(X, N, C, K, assign, MAX_ITER, EPS,
                                          num_threads, schedule_type);
    
    // Calcular speedup se tempo serial foi fornecido
    if (serial_time > 0) {
        metrics.speedup = serial_time / metrics.time_total_ms;
        metrics.efficiency = metrics.speedup / num_threads;
    }
    
    print_results(&metrics);
    
    printf("Centróides finais:\n");
    for (int c = 0; c < K; c++) {
        printf("  C[%d] = %.6f\n", c, C[c]);
    }
    printf("\n");
    
    // Salvar arquivos
    save_csv_int("assign_openmp.csv", assign, N);
    save_csv_double("centroids_openmp.csv", C, K);
    save_sse_history("sse_history_openmp.csv", metrics.sse_history, metrics.iterations);
    
    printf("Arquivos gerados:\n");
    printf("  [OK] assign_openmp.csv\n");
    printf("  [OK] centroids_openmp.csv\n");
    printf("  [OK] sse_history_openmp.csv\n");
    
    // Análise de escalabilidade
    if (run_analysis) {
        if (serial_time <= 0) {
            // Estimar tempo serial executando com 1 thread
            memcpy(C, C_original, K * sizeof(double));
            KMeansMetrics serial_metrics = kmeans_openmp(X, N, C, K, assign,
                                                          MAX_ITER, EPS, 1, "static");
            serial_time = serial_metrics.time_total_ms;
            free(serial_metrics.sse_history);
        }
        
        memcpy(C, C_original, K * sizeof(double));
        run_scalability_analysis(X, N, C_original, K, MAX_ITER, EPS, serial_time);
    }
    
    // Liberar memória
    free(X);
    free(C);
    free(C_original);
    free(assign);
    free(metrics.sse_history);
    
    return 0;
}
