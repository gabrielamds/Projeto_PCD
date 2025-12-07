/**
 * ============================================================================
 * K-MEANS 1D - VERSÃO MPI (MEMÓRIA DISTRIBUÍDA)
 * ============================================================================
 * Projeto PCD - Programação Concorrente e Distribuída
 * 
 * Paralelização usando MPI para sistemas de memória distribuída.
 * Pode executar em múltiplos computadores conectados em rede.
 * 
 * Análises implementadas:
 *   - Curva de speedup por número de processos
 *   - Custo do MPI_Allreduce
 *   - Overhead de comunicação vs computação
 *   - Eficiência da distribuição de dados
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <mpi.h>

/* ============================================================================
 * PARÂMETROS PADRONIZADOS (iguais em todas as versões)
 * ============================================================================ */
#define MAX_ITER 100
#define EPS 1e-6
#define DEFAULT_N 100000
#define DEFAULT_K 5
#define SEED 42

/* ============================================================================
 * ESTRUTURAS DE DADOS
 * ============================================================================ */

typedef struct {
    int iterations;
    double sse_final;
    double time_total_ms;
    double time_assignment_ms;
    double time_update_ms;
    double time_communication_ms;
    double time_allreduce_ms;
    double time_scatter_ms;
    double time_gather_ms;
    double throughput;
    double* sse_history;
    int num_procs;
    int local_N;
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
        MPI_Abort(MPI_COMM_WORLD, 1);
        return NULL;
    }
    
    int n = 0;
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (strlen(line) > 1) n++;
    }
    
    double* data = (double*)malloc(n * sizeof(double));
    
    rewind(file);
    int i = 0;
    while (fgets(line, sizeof(line), file) && i < n) {
        if (strlen(line) > 1) data[i++] = atof(line);
    }
    
    fclose(file);
    *count = i;
    return data;
}

void save_csv_int(const char* filename, int* data, int n) {
    FILE* file = fopen(filename, "w");
    if (!file) return;
    for (int i = 0; i < n; i++) fprintf(file, "%d\n", data[i]);
    fclose(file);
}

void save_csv_double(const char* filename, double* data, int n) {
    FILE* file = fopen(filename, "w");
    if (!file) return;
    for (int i = 0; i < n; i++) fprintf(file, "%.10f\n", data[i]);
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
 * ALGORITMO K-MEANS MPI
 * ============================================================================ */

KMeansMetrics kmeans_mpi(const char* data_file, const char* centroids_file,
                         int max_iter, double eps) {
    KMeansMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    metrics.num_procs = size;
    
    int N = 0, K = 0;
    double* X_all = NULL;
    double* C = NULL;
    int* assign_all = NULL;
    
    double* X_local = NULL;
    int* assign_local = NULL;
    int local_N = 0;
    
    int* sendcounts = NULL;
    int* displs = NULL;
    
    double start_total = MPI_Wtime();
    
    // ========================================
    // PASSO 1: Processo 0 lê os dados
    // ========================================
    if (rank == 0) {
        X_all = read_csv(data_file, &N);
        C = read_csv(centroids_file, &K);
        assign_all = (int*)malloc(N * sizeof(int));
        metrics.sse_history = (double*)malloc(max_iter * sizeof(double));
    }
    
    // Broadcast N e K
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Alocar centróides em todos os processos
    if (rank != 0) {
        C = (double*)malloc(K * sizeof(double));
        metrics.sse_history = (double*)malloc(max_iter * sizeof(double));
    }
    MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // ========================================
    // PASSO 2: Calcular distribuição de dados
    // ========================================
    sendcounts = (int*)malloc(size * sizeof(int));
    displs = (int*)malloc(size * sizeof(int));
    
    int base_count = N / size;
    int remainder = N % size;
    
    int offset = 0;
    for (int i = 0; i < size; i++) {
        sendcounts[i] = base_count + (i < remainder ? 1 : 0);
        displs[i] = offset;
        offset += sendcounts[i];
    }
    
    local_N = sendcounts[rank];
    metrics.local_N = local_N;
    
    X_local = (double*)malloc(local_N * sizeof(double));
    assign_local = (int*)malloc(local_N * sizeof(int));
    
    // Distribuir dados (medir tempo)
    double start_scatter = MPI_Wtime();
    MPI_Scatterv(X_all, sendcounts, displs, MPI_DOUBLE,
                 X_local, local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double end_scatter = MPI_Wtime();
    metrics.time_scatter_ms = (end_scatter - start_scatter) * 1000.0;
    
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
    
    double time_allreduce_total = 0.0;
    double time_assignment_total = 0.0;
    double time_update_total = 0.0;
    
    for (iter = 0; iter < max_iter; iter++) {
        double local_sse = 0.0;
        
        // ================================================================
        // PASSO 3a: ASSIGNMENT (local)
        // ================================================================
        double start_assign = MPI_Wtime();
        
        for (int i = 0; i < local_N; i++) {
            double min_dist = DBL_MAX;
            int best_cluster = 0;
            
            for (int c = 0; c < K; c++) {
                double diff = X_local[i] - C[c];
                double dist = diff * diff;
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            
            assign_local[i] = best_cluster;
            local_sse += min_dist;
        }
        
        double end_assign = MPI_Wtime();
        time_assignment_total += (end_assign - start_assign) * 1000.0;
        
        // Redução global do SSE (medir tempo)
        double start_allreduce = MPI_Wtime();
        MPI_Allreduce(&local_sse, &sse, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double end_allreduce = MPI_Wtime();
        time_allreduce_total += (end_allreduce - start_allreduce) * 1000.0;
        
        metrics.sse_history[iter] = sse;
        
        // Verificar convergência
        if (fabs(prev_sse - sse) < eps) {
            iter++;
            break;
        }
        prev_sse = sse;
        
        // ================================================================
        // PASSO 3b: UPDATE (distribuído)
        // ================================================================
        double start_update = MPI_Wtime();
        
        for (int c = 0; c < K; c++) {
            local_sum[c] = 0.0;
            local_count[c] = 0;
        }
        
        for (int i = 0; i < local_N; i++) {
            int c = assign_local[i];
            local_sum[c] += X_local[i];
            local_count[c]++;
        }
        
        double end_update = MPI_Wtime();
        time_update_total += (end_update - start_update) * 1000.0;
        
        // Redução global de somas e contadores
        double start_allreduce2 = MPI_Wtime();
        MPI_Allreduce(local_sum, global_sum, K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_count, global_count, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        double end_allreduce2 = MPI_Wtime();
        time_allreduce_total += (end_allreduce2 - start_allreduce2) * 1000.0;
        
        // Atualizar centróides (todos os processos)
        for (int c = 0; c < K; c++) {
            if (global_count[c] > 0) {
                C[c] = global_sum[c] / global_count[c];
            } else {
                double x0;
                if (rank == 0) {
                    x0 = X_all[0];
                }
                MPI_Bcast(&x0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                C[c] = x0;
            }
        }
    }
    
    // ========================================
    // PASSO 4: Coletar resultados
    // ========================================
    double start_gather = MPI_Wtime();
    MPI_Gatherv(assign_local, local_N, MPI_INT,
                assign_all, sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    double end_gather = MPI_Wtime();
    metrics.time_gather_ms = (end_gather - start_gather) * 1000.0;
    
    double end_total = MPI_Wtime();
    
    // Preencher métricas
    metrics.iterations = iter;
    metrics.sse_final = sse;
    metrics.time_total_ms = (end_total - start_total) * 1000.0;
    metrics.time_assignment_ms = time_assignment_total;
    metrics.time_update_ms = time_update_total;
    metrics.time_allreduce_ms = time_allreduce_total;
    metrics.time_communication_ms = metrics.time_scatter_ms + 
                                     metrics.time_gather_ms + 
                                     time_allreduce_total;
    metrics.throughput = (double)(N * iter) / (metrics.time_total_ms / 1000.0);
    
    // ========================================
    // PASSO 5: Processo 0 salva resultados
    // ========================================
    if (rank == 0) {
        save_csv_int("assign_mpi.csv", assign_all, N);
        save_csv_double("centroids_mpi.csv", C, K);
        save_sse_history("sse_history_mpi.csv", metrics.sse_history, metrics.iterations);
        
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
    
    return metrics;
}

/* ============================================================================
 * ANÁLISE DE ESCALABILIDADE
 * ============================================================================ */

void run_scalability_analysis(int rank, int size, KMeansMetrics* metrics, double serial_time) {
    if (rank != 0) return;
    
    double speedup = serial_time > 0 ? serial_time / metrics->time_total_ms : 0;
    double efficiency = speedup / size * 100.0;
    
    // Salvar em arquivo para agregação posterior
    FILE* f = fopen("scalability_mpi.csv", "a");
    if (f) {
        // Se arquivo vazio, adicionar cabeçalho
        fseek(f, 0, SEEK_END);
        if (ftell(f) == 0) {
            fprintf(f, "num_procs,time_total_ms,time_compute_ms,time_comm_ms,time_allreduce_ms,speedup,efficiency,throughput\n");
        }
        
        double time_compute = metrics->time_assignment_ms + metrics->time_update_ms;
        
        fprintf(f, "%d,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f,%.0f\n",
                size, metrics->time_total_ms, time_compute,
                metrics->time_communication_ms, metrics->time_allreduce_ms,
                speedup, efficiency / 100.0, metrics->throughput);
        fclose(f);
    }
}

/* ============================================================================
 * FUNÇÕES DE IMPRESSÃO
 * ============================================================================ */

void print_header(int rank) {
    if (rank != 0) return;
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         K-MEANS 1D - VERSÃO MPI (DISTRIBUÍDO)               ║\n");
    printf("║              Memória Distribuída - Multi-processo            ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
}

void print_config(int rank, int N, int K, int max_iter, double eps, int size) {
    if (rank != 0) return;
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│ CONFIGURAÇÃO                                                │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ Pontos (N):           %10d                            │\n", N);
    printf("│ Clusters (K):         %10d                            │\n", K);
    printf("│ Max Iterações:        %10d                            │\n", max_iter);
    printf("│ Epsilon (eps):        %14.2e                        │\n", eps);
    printf("│ Processos MPI:        %10d                            │\n", size);
    printf("│ Pontos/Processo:      %10d (aprox)                   │\n", N / size);
    printf("└─────────────────────────────────────────────────────────────┘\n\n");
}

void print_results(int rank, KMeansMetrics* metrics, double serial_time) {
    if (rank != 0) return;
    
    double time_compute = metrics->time_assignment_ms + metrics->time_update_ms;
    double speedup = serial_time > 0 ? serial_time / metrics->time_total_ms : 0;
    double efficiency = speedup / metrics->num_procs * 100.0;
    
    printf("┌─────────────────────────────────────────────────────────────┐\n");
    printf("│ RESULTADOS                                                  │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ Iterações:            %10d                            │\n", metrics->iterations);
    printf("│ SSE Final:            %14.6f                    │\n", metrics->sse_final);
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ TEMPO DE EXECUÇÃO                                           │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ Tempo Total:          %10.3f ms                        │\n", metrics->time_total_ms);
    printf("│ Tempo Computação:     %10.3f ms (%5.1f%%)               │\n",
           time_compute, 100.0 * time_compute / metrics->time_total_ms);
    printf("│   - Assignment:       %10.3f ms                        │\n", metrics->time_assignment_ms);
    printf("│   - Update:           %10.3f ms                        │\n", metrics->time_update_ms);
    printf("│ Tempo Comunicação:    %10.3f ms (%5.1f%%)               │\n",
           metrics->time_communication_ms,
           100.0 * metrics->time_communication_ms / metrics->time_total_ms);
    printf("│   - Scatter:          %10.3f ms                        │\n", metrics->time_scatter_ms);
    printf("│   - Allreduce:        %10.3f ms                        │\n", metrics->time_allreduce_ms);
    printf("│   - Gather:           %10.3f ms                        │\n", metrics->time_gather_ms);
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ DESEMPENHO                                                  │\n");
    printf("├─────────────────────────────────────────────────────────────┤\n");
    printf("│ Processos MPI:        %10d                            │\n", metrics->num_procs);
    printf("│ Pontos/Processo:      %10d                            │\n", metrics->local_N);
    printf("│ Throughput:           %10.0f pontos/s                  │\n", metrics->throughput);
    if (serial_time > 0) {
        printf("│ Speedup:              %10.2fx                          │\n", speedup);
        printf("│ Eficiência:           %10.1f%%                         │\n", efficiency);
    }
    printf("└─────────────────────────────────────────────────────────────┘\n\n");
}

/* ============================================================================
 * FUNÇÃO PRINCIPAL
 * ============================================================================ */

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    const char* data_file = "../data/dados.csv";
    const char* centroids_file = "../data/centroides_iniciais.csv";
    double serial_time = 0.0;
    int run_analysis = 0;
    
    // Parsing de argumentos
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            data_file = argv[++i];
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            centroids_file = argv[++i];
        } else if (strcmp(argv[i], "--serial-time") == 0 && i + 1 < argc) {
            serial_time = atof(argv[++i]);
        } else if (strcmp(argv[i], "-a") == 0) {
            run_analysis = 1;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            if (rank == 0) {
                printf("Uso: mpirun -np <num_procs> %s [opções]\n", argv[0]);
                printf("Opções:\n");
                printf("  -d <arquivo>       Arquivo de dados\n");
                printf("  -c <arquivo>       Arquivo de centróides\n");
                printf("  --serial-time <ms> Tempo serial (para speedup)\n");
                printf("  -a                 Salvar análise de escalabilidade\n");
                printf("  -h, --help         Mostrar ajuda\n");
            }
            MPI_Finalize();
            return 0;
        }
    }
    
    print_header(rank);
    
    // Ler N e K para exibir configuração
    int N = 0, K = 0;
    if (rank == 0) {
        int temp;
        double* temp_data = read_csv(data_file, &temp);
        N = temp;
        free(temp_data);
        temp_data = read_csv(centroids_file, &temp);
        K = temp;
        free(temp_data);
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    print_config(rank, N, K, MAX_ITER, EPS, size);
    
    // Executar K-Means
    KMeansMetrics metrics = kmeans_mpi(data_file, centroids_file, MAX_ITER, EPS);
    
    print_results(rank, &metrics, serial_time);
    
    if (rank == 0) {
        // Ler centróides finais para exibir
        double* C_final = read_csv("centroids_mpi.csv", &K);
        printf("Centróides finais:\n");
        for (int c = 0; c < K; c++) {
            printf("  C[%d] = %.6f\n", c, C_final[c]);
        }
        printf("\n");
        free(C_final);
        
        printf("Arquivos gerados:\n");
        printf("  ✓ assign_mpi.csv\n");
        printf("  ✓ centroids_mpi.csv\n");
        printf("  ✓ sse_history_mpi.csv\n");
        
        if (run_analysis) {
            run_scalability_analysis(rank, size, &metrics, serial_time);
            printf("  ✓ scalability_mpi.csv (atualizado)\n");
        }
    }
    
    free(metrics.sse_history);
    
    MPI_Finalize();
    return 0;
}
