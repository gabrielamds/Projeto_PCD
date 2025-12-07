/**
 * ============================================================================
 * GERADOR DE DADOS PARA K-MEANS
 * ============================================================================
 * Projeto PCD - Programação Concorrente e Distribuída
 * 
 * Gera datasets sintéticos para testes do K-Means com:
 *   - Semente fixa para reprodutibilidade
 *   - Clusters bem definidos para validação
 *   - Múltiplos tamanhos de dados
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265358979323846

/* ============================================================================
 * GERADOR DE NÚMEROS ALEATÓRIOS COM SEMENTE
 * ============================================================================ */

static unsigned long seed_state = 42;

// LCG (Linear Congruential Generator) para reprodutibilidade
unsigned long lcg_rand() {
    seed_state = (seed_state * 1103515245 + 12345) & 0x7fffffff;
    return seed_state;
}

void lcg_seed(unsigned long seed) {
    seed_state = seed;
}

double rand_double() {
    return (double)lcg_rand() / 0x7fffffff;
}

// Box-Muller para distribuição normal
double rand_normal(double mean, double stddev) {
    double u1 = rand_double();
    double u2 = rand_double();
    
    while (u1 < 1e-10) u1 = rand_double();
    
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
    return mean + z * stddev;
}

/* ============================================================================
 * FUNÇÕES DE GERAÇÃO
 * ============================================================================ */

void generate_clustered_data(const char* data_file, const char* centroids_file,
                             int N, int K, unsigned long seed) {
    lcg_seed(seed);
    
    printf("Gerando dados clusterizados...\n");
    printf("  N = %d pontos\n", N);
    printf("  K = %d clusters\n", K);
    printf("  Seed = %lu\n\n", seed);
    
    // Definir range dos dados
    double data_min = 0.0;
    double data_max = 100.0;
    double range = data_max - data_min;
    
    // Calcular centros reais (distribuídos uniformemente)
    double* true_centers = (double*)malloc(K * sizeof(double));
    double cluster_width = range / K;
    double stddev = cluster_width / 4.0;  // Desvio padrão
    
    printf("Centros reais dos clusters:\n");
    for (int c = 0; c < K; c++) {
        true_centers[c] = data_min + (c + 0.5) * cluster_width;
        printf("  Cluster %d: centro = %.2f, stddev = %.2f\n", c, true_centers[c], stddev);
    }
    printf("\n");
    
    // Gerar pontos
    FILE* data_fp = fopen(data_file, "w");
    if (!data_fp) {
        fprintf(stderr, "ERRO: Não foi possível criar %s\n", data_file);
        exit(1);
    }
    
    // Estatísticas para validação
    double* cluster_sum = (double*)calloc(K, sizeof(double));
    int* cluster_count = (int*)calloc(K, sizeof(int));
    double global_min = data_max, global_max = data_min;
    
    for (int i = 0; i < N; i++) {
        // Escolher cluster (distribuição uniforme)
        int cluster = lcg_rand() % K;
        
        // Gerar ponto com distribuição normal
        double point = rand_normal(true_centers[cluster], stddev);
        
        // Clamp para evitar outliers extremos
        if (point < data_min) point = data_min + rand_double() * stddev;
        if (point > data_max) point = data_max - rand_double() * stddev;
        
        fprintf(data_fp, "%.10f\n", point);
        
        // Estatísticas
        cluster_sum[cluster] += point;
        cluster_count[cluster]++;
        if (point < global_min) global_min = point;
        if (point > global_max) global_max = point;
    }
    
    fclose(data_fp);
    printf("✓ Arquivo criado: %s\n", data_file);
    
    // Mostrar estatísticas
    printf("\nEstatísticas dos dados gerados:\n");
    printf("  Range: [%.2f, %.2f]\n", global_min, global_max);
    printf("  Distribuição por cluster:\n");
    for (int c = 0; c < K; c++) {
        double mean = cluster_count[c] > 0 ? cluster_sum[c] / cluster_count[c] : 0;
        printf("    Cluster %d: %d pontos (%.1f%%), média real = %.2f\n",
               c, cluster_count[c], 100.0 * cluster_count[c] / N, mean);
    }
    
    // Gerar centróides iniciais (perturbados dos centros reais)
    FILE* centroids_fp = fopen(centroids_file, "w");
    if (!centroids_fp) {
        fprintf(stderr, "ERRO: Não foi possível criar %s\n", centroids_file);
        exit(1);
    }
    
    printf("\nCentróides iniciais (perturbados):\n");
    for (int c = 0; c < K; c++) {
        // Perturbar o centro real em ±50% da largura do cluster
        double perturbation = (rand_double() - 0.5) * cluster_width;
        double centroid = true_centers[c] + perturbation;
        
        // Clamp
        if (centroid < data_min) centroid = data_min;
        if (centroid > data_max) centroid = data_max;
        
        fprintf(centroids_fp, "%.10f\n", centroid);
        printf("  C[%d] = %.6f (centro real: %.2f)\n", c, centroid, true_centers[c]);
    }
    
    fclose(centroids_fp);
    printf("\n✓ Arquivo criado: %s\n", centroids_file);
    
    free(true_centers);
    free(cluster_sum);
    free(cluster_count);
}

void print_usage(const char* prog_name) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           GERADOR DE DADOS PARA K-MEANS                     ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    printf("Uso: %s [opções]\n\n", prog_name);
    printf("Opções:\n");
    printf("  -n <num>      Número de pontos (padrão: 100000)\n");
    printf("  -k <num>      Número de clusters (padrão: 5)\n");
    printf("  -s <num>      Semente para reprodutibilidade (padrão: 42)\n");
    printf("  -o <prefixo>  Prefixo dos arquivos de saída (padrão: nenhum)\n");
    printf("  -h, --help    Mostrar esta ajuda\n");
    printf("\n");
    printf("Exemplos:\n");
    printf("  %s                          # 100k pontos, 5 clusters\n", prog_name);
    printf("  %s -n 1000000 -k 10         # 1M pontos, 10 clusters\n", prog_name);
    printf("  %s -n 50000 -k 3 -s 123     # 50k pontos, seed 123\n", prog_name);
    printf("\n");
    printf("Arquivos gerados:\n");
    printf("  - dados.csv               Pontos (N linhas)\n");
    printf("  - centroides_iniciais.csv Centróides (K linhas)\n");
    printf("\n");
}

int main(int argc, char* argv[]) {
    int N = 100000;
    int K = 5;
    unsigned long seed = 42;
    const char* prefix = "";
    
    // Parse argumentos
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            K = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            seed = (unsigned long)atol(argv[++i]);
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            prefix = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Argumento desconhecido: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Validar parâmetros
    if (N < 1) {
        fprintf(stderr, "ERRO: N deve ser >= 1\n");
        return 1;
    }
    if (K < 1 || K > N) {
        fprintf(stderr, "ERRO: K deve estar entre 1 e N\n");
        return 1;
    }
    
    // Gerar nomes dos arquivos
    char data_file[256], centroids_file[256];
    if (strlen(prefix) > 0) {
        snprintf(data_file, sizeof(data_file), "%s_dados.csv", prefix);
        snprintf(centroids_file, sizeof(centroids_file), "%s_centroides_iniciais.csv", prefix);
    } else {
        strcpy(data_file, "dados.csv");
        strcpy(centroids_file, "centroides_iniciais.csv");
    }
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║           GERADOR DE DADOS PARA K-MEANS                     ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    generate_clustered_data(data_file, centroids_file, N, K, seed);
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                    GERAÇÃO CONCLUÍDA                        ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    return 0;
}
