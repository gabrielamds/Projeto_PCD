/**
 * Gerador de Dados de Teste para K-Means
 * 
 * Gera arquivos CSV com dados e centróides iniciais para testes.
 * 
 * Uso: ./generate_data [N] [K] [seed]
 *   N    - número de pontos (padrão: 100000)
 *   K    - número de clusters (padrão: 5)
 *   seed - semente para números aleatórios (padrão: 42)
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEFAULT_N 100000
#define DEFAULT_K 5
#define DEFAULT_SEED 42

// Gera número aleatório double entre min e max
double random_double(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
}

// Gera número com distribuição normal aproximada (Box-Muller)
double random_normal(double mean, double stddev) {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    
    // Evitar log(0)
    while (u1 <= 1e-10) {
        u1 = (double)rand() / RAND_MAX;
    }
    
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + z * stddev;
}

int main(int argc, char* argv[]) {
    int N = DEFAULT_N;
    int K = DEFAULT_K;
    int seed = DEFAULT_SEED;
    
    // Parsing de argumentos
    if (argc >= 2) N = atoi(argv[1]);
    if (argc >= 3) K = atoi(argv[2]);
    if (argc >= 4) seed = atoi(argv[3]);
    
    srand(seed);
    
    printf("Gerando dados de teste...\n");
    printf("  Pontos (N): %d\n", N);
    printf("  Clusters (K): %d\n", K);
    printf("  Seed: %d\n\n", seed);
    
    // ========================================
    // Gerar centros "reais" para criar clusters
    // ========================================
    double* true_centers = (double*)malloc(K * sizeof(double));
    double range = 100.0;  // Intervalo dos dados
    
    for (int c = 0; c < K; c++) {
        // Distribuir centros uniformemente
        true_centers[c] = (c + 0.5) * range / K;
    }
    
    printf("Centros reais (para gerar dados):\n");
    for (int c = 0; c < K; c++) {
        printf("  Centro[%d] = %.2f\n", c, true_centers[c]);
    }
    printf("\n");
    
    // ========================================
    // Gerar pontos em torno dos centros
    // ========================================
    FILE* data_file = fopen("dados.csv", "w");
    if (!data_file) {
        fprintf(stderr, "Erro ao criar dados.csv\n");
        return 1;
    }
    
    double stddev = range / K / 3.0;  // Desvio padrão
    
    for (int i = 0; i < N; i++) {
        // Escolher cluster aleatório
        int cluster = rand() % K;
        
        // Gerar ponto com distribuição normal em torno do centro
        double point = random_normal(true_centers[cluster], stddev);
        
        fprintf(data_file, "%.6f\n", point);
    }
    
    fclose(data_file);
    printf("Arquivo gerado: dados.csv (%d pontos)\n", N);
    
    // ========================================
    // Gerar centróides iniciais (aleatórios)
    // ========================================
    FILE* centroids_file = fopen("centroides_iniciais.csv", "w");
    if (!centroids_file) {
        fprintf(stderr, "Erro ao criar centroides_iniciais.csv\n");
        return 1;
    }
    
    printf("\nCentróides iniciais (aleatórios):\n");
    for (int c = 0; c < K; c++) {
        double centroid = random_double(0, range);
        fprintf(centroids_file, "%.6f\n", centroid);
        printf("  C[%d] = %.6f\n", c, centroid);
    }
    
    fclose(centroids_file);
    printf("\nArquivo gerado: centroides_iniciais.csv (%d centróides)\n", K);
    
    free(true_centers);
    
    printf("\n✓ Dados gerados com sucesso!\n");
    
    return 0;
}
