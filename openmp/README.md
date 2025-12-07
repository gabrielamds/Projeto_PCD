# K-Means 1D - Versão OpenMP (CPU Paralelo)

## Descrição

Versão paralelizada do K-Means usando **OpenMP** para CPUs multi-core com memória compartilhada.

## Características

- Paralelização do loop de Assignment com `reduction` para SSE
- Paralelização do loop de Update com arrays locais por thread
- Suporte a diferentes estratégias de scheduling (`static`, `dynamic`, `guided`)
- Análise automática de escalabilidade (speedup por número de threads)

## Compilação

### Linux/macOS
```bash
gcc -O3 -fopenmp -o kmeans_openmp kmeans_openmp.c -lm
```

### Windows (MinGW)
```bash
gcc -O3 -fopenmp -o kmeans_openmp.exe kmeans_openmp.c
```

### Windows (Visual Studio)
```bash
cl /O2 /openmp kmeans_openmp.c
```

## Execução

### Uso básico
```bash
# Usar todos os threads disponíveis
./kmeans_openmp

# Especificar número de threads
./kmeans_openmp -t 4

# Especificar estratégia de schedule
./kmeans_openmp -t 4 -s dynamic
```

### Todas as opções
```bash
./kmeans_openmp [opções]

Opções:
  -t <num>           Número de threads (padrão: máximo disponível)
  -s <tipo>          Schedule: static, dynamic, guided (padrão: static)
  -a                 Executar análise de escalabilidade
  --serial-time <ms> Tempo da versão serial (para calcular speedup)
  -d <arquivo>       Arquivo de dados de entrada
  -c <arquivo>       Arquivo de centróides iniciais
  -h, --help         Mostrar ajuda
```

### Exemplos
```bash
# Execução básica com 4 threads
./kmeans_openmp -t 4

# Análise completa de escalabilidade
./kmeans_openmp -a

# Com schedule dinâmico
./kmeans_openmp -t 8 -s dynamic

# Calculando speedup com tempo serial conhecido
./kmeans_openmp -t 4 --serial-time 125.5
```

## Parâmetros Fixos

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `MAX_ITER` | 100 | Número máximo de iterações |
| `EPS` | 1e-6 | Tolerância para convergência |
| `SEED` | 42 | Semente para reprodutibilidade |

## Estratégias de Schedule

| Schedule | Descrição | Melhor para |
|----------|-----------|-------------|
| `static` | Divide igualmente entre threads | Carga uniforme |
| `dynamic` | Atribui chunks sob demanda | Carga desbalanceada |
| `guided` | Chunks decrescentes | Balanceamento adaptativo |

## Arquivos de Saída

| Arquivo | Descrição |
|---------|-----------|
| `assign_openmp.csv` | Atribuição de clusters (N linhas) |
| `centroids_openmp.csv` | Centróides finais (K linhas) |
| `sse_history_openmp.csv` | SSE por iteração |
| `scalability_openmp.csv` | Resultados de escalabilidade (com `-a`) |

## Métricas Coletadas

- **Tempo Total**: tempo de execução completo
- **Tempo Assignment**: tempo na fase de atribuição
- **Tempo Update**: tempo na atualização de centróides
- **Speedup**: T_serial / T_paralelo
- **Eficiência**: Speedup / Número_de_threads
- **Throughput**: pontos processados por segundo

## Exemplo de Saída

```
╔══════════════════════════════════════════════════════════════╗
║         K-MEANS 1D - VERSÃO OPENMP (CPU PARALELO)           ║
╚══════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────┐
│ RESULTADOS                                                  │
├─────────────────────────────────────────────────────────────┤
│ Iterações:                    15                            │
│ SSE Final:            2847263.123456                        │
│ Tempo Total:              35.432 ms                         │
│ Threads utilizadas:            4                            │
│ Schedule:                 static                            │
│ Throughput:          42356789 pontos/s                      │
└─────────────────────────────────────────────────────────────┘
```

## Análise de Escalabilidade

Com a opção `-a`, o programa testa automaticamente diferentes configurações:

```
╔══════════════════════════════════════════════════════════════╗
║              ANÁLISE DE ESCALABILIDADE                       ║
╚══════════════════════════════════════════════════════════════╝

┌─────────┬──────────┬───────────┬─────────┬────────────┬─────────────┐
│ Threads │ Schedule │ Tempo(ms) │ Speedup │ Eficiência │ Throughput  │
├─────────┼──────────┼───────────┼─────────┼────────────┼─────────────┤
│    1    │ static   │   125.432 │   1.00x │   100.0%   │   1194853/s │
│    2    │ static   │    65.234 │   1.92x │    96.1%   │   2298765/s │
│    4    │ static   │    35.123 │   3.57x │    89.3%   │   4267890/s │
│    8    │ static   │    22.456 │   5.59x │    69.8%   │   6678901/s │
└─────────┴──────────┴───────────┴─────────┴────────────┴─────────────┘
```

## Estratégia de Paralelização

### Assignment (passo 1)
```c
#pragma omp parallel for reduction(+:sse) schedule(static)
for (int i = 0; i < N; i++) {
    // Cada thread processa parte dos pontos
    // SSE é somado automaticamente via reduction
}
```

### Update (passo 2)
```c
#pragma omp parallel
{
    // Arrays locais por thread (evita contenção)
    double* local_sum = calloc(K, sizeof(double));
    int* local_count = calloc(K, sizeof(int));
    
    #pragma omp for
    for (int i = 0; i < N; i++) { ... }
    
    // Combina resultados
    #pragma omp critical
    { ... }
}
```

## Observações para o Relatório

1. **Efeito do número de threads**: Speedup geralmente sub-linear devido a overhead de sincronização
2. **Impacto do schedule**: Para K-Means com carga uniforme, `static` geralmente é mais eficiente
3. **Overhead da seção crítica**: A fase de Update tem sincronização que limita escalabilidade
4. **Lei de Amdahl**: Parte sequencial limita speedup máximo teórico

## Autor

Projeto PCD - Programação Concorrente e Distribuída
