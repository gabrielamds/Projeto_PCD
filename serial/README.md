# K-Means 1D - Versão Serial (Sequencial)

## Descrição

Esta é a versão **baseline** (referência) do algoritmo K-Means 1D. É uma implementação sequencial que serve como base para comparação de desempenho com as versões paralelas.

## Compilação

### Linux/macOS
```bash
gcc -O3 -o kmeans_serial kmeans_serial.c -lm
```

### Windows (MinGW)
```bash
gcc -O3 -o kmeans_serial.exe kmeans_serial.c
```

### Windows (Visual Studio)
```bash
cl /O2 kmeans_serial.c
```

## Execução

```bash
# Usando arquivos padrão (../data/dados.csv e ../data/centroides_iniciais.csv)
./kmeans_serial

# Especificando arquivos de entrada
./kmeans_serial <arquivo_dados.csv> <arquivo_centroides.csv>

# Exemplo
./kmeans_serial ../data/dados.csv ../data/centroides_iniciais.csv
```

## Parâmetros Fixos

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `MAX_ITER` | 100 | Número máximo de iterações |
| `EPS` | 1e-6 | Tolerância para convergência |
| `SEED` | 42 | Semente para reprodutibilidade |

## Arquivos de Saída

| Arquivo | Descrição |
|---------|-----------|
| `assign_serial.csv` | Índice do cluster de cada ponto (N linhas) |
| `centroids_serial.csv` | Coordenadas finais dos centróides (K linhas) |
| `sse_history_serial.csv` | SSE de cada iteração (para análise de convergência) |

## Métricas Coletadas

- **Tempo Total**: tempo de execução completo
- **Tempo Assignment**: tempo gasto na fase de atribuição de pontos
- **Tempo Update**: tempo gasto na atualização de centróides
- **Throughput**: pontos processados por segundo
- **SSE por iteração**: para validação da convergência

## Exemplo de Saída

```
╔══════════════════════════════════════════════════════════════╗
║          K-MEANS 1D - VERSÃO SERIAL (SEQUENCIAL)            ║
║                  Projeto PCD - Baseline                      ║
╚══════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────┐
│ CONFIGURAÇÃO                                                │
├─────────────────────────────────────────────────────────────┤
│ Pontos (N):               100000                            │
│ Clusters (K):                  5                            │
│ Max Iterações:               100                            │
│ Epsilon (eps):           1.00e-06                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ RESULTADOS                                                  │
├─────────────────────────────────────────────────────────────┤
│ Iterações:                    15                            │
│ SSE Final:            2847263.123456                        │
│ Tempo Total:             125.432 ms                         │
│ Throughput:          11948532 pontos/segundo                │
└─────────────────────────────────────────────────────────────┘
```

## Complexidade

- **Assignment**: O(N × K) por iteração
- **Update**: O(N + K) por iteração
- **Total**: O(I × N × K), onde I = número de iterações

## Uso para Benchmark

Este código serve como referência para calcular:

- **Speedup** = Tempo_Serial / Tempo_Paralelo
- **Eficiência** = Speedup / Número_de_Processadores

## Autor

Projeto PCD - Programação Concorrente e Distribuída
