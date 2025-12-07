# 🎯 Projeto K-Means 1D - Programação Concorrente e Distribuída

Implementação do algoritmo K-Means para dados unidimensionais com **paralelização progressiva** usando OpenMP, CUDA e MPI.

## 📖 Descrição

Este projeto implementa o algoritmo de clustering K-Means para dados 1D em 4 versões:

1. **Serial** - Implementação baseline para comparação
2. **OpenMP** - Paralelização em CPU com memória compartilhada
3. **CUDA** - Paralelização massiva em GPU NVIDIA
4. **MPI** - Paralelização distribuída para clusters

## 📁 Estrutura do Projeto

```
Projeto_PCD/
├── serial/                     # Versão sequencial
│   ├── kmeans_serial.c
│   └── README.md
├── openmp/                     # Versão OpenMP (CPU paralela)
│   ├── kmeans_openmp.c
│   └── README.md
├── cuda/                       # Versão CUDA (GPU)
│   ├── kmeans_cuda.cu
│   └── README.md
├── mpi/                        # Versão MPI (distribuída)
│   ├── kmeans_mpi.c
│   └── README.md
├── data/                       # Dados de entrada
│   ├── generate_data.c         # Gerador de dados
│   ├── dados.csv               # Dataset principal
│   └── centroides_iniciais.csv
├── results/                    # Resultados dos benchmarks
├── run_benchmark.ps1           # Script de benchmark (Windows)
├── run_benchmark.sh            # Script de benchmark (Linux)
└── README.md                   # Este arquivo
```

## ⚙️ Parâmetros Padrão

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `N` | 100.000 | Número de pontos |
| `K` | 5 | Número de clusters |
| `MAX_ITER` | 100 | Iterações máximas |
| `EPS` | 1e-6 | Convergência (tolerância) |
| `SEED` | 42 | Semente para reprodutibilidade |

## 🔧 Pré-requisitos

### Mínimos
- **GCC** 4.9+ (com suporte a C99)
- **Make** (opcional)

### Para OpenMP
- **GCC com OpenMP** (-fopenmp)

### Para CUDA
- **NVIDIA GPU** (Compute Capability 3.0+)
- **CUDA Toolkit** 10.0+

### Para MPI
- **OpenMPI** 3.0+ ou **MS-MPI** (Windows)

### Instalação (Linux/Ubuntu)
```bash
# GCC com suporte a OpenMP
sudo apt-get install gcc

# MPI
sudo apt-get install mpich
# ou
sudo apt-get install openmpi-bin libopenmpi-dev

# CUDA (opcional - requer GPU NVIDIA)
# Instalar CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
```

### Instalação (Windows)
- **MinGW-w64** ou **Visual Studio** com suporte C
- **Microsoft MPI**: https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-downloads

## 🚀 Compilação

### Windows (PowerShell)

```powershell
# Serial
cd serial; gcc -O3 -o kmeans_serial.exe kmeans_serial.c -lm

# OpenMP
cd openmp; gcc -O3 -fopenmp -o kmeans_openmp.exe kmeans_openmp.c -lm

# CUDA (requer NVCC)
cd cuda; nvcc -O3 -o kmeans_cuda.exe kmeans_cuda.cu

# MPI (requer MS-MPI)
cd mpi; mpicc -O3 -o kmeans_mpi.exe kmeans_mpi.c -lm

# Gerador de dados
cd data; gcc -O3 -o generate_data.exe generate_data.c -lm
```

### Linux/macOS

```bash
# Serial
cd serial && gcc -O3 -o kmeans_serial kmeans_serial.c -lm

# OpenMP
cd openmp && gcc -O3 -fopenmp -o kmeans_openmp kmeans_openmp.c -lm

# CUDA
cd cuda && nvcc -O3 -o kmeans_cuda kmeans_cuda.cu

# MPI
cd mpi && mpicc -O3 -o kmeans_mpi kmeans_mpi.c -lm
```

## 📊 Execução do Benchmark

### Windows
```powershell
.\run_benchmark.ps1
.\run_benchmark.ps1 -N 500000 -K 10
```

### Linux
```bash
./run_benchmark.sh
./run_benchmark.sh -n 500000 -k 10
```

## 📈 Métricas Coletadas

### Desempenho
- **Tempo Total** (ms)
- **Speedup** = T_serial / T_paralelo
- **Eficiência** = Speedup / num_threads
- **Throughput** = N / tempo (pontos/segundo)

### Qualidade do Clustering
- **SSE** (Sum of Squared Errors) por iteração
- **Convergência** (número de iterações até EPS)

### Análise de Gargalos
- **OpenMP**: Tempo por schedule (static, dynamic, guided)
- **CUDA**: Tempo kernel vs. transferência de dados
- **MPI**: Tempo computação vs. comunicação (Allreduce breakdown)

## 📋 Formato dos Arquivos de Entrada

### dados.csv
Arquivo CSV com N valores (1 por linha, sem cabeçalho):
```
10.234567
25.891234
15.456789
...
```

### centroides_iniciais.csv
Arquivo CSV com K centróides iniciais:
```
10.0
30.0
50.0
```

## 📐 Algoritmo K-Means 1D

### Fase de Atribuição (Assignment)
Para cada ponto $x_i$, encontrar o centroide mais próximo:
$$c_i = \arg\min_j |x_i - \mu_j|$$

### Fase de Atualização (Update)
Para cada cluster $j$, recalcular a média:
$$\mu_j = \frac{1}{|C_j|} \sum_{i \in C_j} x_i$$

### Métrica SSE (Sum of Squared Errors)
$$SSE = \sum_{i=1}^{N} (x_i - \mu_{c_i})^2$$

## 📤 Arquivos de Saída

| Arquivo | Descrição |
|---------|-----------|
| `assign_*.csv` | N linhas com índice do cluster de cada ponto |
| `centroids_*.csv` | K linhas com coordenadas finais dos centróides |
| `sse_history_*.csv` | SSE por iteração (para análise de convergência) |
| `scalability_*.csv` | Resultados de escalabilidade |

## 📋 Exemplo de Saída

```
============================================================
                  RELATÓRIO K-MEANS SERIAL
============================================================
Dados: N=100000, K=5
------------------------------------------------------------

DESEMPENHO:
  Tempo Total:           45.23 ms
  Tempo Assignment:      32.15 ms (71.1%)
  Tempo Update:          13.08 ms (28.9%)
  Throughput:            2211283 pontos/seg

CONVERGÊNCIA:
  Iterações:             47
  SSE Final:             125847.32

ARQUIVOS GERADOS:
  Assignments:     assign_serial.csv
  Centroids:       centroids_serial.csv
  SSE History:     sse_history_serial.csv
============================================================
```

## 📚 Documentação por Versão

Cada subpasta contém um README detalhado:

- [Serial - README](serial/README.md)
- [OpenMP - README](openmp/README.md)
- [CUDA - README](cuda/README.md)
- [MPI - README](mpi/README.md)

## 🧪 Reprodutibilidade

Todos os experimentos são reproduzíveis:
- **Seed fixa** (42) para geração de dados
- **Mesmos centroides iniciais** para todas as versões
- **Determinismo** garantido nas versões serial e MPI

⚠️ **Nota**: CUDA e OpenMP podem ter variações mínimas devido a ordem de operações em ponto flutuante.

## ✅ Validação de Corretude

Para validar que as versões paralelas produzem resultados corretos:

```powershell
# Comparar SSE final entre versões (Windows)
Get-Content results\output_*.txt | Select-String "SSE Final"
```

```bash
# Comparar SSE final entre versões (Linux)
grep "SSE Final" results/output_*.txt
```

O SSE final deve ser idêntico (ou muito próximo) entre todas as versões.

## 👤 Autor

- Projeto desenvolvido para a disciplina de **Programação Concorrente e Distribuída**

## 📖 Referências

1. Lloyd, S. (1982). "Least squares quantization in PCM". IEEE Transactions on Information Theory.
2. OpenMP Architecture Review Board. "OpenMP Application Programming Interface".
3. NVIDIA Corporation. "CUDA C++ Programming Guide".
4. Message Passing Interface Forum. "MPI: A Message-Passing Interface Standard".

## 🧪 Algoritmo K-Means

### Pseudocódigo

```
Entrada: X[N] (pontos), C[K] (centróides iniciais)
Saída: C[K] (centróides finais), assign[N] (atribuições)

Repetir até convergência (max_iter ou SSE estável):
    1. ASSIGNMENT:
       Para cada ponto X[i]:
           Encontrar centróide mais próximo: argmin_c (X[i] - C[c])²
           assign[i] = c
           SSE += (X[i] - C[c])²
    
    2. UPDATE:
       Para cada cluster c:
           C[c] = média dos pontos atribuídos a c
           Se cluster vazio: C[c] = X[0]
```

### Critério de Parada
- Número máximo de iterações (`MAX_ITER = 1000`)
- Variação do SSE menor que epsilon (`EPS = 1e-6`)

## ⚡ Estratégias de Paralelização

### OpenMP (Memória Compartilhada)
- **Assignment**: `#pragma omp parallel for reduction(+:sse)`
- **Update**: Arrays locais por thread + seção crítica para combinar

### CUDA (GPU)
- **Assignment**: 1 thread por ponto, kernel paralelo
- **Update**: `atomicAdd` para acumular somas por cluster
- **SSE**: Redução paralela em memória compartilhada

### MPI (Memória Distribuída)
- **Distribuição**: `MPI_Scatterv` divide pontos entre processos
- **Assignment**: Cada processo processa sua porção
- **Update**: `MPI_Allreduce` para combinar somas globalmente
- **Coleta**: `MPI_Gatherv` junta resultados no processo 0

## 📊 Comparação de Desempenho

| Versão | N=100k, K=5 | Speedup |
|--------|-------------|---------|
| Naive (1 thread) | ~125 ms | 1.0x |
| OpenMP (4 threads) | ~35 ms | ~3.5x |
| OpenMP (8 threads) | ~20 ms | ~6.0x |
| CUDA (GPU) | ~5 ms | ~25x |
| MPI (4 processos) | ~40 ms | ~3.0x |

*Valores aproximados, dependem do hardware*

## 🔍 Métricas

- **SSE (Sum of Squared Errors)**: Soma dos erros quadráticos - mede qualidade do clustering
- **Tempo (ms)**: Tempo total de execução em milissegundos
- **Iterações**: Número de iterações até convergência

## 👨‍💻 Autor

Projeto desenvolvido para a disciplina de **Programação Concorrente e Distribuída**.

## 📝 Licença

Este projeto é para fins educacionais.
