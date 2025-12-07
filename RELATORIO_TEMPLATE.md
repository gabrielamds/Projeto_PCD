# Relatório - Projeto K-Means 1D Paralelo

**Disciplina:** Programação Concorrente e Distribuída  
**Aluno:** [Seu Nome]  
**Data:** [Data]

---

## 1. Introdução

### 1.1 Objetivo
Este relatório apresenta a implementação e análise de desempenho do algoritmo K-Means para dados unidimensionais, utilizando diferentes paradigmas de paralelização: OpenMP (memória compartilhada), CUDA (GPU) e MPI (memória distribuída).

### 1.2 Motivação
O algoritmo K-Means é um dos métodos de clustering mais utilizados em aprendizado de máquina. Sua natureza iterativa e as operações independentes sobre cada ponto o tornam um excelente candidato para paralelização.

---

## 2. Ambiente de Execução

### 2.1 Hardware

| Componente | Especificação |
|------------|---------------|
| **CPU** | [Ex: Intel Core i7-10700 @ 2.90GHz, 8 cores/16 threads] |
| **RAM** | [Ex: 16GB DDR4 3200MHz] |
| **GPU** | [Ex: NVIDIA GeForce RTX 3060, 12GB VRAM, 3584 CUDA cores] |
| **Armazenamento** | [Ex: SSD NVMe 500GB] |

### 2.2 Software

| Software | Versão |
|----------|--------|
| **Sistema Operacional** | [Ex: Windows 11 / Ubuntu 22.04] |
| **Compilador C** | [Ex: GCC 11.4.0] |
| **CUDA Toolkit** | [Ex: 12.1] |
| **MPI** | [Ex: OpenMPI 4.1.4 / MS-MPI 10.1] |

### 2.3 Parâmetros do Experimento

| Parâmetro | Valor |
|-----------|-------|
| N (pontos) | 100.000 |
| K (clusters) | 5 |
| MAX_ITER | 100 |
| EPS (tolerância) | 1e-6 |
| SEED | 42 |

---

## 3. Resultados de Desempenho

### 3.1 Comparação de Tempos de Execução

| Versão | Tempo (ms) | Speedup | Eficiência |
|--------|-----------|---------|------------|
| Serial | [tempo] | 1.00x | 100% |
| OpenMP (4 threads) | [tempo] | [x]x | [%]% |
| OpenMP (8 threads) | [tempo] | [x]x | [%]% |
| CUDA | [tempo] | [x]x | - |
| MPI (4 processos) | [tempo] | [x]x | [%]% |

### 3.2 Gráfico de Speedup

```
[Inserir gráfico de barras ou linha mostrando speedup vs. número de threads/processos]
```

### 3.3 Análise de Escalabilidade - OpenMP

| Threads | Tempo (ms) | Speedup | Eficiência |
|---------|-----------|---------|------------|
| 1 | [tempo] | 1.00x | 100% |
| 2 | [tempo] | [x]x | [%]% |
| 4 | [tempo] | [x]x | [%]% |
| 8 | [tempo] | [x]x | [%]% |
| 16 | [tempo] | [x]x | [%]% |

### 3.4 Análise de Block Size - CUDA

| Block Size | Tempo Kernel (ms) | Tempo Transferência (ms) | Tempo Total (ms) |
|------------|------------------|-------------------------|-----------------|
| 32 | [tempo] | [tempo] | [tempo] |
| 64 | [tempo] | [tempo] | [tempo] |
| 128 | [tempo] | [tempo] | [tempo] |
| 256 | [tempo] | [tempo] | [tempo] |
| 512 | [tempo] | [tempo] | [tempo] |

### 3.5 Breakdown de Comunicação - MPI

| Processos | Computação (ms) | Comunicação (ms) | % Comunicação |
|-----------|-----------------|------------------|---------------|
| 2 | [tempo] | [tempo] | [%]% |
| 4 | [tempo] | [tempo] | [%]% |
| 8 | [tempo] | [tempo] | [%]% |

---

## 4. Validação de Corretude

### 4.1 Comparação de SSE Final

| Versão | SSE Final | Diferença vs Serial |
|--------|-----------|---------------------|
| Serial | [valor] | - |
| OpenMP | [valor] | [diferença] |
| CUDA | [valor] | [diferença] |
| MPI | [valor] | [diferença] |

### 4.2 Análise de Convergência

Todas as versões convergiram em **[N] iterações**, demonstrando a corretude da implementação.

```
[Inserir gráfico de SSE vs. Iteração mostrando convergência idêntica]
```

---

## 5. Análise de Gargalos

### 5.1 Versão Serial
- **Gargalo principal:** [Ex: Fase de assignment ocupa 71% do tempo]
- **Throughput:** [X] pontos/segundo

### 5.2 Versão OpenMP
- **Overhead de sincronização:** [descrição]
- **Melhor schedule encontrado:** [static/dynamic/guided]
- **Limitação observada:** [Ex: falso compartilhamento, contenção de cache]

### 5.3 Versão CUDA
- **Overhead de transferência:** [X]% do tempo total
- **Ocupação da GPU:** [X]%
- **Block size ótimo:** [valor]
- **Limitação:** [Ex: largura de banda de memória, latência de transferência]

### 5.4 Versão MPI
- **Overhead de comunicação:** [X]% do tempo total
- **Tempo em Allreduce:** [X] ms ([X]% da comunicação)
- **Escalabilidade:** [forte/fraca - descrição]

---

## 6. Discussão

### 6.1 Comparação entre Paradigmas

| Paradigma | Vantagens | Desvantagens |
|-----------|-----------|--------------|
| OpenMP | Facilidade de implementação, baixo overhead | Limitado ao número de cores |
| CUDA | Massivamente paralelo, alto throughput | Overhead de transferência, programação complexa |
| MPI | Escalabilidade distribuída | Overhead de comunicação, complexidade de implementação |

### 6.2 Quando usar cada versão?

- **Serial:** Datasets pequenos (< 10.000 pontos)
- **OpenMP:** Datasets médios, execução em única máquina
- **CUDA:** Datasets grandes, GPU disponível
- **MPI:** Datasets muito grandes, cluster disponível

### 6.3 Otimizações Futuras

1. [Ex: Uso de memória compartilhada em CUDA para redução]
2. [Ex: Overlapping de comunicação e computação em MPI]
3. [Ex: Vetorização SIMD na versão serial/OpenMP]

---

## 7. Conclusão

[Resumo dos principais resultados e conclusões do projeto]

---

## 8. Referências Bibliográficas

1. Lloyd, S. (1982). "Least squares quantization in PCM". *IEEE Transactions on Information Theory*, 28(2), 129-137.

2. OpenMP Architecture Review Board. (2021). *OpenMP Application Programming Interface Version 5.1*.

3. NVIDIA Corporation. (2023). *CUDA C++ Programming Guide*. https://docs.nvidia.com/cuda/cuda-c-programming-guide/

4. Message Passing Interface Forum. (2021). *MPI: A Message-Passing Interface Standard Version 4.0*.

5. [Adicionar outras referências utilizadas]

---

## Apêndice A - Instruções de Reprodução

### Compilação
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

### Geração de Dados
```bash
cd data && gcc -O3 -o generate_data generate_data.c -lm
./generate_data -n 100000 -k 5 -s 42
```

### Execução do Benchmark
```bash
# Windows
.\run_benchmark.ps1

# Linux
./run_benchmark.sh
```

---

## Apêndice B - Código Fonte

O código fonte completo está disponível nas seguintes pastas:
- `serial/` - Versão sequencial
- `openmp/` - Versão OpenMP
- `cuda/` - Versão CUDA
- `mpi/` - Versão MPI
- `data/` - Gerador de dados
