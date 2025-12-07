# K-Means 1D - Versão CUDA (GPU)

## Descrição

Versão paralelizada do K-Means usando **CUDA** para GPUs NVIDIA com paralelização massiva.

## Requisitos

- GPU NVIDIA com Compute Capability >= 3.0
- CUDA Toolkit instalado (versão 10.0 ou superior recomendada)
- Driver NVIDIA atualizado

## Compilação

### Linux/macOS
```bash
nvcc -O3 -arch=sm_50 -o kmeans_cuda kmeans_cuda.cu
```

### Windows
```bash
nvcc -O3 -arch=sm_50 -o kmeans_cuda.exe kmeans_cuda.cu
```

### Notas sobre arch
- `-arch=sm_50`: GPUs Maxwell (GTX 9xx)
- `-arch=sm_60`: GPUs Pascal (GTX 10xx)
- `-arch=sm_70`: GPUs Volta (Tesla V100)
- `-arch=sm_75`: GPUs Turing (RTX 20xx)
- `-arch=sm_80`: GPUs Ampere (RTX 30xx)
- `-arch=sm_86`: GPUs Ampere (RTX 30xx mobile)
- `-arch=sm_89`: GPUs Ada Lovelace (RTX 40xx)

## Execução

### Uso básico
```bash
# Usar block size padrão (256)
./kmeans_cuda

# Especificar block size
./kmeans_cuda -b 512
```

### Todas as opções
```bash
./kmeans_cuda [opções]

Opções:
  -b <num>           Block size (padrão: 256)
  -a                 Executar análise de block size
  --serial-time <ms> Tempo serial (para calcular speedup)
  -d <arquivo>       Arquivo de dados
  -c <arquivo>       Arquivo de centróides
  -h, --help         Mostrar ajuda
```

### Exemplos
```bash
# Execução básica
./kmeans_cuda

# Com block size específico
./kmeans_cuda -b 512

# Análise completa de block size
./kmeans_cuda -a

# Com speedup vs serial
./kmeans_cuda -b 256 --serial-time 125.5
```

## Parâmetros Fixos

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `MAX_ITER` | 100 | Número máximo de iterações |
| `EPS` | 1e-6 | Tolerância para convergência |
| `SEED` | 42 | Semente para reprodutibilidade |

## Block Sizes Recomendados

| Block Size | Uso recomendado |
|------------|-----------------|
| 64 | GPUs antigas, K pequeno |
| 128 | Uso geral |
| 256 | **Recomendado** - bom equilíbrio |
| 512 | N muito grande |
| 1024 | Máximo ocupação |

## Arquivos de Saída

| Arquivo | Descrição |
|---------|-----------|
| `assign_cuda.csv` | Atribuição de clusters (N linhas) |
| `centroids_cuda.csv` | Centróides finais (K linhas) |
| `sse_history_cuda.csv` | SSE por iteração |
| `blocksize_cuda.csv` | Resultados de análise de block size (com `-a`) |

## Métricas Coletadas

- **Tempo Total**: tempo completo incluindo transferências
- **Tempo Kernels**: tempo apenas de execução dos kernels
- **Tempo Transferência H→D**: tempo de cópia Host para Device
- **Tempo Transferência D→H**: tempo de cópia Device para Host
- **Throughput**: pontos processados por segundo

## Exemplo de Saída

```
╔══════════════════════════════════════════════════════════════╗
║            K-MEANS 1D - VERSÃO CUDA (GPU)                   ║
╚══════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────┐
│ INFORMAÇÕES DA GPU                                          │
├─────────────────────────────────────────────────────────────┤
│ Nome: NVIDIA GeForce RTX 3080                               │
│ Compute Capability: 8.6                                     │
│ Multiprocessadores: 68                                      │
│ Memória Global: 10.00 GB                                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ RESULTADOS                                                  │
├─────────────────────────────────────────────────────────────┤
│ Iterações:                    15                            │
│ SSE Final:            2847263.123456                        │
│ Tempo Total:               5.234 ms                         │
│ Tempo Kernels:             3.456 ms (66.0%)                 │
│ Tempo Transferência:       1.234 ms (23.6%)                 │
│ Throughput:         286543210 pontos/s                      │
└─────────────────────────────────────────────────────────────┘
```

## Análise de Block Size

Com a opção `-a`:

```
╔══════════════════════════════════════════════════════════════╗
║              ANÁLISE DE BLOCK SIZE                           ║
╚══════════════════════════════════════════════════════════════╝

┌────────────┬───────────┬───────────┬───────────┬───────────┬─────────┐
│ Block Size │ Grid Size │ Total(ms) │Kernel(ms) │Transfer(ms)│ Speedup │
├────────────┼───────────┼───────────┼───────────┼───────────┼─────────┤
│     32     │     3125  │     8.234 │     6.123 │     1.456 │   15.2x │
│     64     │     1563  │     6.456 │     4.890 │     1.234 │   19.4x │
│    128     │      782  │     5.678 │     4.123 │     1.234 │   22.1x │
│    256     │      391  │     5.234 │     3.456 │     1.234 │   24.0x │
│    512     │      196  │     5.456 │     3.678 │     1.234 │   23.0x │
│   1024     │       98  │     5.890 │     4.012 │     1.234 │   21.3x │
└────────────┴───────────┴───────────┴───────────┴───────────┴─────────┘
```

## Arquitetura CUDA

### Kernels Implementados

1. **assignment_kernel**: Cada thread processa um ponto
2. **reduce_sse_kernel**: Redução paralela para calcular SSE total
3. **accumulate_kernel**: Acumula somas por cluster com atomicAdd
4. **update_centroids_kernel**: Atualiza centróides

### Hierarquia de Memória

```
┌─────────────────────────────────────────┐
│ Memória Global (lenta, grande)          │
│   - X[], C[], assign[]                  │
├─────────────────────────────────────────┤
│ Memória Compartilhada (rápida, pequena) │
│   - Usada na redução de SSE             │
├─────────────────────────────────────────┤
│ Registradores (mais rápida)             │
│   - Variáveis locais                    │
└─────────────────────────────────────────┘
```

## Observações para o Relatório

1. **Custo de transferência**: Pode dominar para N pequeno
2. **Block size ótimo**: Geralmente 128-256 para K-Means
3. **Ocupação**: Mais threads nem sempre = mais rápido
4. **atomicAdd**: Pode ser gargalo para K grande
5. **Comparação com CPU**: Speedup significativo para N > 10.000

## Solução de Problemas

### Erro "no CUDA-capable device"
```bash
nvidia-smi  # Verificar se GPU é detectada
```

### Erro de compilação com arch
```bash
nvcc --list-gpu-arch  # Ver arquiteturas suportadas
```

### Memória insuficiente
Reduza N ou use GPU com mais memória.

## Autor

Projeto PCD - Programação Concorrente e Distribuída
