# K-Means 1D - Versão MPI (Memória Distribuída)

## Descrição

Versão paralelizada do K-Means usando **MPI** (Message Passing Interface) para sistemas de memória distribuída. Permite execução em múltiplos computadores conectados em rede (clusters).

## Requisitos

- MPI instalado (MPICH, OpenMPI, ou Microsoft MPI)
- Compilador C compatível (gcc, mpicc)

## Instalação do MPI

### Linux (Ubuntu/Debian)
```bash
sudo apt-get install mpich
# ou
sudo apt-get install openmpi-bin libopenmpi-dev
```

### Windows
Baixar e instalar Microsoft MPI:
https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi

### macOS
```bash
brew install mpich
# ou
brew install open-mpi
```

## Compilação

### Linux/macOS
```bash
mpicc -O3 -o kmeans_mpi kmeans_mpi.c -lm
```

### Windows (Microsoft MPI)
```bash
cl /O2 kmeans_mpi.c msmpi.lib /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" /link /LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"
```

## Execução

### Uso básico
```bash
# Executar com 4 processos
mpirun -np 4 ./kmeans_mpi

# Executar com 8 processos
mpirun -np 8 ./kmeans_mpi
```

### Todas as opções
```bash
mpirun -np <num_procs> ./kmeans_mpi [opções]

Opções:
  -d <arquivo>       Arquivo de dados
  -c <arquivo>       Arquivo de centróides
  --serial-time <ms> Tempo serial (para calcular speedup)
  -a                 Salvar análise de escalabilidade
  -h, --help         Mostrar ajuda
```

### Exemplos
```bash
# Execução básica
mpirun -np 4 ./kmeans_mpi

# Com arquivos específicos
mpirun -np 4 ./kmeans_mpi -d dados.csv -c centroides.csv

# Com speedup calculado
mpirun -np 4 ./kmeans_mpi --serial-time 125.5

# Salvando análise
mpirun -np 4 ./kmeans_mpi -a --serial-time 125.5
```

### Execução em múltiplas máquinas
```bash
# Criar arquivo hosts
echo "node1:4" > hosts
echo "node2:4" >> hosts
echo "node3:4" >> hosts

# Executar
mpirun -np 12 --hostfile hosts ./kmeans_mpi
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
| `assign_mpi.csv` | Atribuição de clusters (N linhas) |
| `centroids_mpi.csv` | Centróides finais (K linhas) |
| `sse_history_mpi.csv` | SSE por iteração |
| `scalability_mpi.csv` | Resultados de escalabilidade (com `-a`) |

## Métricas Coletadas

- **Tempo Total**: tempo completo de execução
- **Tempo Computação**: Assignment + Update
- **Tempo Comunicação**: Scatter + Allreduce + Gather
- **Tempo Allreduce**: tempo específico das reduções globais
- **Speedup**: T_serial / T_paralelo
- **Eficiência**: Speedup / Número_de_processos

## Exemplo de Saída

```
╔══════════════════════════════════════════════════════════════╗
║         K-MEANS 1D - VERSÃO MPI (DISTRIBUÍDO)               ║
╚══════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────┐
│ CONFIGURAÇÃO                                                │
├─────────────────────────────────────────────────────────────┤
│ Pontos (N):               100000                            │
│ Clusters (K):                  5                            │
│ Processos MPI:                 4                            │
│ Pontos/Processo:           25000 (aprox)                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ RESULTADOS                                                  │
├─────────────────────────────────────────────────────────────┤
│ Iterações:                    15                            │
│ SSE Final:            2847263.123456                        │
│ Tempo Total:              45.234 ms                         │
│ Tempo Computação:         30.123 ms (66.6%)                 │
│ Tempo Comunicação:        15.111 ms (33.4%)                 │
│   - Allreduce:            12.345 ms                         │
│ Speedup:                   2.77x                            │
│ Eficiência:               69.3%                             │
└─────────────────────────────────────────────────────────────┘
```

## Distribuição de Dados

```
┌─────────────────────────────────────────────────────────────┐
│                  N = 100.000 pontos                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Processo 0      Processo 1      Processo 2      Processo 3│
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  │  25.000  │    │  25.000  │    │  25.000  │    │  25.000  │
│  │  pontos  │    │  pontos  │    │  pontos  │    │  pontos  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘
│       ↓               ↓               ↓               ↓      │
│   Assignment      Assignment      Assignment      Assignment │
│       ↓               ↓               ↓               ↓      │
│       └───────────────┴───────────────┴───────────────┘      │
│                           │                                  │
│                      MPI_Allreduce                          │
│                     (combina SSE)                           │
│                           │                                  │
│       ┌───────────────┬───────────────┬───────────────┐      │
│       ↓               ↓               ↓               ↓      │
│    Update          Update          Update          Update   │
│  (centroides)    (centroides)    (centroides)    (centroides)│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Comunicações MPI Utilizadas

| Função | Uso | Custo |
|--------|-----|-------|
| `MPI_Bcast` | Broadcast de N, K, centróides | O(log P) |
| `MPI_Scatterv` | Distribuir pontos | O(N/P) |
| `MPI_Allreduce` | Somar SSE e acumuladores | O(K × log P) |
| `MPI_Gatherv` | Coletar atribuições | O(N/P) |

## Custo do Allreduce

O **MPI_Allreduce** é chamado 3 vezes por iteração:
1. Redução do SSE (1 double)
2. Redução das somas (K doubles)
3. Redução das contagens (K ints)

Para K grande ou muitos processos, isso pode ser gargalo.

## Análise de Escalabilidade

Para gerar curva de speedup, execute com diferentes números de processos:

```bash
# Limpar arquivo anterior
rm -f scalability_mpi.csv

# Executar com diferentes configurações
for np in 1 2 4 8 16; do
    mpirun -np $np ./kmeans_mpi -a --serial-time 125.5
done

# Resultado em scalability_mpi.csv
```

## Observações para o Relatório

1. **Overhead de comunicação**: Aumenta com número de processos
2. **Custo do Allreduce**: Cresce logaritmicamente com P
3. **Eficiência**: Diminui para N/P pequeno (overhead domina)
4. **Latência de rede**: Importante em clusters reais
5. **Balanceamento**: Distribuição uniforme assume carga igual

## Lei de Amdahl aplicada

Se 20% do código é sequencial (comunicação), speedup máximo teórico:
- 4 processos: 2.5x
- 8 processos: 3.3x
- 16 processos: 4.0x
- ∞ processos: 5.0x

## Solução de Problemas

### Erro "mpirun not found"
```bash
export PATH=$PATH:/usr/lib64/mpich/bin
```

### Erro de conexão entre nós
```bash
# Verificar SSH sem senha
ssh-keygen -t rsa
ssh-copy-id user@node2
```

### Timeout em Allreduce
Verifique conectividade de rede e firewall.

## Autor

Projeto PCD - Programação Concorrente e Distribuída
