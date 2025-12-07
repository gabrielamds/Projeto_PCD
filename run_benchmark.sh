#!/bin/bash
# ============================================================================
# SCRIPT DE BENCHMARK - K-MEANS PCD
# ============================================================================
# Executa todas as versões e gera relatório comparativo
#
# Uso: ./run_benchmark.sh [N] [K]
#   N - número de pontos (padrão: 100000)
#   K - número de clusters (padrão: 5)
# ============================================================================

set -e

# Parâmetros
N=${1:-100000}
K=${2:-5}
SEED=42

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              BENCHMARK K-MEANS PCD                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo -e "Configuração: ${BLUE}N=$N, K=$K, SEED=$SEED${NC}"
echo ""

# Diretórios
ROOT_DIR=$(pwd)
DATA_DIR="$ROOT_DIR/data"
SERIAL_DIR="$ROOT_DIR/serial"
OPENMP_DIR="$ROOT_DIR/openmp"
CUDA_DIR="$ROOT_DIR/cuda"
MPI_DIR="$ROOT_DIR/mpi"
RESULTS_DIR="$ROOT_DIR/results"

# Criar diretório de resultados
mkdir -p "$RESULTS_DIR"

# ============================================================================
# PASSO 1: GERAR DADOS
# ============================================================================
echo -e "${YELLOW}[1/5] Gerando dados de teste...${NC}"
cd "$DATA_DIR"

if [ ! -f generate_data ]; then
    echo "Compilando gerador de dados..."
    gcc -O3 -o generate_data generate_data.c -lm
fi

./generate_data -n $N -k $K -s $SEED
echo -e "${GREEN}✓ Dados gerados${NC}"
echo ""

# ============================================================================
# PASSO 2: VERSÃO SERIAL
# ============================================================================
echo -e "${YELLOW}[2/5] Executando versão SERIAL...${NC}"
cd "$SERIAL_DIR"

if [ ! -f kmeans_serial ]; then
    echo "Compilando versão serial..."
    gcc -O3 -o kmeans_serial kmeans_serial.c -lm
fi

./kmeans_serial -d "$DATA_DIR/dados.csv" -c "$DATA_DIR/centroides_iniciais.csv" | tee "$RESULTS_DIR/output_serial.txt"

# Extrair tempo serial para speedup
SERIAL_TIME=$(grep "Tempo Total:" "$RESULTS_DIR/output_serial.txt" | awk '{print $3}')
echo -e "${GREEN}✓ Serial concluído - Tempo: ${SERIAL_TIME}ms${NC}"
echo ""

# Copiar resultados
cp assign_serial.csv "$RESULTS_DIR/"
cp centroids_serial.csv "$RESULTS_DIR/"
cp sse_history_serial.csv "$RESULTS_DIR/"

# ============================================================================
# PASSO 3: VERSÃO OPENMP
# ============================================================================
echo -e "${YELLOW}[3/5] Executando versão OPENMP...${NC}"
cd "$OPENMP_DIR"

if [ ! -f kmeans_openmp ]; then
    echo "Compilando versão OpenMP..."
    gcc -O3 -fopenmp -o kmeans_openmp kmeans_openmp.c -lm
fi

# Executar com análise de escalabilidade
./kmeans_openmp -d "$DATA_DIR/dados.csv" -c "$DATA_DIR/centroides_iniciais.csv" -a --serial-time $SERIAL_TIME | tee "$RESULTS_DIR/output_openmp.txt"

echo -e "${GREEN}✓ OpenMP concluído${NC}"
echo ""

# Copiar resultados
cp assign_openmp.csv "$RESULTS_DIR/"
cp centroids_openmp.csv "$RESULTS_DIR/"
cp sse_history_openmp.csv "$RESULTS_DIR/"
cp scalability_openmp.csv "$RESULTS_DIR/"

# ============================================================================
# PASSO 4: VERSÃO CUDA (se disponível)
# ============================================================================
echo -e "${YELLOW}[4/5] Verificando CUDA...${NC}"
cd "$CUDA_DIR"

if command -v nvcc &> /dev/null; then
    echo "CUDA encontrado, compilando..."
    
    if [ ! -f kmeans_cuda ]; then
        nvcc -O3 -arch=sm_50 -o kmeans_cuda kmeans_cuda.cu 2>/dev/null || {
            echo -e "${RED}Erro na compilação CUDA - tentando arch genérica${NC}"
            nvcc -O3 -o kmeans_cuda kmeans_cuda.cu 2>/dev/null || {
                echo -e "${RED}✗ Falha na compilação CUDA${NC}"
            }
        }
    fi
    
    if [ -f kmeans_cuda ]; then
        ./kmeans_cuda -d "$DATA_DIR/dados.csv" -c "$DATA_DIR/centroides_iniciais.csv" -a --serial-time $SERIAL_TIME | tee "$RESULTS_DIR/output_cuda.txt"
        
        # Copiar resultados
        cp assign_cuda.csv "$RESULTS_DIR/" 2>/dev/null || true
        cp centroids_cuda.csv "$RESULTS_DIR/" 2>/dev/null || true
        cp sse_history_cuda.csv "$RESULTS_DIR/" 2>/dev/null || true
        cp blocksize_cuda.csv "$RESULTS_DIR/" 2>/dev/null || true
        
        echo -e "${GREEN}✓ CUDA concluído${NC}"
    fi
else
    echo -e "${YELLOW}⚠ CUDA não disponível - pulando${NC}"
fi
echo ""

# ============================================================================
# PASSO 5: VERSÃO MPI
# ============================================================================
echo -e "${YELLOW}[5/5] Executando versão MPI...${NC}"
cd "$MPI_DIR"

if command -v mpicc &> /dev/null; then
    if [ ! -f kmeans_mpi ]; then
        echo "Compilando versão MPI..."
        mpicc -O3 -o kmeans_mpi kmeans_mpi.c -lm
    fi
    
    # Limpar arquivo de escalabilidade anterior
    rm -f scalability_mpi.csv
    
    # Testar com diferentes números de processos
    for np in 1 2 4; do
        echo "Executando com $np processos..."
        mpirun -np $np ./kmeans_mpi -d "$DATA_DIR/dados.csv" -c "$DATA_DIR/centroides_iniciais.csv" -a --serial-time $SERIAL_TIME 2>/dev/null || true
    done
    
    echo ""
    mpirun -np 4 ./kmeans_mpi -d "$DATA_DIR/dados.csv" -c "$DATA_DIR/centroides_iniciais.csv" --serial-time $SERIAL_TIME | tee "$RESULTS_DIR/output_mpi.txt"
    
    # Copiar resultados
    cp assign_mpi.csv "$RESULTS_DIR/" 2>/dev/null || true
    cp centroids_mpi.csv "$RESULTS_DIR/" 2>/dev/null || true
    cp sse_history_mpi.csv "$RESULTS_DIR/" 2>/dev/null || true
    cp scalability_mpi.csv "$RESULTS_DIR/" 2>/dev/null || true
    
    echo -e "${GREEN}✓ MPI concluído${NC}"
else
    echo -e "${YELLOW}⚠ MPI não disponível - pulando${NC}"
fi
echo ""

# ============================================================================
# VALIDAÇÃO DE CORRETUDE
# ============================================================================
echo -e "${YELLOW}Validando corretude (comparando SSE)...${NC}"
cd "$RESULTS_DIR"

echo ""
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│ VALIDAÇÃO DE CORRETUDE                                     │"
echo "├─────────────────────────────────────────────────────────────┤"

# Extrair SSE de cada versão
SSE_SERIAL=$(grep "SSE Final:" output_serial.txt 2>/dev/null | awk '{print $3}' || echo "N/A")
SSE_OPENMP=$(grep "SSE Final:" output_openmp.txt 2>/dev/null | awk '{print $3}' || echo "N/A")
SSE_CUDA=$(grep "SSE Final:" output_cuda.txt 2>/dev/null | awk '{print $3}' || echo "N/A")
SSE_MPI=$(grep "SSE Final:" output_mpi.txt 2>/dev/null | awk '{print $3}' || echo "N/A")

printf "│ Serial:  SSE = %-20s                     │\n" "$SSE_SERIAL"
printf "│ OpenMP:  SSE = %-20s                     │\n" "$SSE_OPENMP"
printf "│ CUDA:    SSE = %-20s                     │\n" "$SSE_CUDA"
printf "│ MPI:     SSE = %-20s                     │\n" "$SSE_MPI"
echo "└─────────────────────────────────────────────────────────────┘"

# ============================================================================
# RESUMO DE DESEMPENHO
# ============================================================================
echo ""
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│ RESUMO DE DESEMPENHO                                       │"
echo "├─────────────────────────────────────────────────────────────┤"

TIME_SERIAL=$(grep "Tempo Total:" output_serial.txt 2>/dev/null | awk '{print $3}' || echo "N/A")
TIME_OPENMP=$(grep "Tempo Total:" output_openmp.txt 2>/dev/null | awk '{print $3}' || echo "N/A")
TIME_CUDA=$(grep "Tempo Total:" output_cuda.txt 2>/dev/null | awk '{print $3}' || echo "N/A")
TIME_MPI=$(grep "Tempo Total:" output_mpi.txt 2>/dev/null | awk '{print $3}' || echo "N/A")

printf "│ Serial:  %-10s ms                                    │\n" "$TIME_SERIAL"
printf "│ OpenMP:  %-10s ms                                    │\n" "$TIME_OPENMP"
printf "│ CUDA:    %-10s ms                                    │\n" "$TIME_CUDA"
printf "│ MPI:     %-10s ms                                    │\n" "$TIME_MPI"
echo "└─────────────────────────────────────────────────────────────┘"

echo ""
echo -e "${GREEN}✓ Benchmark concluído!${NC}"
echo -e "Resultados salvos em: ${BLUE}$RESULTS_DIR/${NC}"
echo ""
echo "Arquivos gerados:"
ls -la "$RESULTS_DIR"
