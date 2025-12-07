# ============================================
# Makefile - Projeto K-Means PCD
# Programação Concorrente e Distribuída
# ============================================

# Compiladores
CC = gcc
NVCC = nvcc
MPICC = mpicc

# Flags de compilação
CFLAGS = -O3 -Wall -lm
OPENMP_FLAGS = -fopenmp
CUDA_FLAGS = -O3 -arch=sm_50
MPI_FLAGS = -O3 -Wall

# Diretórios
SRC_DIR = src
BIN_DIR = bin
DATA_DIR = data

# Arquivos fonte
NAIVE_SRC = $(SRC_DIR)/kmeans_naive.c
OPENMP_SRC = $(SRC_DIR)/kmeans_openmp.c
CUDA_SRC = $(SRC_DIR)/kmeans_cuda.cu
MPI_SRC = $(SRC_DIR)/kmeans_mpi.c
GENERATOR_SRC = $(SRC_DIR)/generate_data.c

# Executáveis
NAIVE_BIN = $(BIN_DIR)/kmeans_naive
OPENMP_BIN = $(BIN_DIR)/kmeans_openmp
CUDA_BIN = $(BIN_DIR)/kmeans_cuda
MPI_BIN = $(BIN_DIR)/kmeans_mpi
GENERATOR_BIN = $(BIN_DIR)/generate_data

# ============================================
# Regras principais
# ============================================

.PHONY: all clean naive openmp cuda mpi generator run-all help

all: naive openmp mpi generator
	@echo ""
	@echo "✓ Compilação concluída!"
	@echo "  Executáveis em: $(BIN_DIR)/"

# Criar diretório bin se não existir
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# ============================================
# Compilação individual
# ============================================

naive: $(BIN_DIR) $(NAIVE_SRC)
	$(CC) $(CFLAGS) -o $(NAIVE_BIN) $(NAIVE_SRC)
	@echo "✓ kmeans_naive compilado"

openmp: $(BIN_DIR) $(OPENMP_SRC)
	$(CC) $(CFLAGS) $(OPENMP_FLAGS) -o $(OPENMP_BIN) $(OPENMP_SRC)
	@echo "✓ kmeans_openmp compilado"

cuda: $(BIN_DIR) $(CUDA_SRC)
	$(NVCC) $(CUDA_FLAGS) -o $(CUDA_BIN) $(CUDA_SRC)
	@echo "✓ kmeans_cuda compilado"

mpi: $(BIN_DIR) $(MPI_SRC)
	$(MPICC) $(MPI_FLAGS) -o $(MPI_BIN) $(MPI_SRC) -lm
	@echo "✓ kmeans_mpi compilado"

generator: $(BIN_DIR) $(GENERATOR_SRC)
	$(CC) $(CFLAGS) -o $(GENERATOR_BIN) $(GENERATOR_SRC)
	@echo "✓ generate_data compilado"

# ============================================
# Execução
# ============================================

# Gerar dados de teste
generate: generator
	cd $(DATA_DIR) && ../$(GENERATOR_BIN) 100000 5 42

# Executar versão naive
run-naive: naive
	cd $(DATA_DIR) && ../$(NAIVE_BIN)

# Executar versão OpenMP (4 threads)
run-openmp: openmp
	cd $(DATA_DIR) && ../$(OPENMP_BIN) dados.csv centroides_iniciais.csv 4

# Executar versão CUDA
run-cuda: cuda
	cd $(DATA_DIR) && ../$(CUDA_BIN)

# Executar versão MPI (4 processos)
run-mpi: mpi
	cd $(DATA_DIR) && mpirun -np 4 ../$(MPI_BIN)

# Executar todas as versões para comparação
run-all: naive openmp mpi
	@echo ""
	@echo "=========================================="
	@echo "  Executando comparação de desempenho"
	@echo "=========================================="
	@echo ""
	@echo "--- Versão NAIVE (Sequencial) ---"
	cd $(DATA_DIR) && ../$(NAIVE_BIN)
	@echo ""
	@echo "--- Versão OpenMP (4 threads) ---"
	cd $(DATA_DIR) && ../$(OPENMP_BIN) dados.csv centroides_iniciais.csv 4
	@echo ""
	@echo "--- Versão MPI (4 processos) ---"
	cd $(DATA_DIR) && mpirun -np 4 ../$(MPI_BIN)

# ============================================
# Limpeza
# ============================================

clean:
	rm -rf $(BIN_DIR)
	rm -f $(DATA_DIR)/assign*.csv $(DATA_DIR)/centroids*.csv
	@echo "✓ Arquivos limpos"

# ============================================
# Ajuda
# ============================================

help:
	@echo ""
	@echo "Projeto K-Means PCD - Makefile"
	@echo "=============================="
	@echo ""
	@echo "Compilação:"
	@echo "  make all        - Compila todas as versões (exceto CUDA)"
	@echo "  make naive      - Compila versão sequencial"
	@echo "  make openmp     - Compila versão OpenMP"
	@echo "  make cuda       - Compila versão CUDA (requer NVCC)"
	@echo "  make mpi        - Compila versão MPI"
	@echo "  make generator  - Compila gerador de dados"
	@echo ""
	@echo "Execução:"
	@echo "  make generate   - Gera dados de teste (100k pontos, 5 clusters)"
	@echo "  make run-naive  - Executa versão sequencial"
	@echo "  make run-openmp - Executa versão OpenMP (4 threads)"
	@echo "  make run-cuda   - Executa versão CUDA"
	@echo "  make run-mpi    - Executa versão MPI (4 processos)"
	@echo "  make run-all    - Executa todas e compara desempenho"
	@echo ""
	@echo "Outros:"
	@echo "  make clean      - Remove executáveis e arquivos gerados"
	@echo "  make help       - Mostra esta ajuda"
	@echo ""
