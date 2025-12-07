<# 
.SYNOPSIS
    Script de Benchmark para K-Means PCD (Windows PowerShell)

.DESCRIPTION
    Executa todas as versões do K-Means e gera relatório comparativo

.PARAMETER N
    Número de pontos (padrão: 100000)

.PARAMETER K
    Número de clusters (padrão: 5)

.EXAMPLE
    .\run_benchmark.ps1
    .\run_benchmark.ps1 -N 500000 -K 10
#>

param(
    [int]$N = 100000,
    [int]$K = 5,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "              BENCHMARK K-MEANS PCD                            " -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuracao: N=$N, K=$K, SEED=$Seed" -ForegroundColor Blue
Write-Host ""

# Diretórios
$RootDir = $PSScriptRoot
$DataDir = Join-Path $RootDir "data"
$SerialDir = Join-Path $RootDir "serial"
$OpenMPDir = Join-Path $RootDir "openmp"
$CudaDir = Join-Path $RootDir "cuda"
$MpiDir = Join-Path $RootDir "mpi"
$ResultsDir = Join-Path $RootDir "results"

# Criar diretório de resultados
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

# ============================================================================
# PASSO 1: GERAR DADOS
# ============================================================================
Write-Host "[1/5] Gerando dados de teste..." -ForegroundColor Yellow
Set-Location $DataDir

if (-not (Test-Path "generate_data.exe")) {
    Write-Host "Compilando gerador de dados..."
    gcc -O3 -o generate_data.exe generate_data.c -lm
}

& .\generate_data.exe -n $N -k $K -s $Seed
Write-Host "[OK] Dados gerados" -ForegroundColor Green
Write-Host ""

# ============================================================================
# PASSO 2: VERSÃO SERIAL
# ============================================================================
Write-Host "[2/5] Executando versão SERIAL..." -ForegroundColor Yellow
Set-Location $SerialDir

if (-not (Test-Path "kmeans_serial.exe")) {
    Write-Host "Compilando versão serial..."
    gcc -O3 -o kmeans_serial.exe kmeans_serial.c -lm
}

$serialOutput = & .\kmeans_serial.exe -d "$DataDir\dados.csv" -c "$DataDir\centroides_iniciais.csv"
$serialOutput | Tee-Object -FilePath "$ResultsDir\output_serial.txt"

# Extrair tempo serial
$serialTime = ($serialOutput | Select-String "Tempo Total:" | ForEach-Object { $_ -replace '.*:\s*(\d+\.?\d*)\s*ms.*', '$1' })
Write-Host "[OK] Serial concluido - Tempo: ${serialTime}ms" -ForegroundColor Green
Write-Host ""

# Copiar resultados
Copy-Item "assign_serial.csv" $ResultsDir -Force -ErrorAction SilentlyContinue
Copy-Item "centroids_serial.csv" $ResultsDir -Force -ErrorAction SilentlyContinue
Copy-Item "sse_history_serial.csv" $ResultsDir -Force -ErrorAction SilentlyContinue

# ============================================================================
# PASSO 3: VERSÃO OPENMP
# ============================================================================
Write-Host "[3/5] Executando versão OPENMP..." -ForegroundColor Yellow
Set-Location $OpenMPDir

if (-not (Test-Path "kmeans_openmp.exe")) {
    Write-Host "Compilando versão OpenMP..."
    gcc -O3 -fopenmp -o kmeans_openmp.exe kmeans_openmp.c -lm
}

$openmpOutput = & .\kmeans_openmp.exe -d "$DataDir\dados.csv" -c "$DataDir\centroides_iniciais.csv" -a --serial-time $serialTime
$openmpOutput | Tee-Object -FilePath "$ResultsDir\output_openmp.txt"

Write-Host "[OK] OpenMP concluido" -ForegroundColor Green
Write-Host ""

# Copiar resultados
Copy-Item "assign_openmp.csv" $ResultsDir -Force -ErrorAction SilentlyContinue
Copy-Item "centroids_openmp.csv" $ResultsDir -Force -ErrorAction SilentlyContinue
Copy-Item "sse_history_openmp.csv" $ResultsDir -Force -ErrorAction SilentlyContinue
Copy-Item "scalability_openmp.csv" $ResultsDir -Force -ErrorAction SilentlyContinue

# ============================================================================
# PASSO 4: VERSÃO CUDA
# ============================================================================
Write-Host "[4/5] Verificando CUDA..." -ForegroundColor Yellow
Set-Location $CudaDir

$nvccExists = Get-Command nvcc -ErrorAction SilentlyContinue
if ($nvccExists) {
    Write-Host "CUDA encontrado, compilando..."
    
    if (-not (Test-Path "kmeans_cuda.exe")) {
        try {
            nvcc -O3 -o kmeans_cuda.exe kmeans_cuda.cu
        } catch {
            Write-Host "[WARN] Erro na compilacao CUDA" -ForegroundColor Red
        }
    }
    
    if (Test-Path "kmeans_cuda.exe") {
        $cudaOutput = & .\kmeans_cuda.exe -d "$DataDir\dados.csv" -c "$DataDir\centroides_iniciais.csv" -a --serial-time $serialTime
        $cudaOutput | Tee-Object -FilePath "$ResultsDir\output_cuda.txt"
        
        Copy-Item "assign_cuda.csv" $ResultsDir -Force -ErrorAction SilentlyContinue
        Copy-Item "centroids_cuda.csv" $ResultsDir -Force -ErrorAction SilentlyContinue
        Copy-Item "sse_history_cuda.csv" $ResultsDir -Force -ErrorAction SilentlyContinue
        Copy-Item "blocksize_cuda.csv" $ResultsDir -Force -ErrorAction SilentlyContinue
        
        Write-Host "[OK] CUDA concluido" -ForegroundColor Green
    }
} else {
    Write-Host "[WARN] CUDA nao disponivel - pulando" -ForegroundColor Yellow
}
Write-Host ""

# ============================================================================
# PASSO 5: VERSÃO MPI
# ============================================================================
Write-Host "[5/5] Verificando MPI..." -ForegroundColor Yellow
Set-Location $MpiDir

$mpiccExists = Get-Command mpicc -ErrorAction SilentlyContinue
$mpiexecExists = Get-Command mpiexec -ErrorAction SilentlyContinue

if ($mpiccExists -or $mpiexecExists) {
    if (-not (Test-Path "kmeans_mpi.exe")) {
        Write-Host "Compilando versão MPI..."
        try {
            mpicc -O3 -o kmeans_mpi.exe kmeans_mpi.c -lm
        } catch {
            # Tentar com cl para MS-MPI
            Write-Host "Tentando compilação com MS-MPI..."
        }
    }
    
    if (Test-Path "kmeans_mpi.exe") {
        # Limpar arquivo anterior
        Remove-Item "scalability_mpi.csv" -Force -ErrorAction SilentlyContinue
        
        # Testar com diferentes números de processos
        foreach ($np in @(1, 2, 4)) {
            Write-Host "Executando com $np processos..."
            try {
                mpiexec -n $np .\kmeans_mpi.exe -d "$DataDir\dados.csv" -c "$DataDir\centroides_iniciais.csv" -a --serial-time $serialTime 2>$null
            } catch {}
        }
        
        $mpiOutput = mpiexec -n 4 .\kmeans_mpi.exe -d "$DataDir\dados.csv" -c "$DataDir\centroides_iniciais.csv" --serial-time $serialTime
        $mpiOutput | Tee-Object -FilePath "$ResultsDir\output_mpi.txt"
        
        Copy-Item "assign_mpi.csv" $ResultsDir -Force -ErrorAction SilentlyContinue
        Copy-Item "centroids_mpi.csv" $ResultsDir -Force -ErrorAction SilentlyContinue
        Copy-Item "sse_history_mpi.csv" $ResultsDir -Force -ErrorAction SilentlyContinue
        Copy-Item "scalability_mpi.csv" $ResultsDir -Force -ErrorAction SilentlyContinue
        
        Write-Host "[OK] MPI concluido" -ForegroundColor Green
    }
} else {
    Write-Host "[WARN] MPI nao disponivel - pulando" -ForegroundColor Yellow
}
Write-Host ""

# ============================================================================
# RESUMO
# ============================================================================
Set-Location $ResultsDir

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  RESUMO DE DESEMPENHO                                        " -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

$files = @("output_serial.txt", "output_openmp.txt", "output_cuda.txt", "output_mpi.txt")
$names = @("Serial", "OpenMP", "CUDA  ", "MPI   ")

for ($i = 0; $i -lt $files.Length; $i++) {
    $file = $files[$i]
    $name = $names[$i]
    
    if (Test-Path $file) {
        $content = Get-Content $file -Raw
        $time = if ($content -match "Tempo Total:\s*([\d.]+)") { $matches[1] } else { "N/A" }
        $sse = if ($content -match "SSE Final:\s*([\d.]+)") { $matches[1] } else { "N/A" }
        Write-Host "  ${name}: Tempo = ${time}ms, SSE = $sse"
    }
}

Write-Host ""
Write-Host "[OK] Benchmark concluido!" -ForegroundColor Green
Write-Host "Resultados salvos em: $ResultsDir" -ForegroundColor Blue
Write-Host ""

# Voltar ao diretório raiz
Set-Location $RootDir
