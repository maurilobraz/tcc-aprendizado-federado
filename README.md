# TCC - Aprendizado Federado com LLM para Classificação de CATs

Trabalho de Conclusão de Curso (TCC) do Curso de Engenharia de Segurança do Trabalho.

## Resumo

Este trabalho propõe uma arquitetura de Aprendizado Federado combinada com Modelos de Linguagem (LLM) e armazenamento criptografado descentralizado (IPFS + AES-256) para classificar a gravidade de acidentes de trabalho (CATs) sem compartilhar dados pessoais.

## Estrutura do Projeto

```
experimentos/
├── gerar_cats_sinteticas.py    # Geração de dados sintéticos (600 CATs)
├── crypto_utils.py             # Criptografia AES-256 e IPFS
├── zero_shot.py                # Experimento baseline (sem treino)
├── fl_50_cats.py               # FL com 50 CATs por empresa
├── fl_encrypted.py             # FL com criptografia AES-256 e IPFS
├── viabilidade.py              # Análise de viabilidade para empresa
├── tcc_final_ate_fl100.ipynb   # Notebook FL (10 rounds)
├── tcc_crypto_5rounds.ipynb    # Notebook FL + Criptografia (5 rounds)
└── tcc_5rounds.ipynb           # Notebook FL (5 rounds, sem criptografia)
```

## Como Executar

### Opção 1: Google Colab (Recomendado para testes)

#### Notebook FL (5 rounds)
1. Abra `tcc_5rounds.ipynb` no Google Colab
2. Configure GPU: Runtime > Change runtime type > GPU: T4
3. Execute todas as células

#### Notebook FL + Criptografia (5 rounds)
1. Abra `tcc_crypto_5rounds.ipynb` no Google Colab
2. Configure GPU: Runtime > Change runtime type > GPU: T4
3. Execute todas as células

### Opção 2: Localmente

```bash
# Instalar dependências
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets bitsandbytes peft accelerate scikit-learn matplotlib cryptography psutil

# Gerar dados
python experimentos/gerar_cats_sinteticas.py

# Executar Zero-Shot
python experimentos/zero_shot.py

# Executar FL 50
python experimentos/fl_50_cats.py

# Executar FL com Criptografia
python experimentos/fl_encrypted.py

# Analisar viabilidade
python experimentos/viabilidade.py
```

### Opção 3: Software para Empresa

```bash
# Executar análise de viabilidade
python experimentos/viabilidade.py

# Executar software de análise de CATs
python experimentos/analisar_cats.py [pasta_com_cats]
```

## Parâmetros dos Experimentos

| Parâmetro | Valor |
|-----------|-------|
| Modelo | Qwen/Qwen2.5-1.5B-Instruct |
| Quantização | 4-bit NF4 |
| LoRA | r=8, alpha=16, dropout=0.1 |
| Rounds | 5 ou 10 |
| Epochs | 1 |
| Learning Rate | 1e-4 |
| Batch Size | 4 |
| Criptografia | AES-256-CBC |
| Armazenamento | IPFS (simulado) |

## Resultados

### FL (10 rounds)

| Experimento | CATs | Melhor Acc | F1 | Round |
|-------------|------|------------|-----|-------|
| Zero-Shot | 0 | 0.6467 | 0.5066 | N/A |
| FL 50 | 50 | 0.7622 | 0.6676 | 7 |
| FL 100 | 100 | 0.8733 | 0.8336 | 8 |

### FL + Criptografia (5 rounds)

| Experimento | CATs | Melhor Acc | F1 | Round | IPFS CIDs |
|-------------|------|------------|-----|-------|-----------|
| Zero-Shot | 0 | - | - | N/A | N/A |
| FL 50 | 50 | - | - | - | 15 |
| FL 100 | 100 | - | - | - | 15 |

## Hipótese Confirmada

Mais CATs disponíveis para treino levam a melhor acurácia, mesmo sem compartilhar os dados diretamente.

## Viabilidade para Empresa de Médio Porte

### Configuração Mínima

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| RAM | 8 GB | 16 GB |
| GPU | Sem GPU (CPU only) | NVIDIA GTX 1650 (4GB) |
| Armazenamento | 50 GB SSD | 100 GB SSD |
| Internet | 10 Mbps | 50 Mbps |

### Custo Estimado

| Configuração | Custo (R$) | Observação |
|--------------|------------|------------|
| Básica (CPU only) | R$ 3.000-4.000 | Funciona, mas lento |
| Intermediária (GPU) | R$ 5.000-7.000 | Melhor custo-benefício |
| Avançada (GPU potente) | R$ 8.000-12.000 | Para alto volume |

### Overhead de Criptografia

| Componente | Overhead |
|------------|----------|
| AES-256 | < 1% |
| IPFS | Dependente da rede |
| Total | Viável para internet 10+ Mbps |

## Arquitetura de Segurança

### Criptografia AES-256

Os parâmetros LoRA são criptografados com AES-256-CBC antes de serem enviados ao servidor:

1. **Empresa treina** → Atualiza parâmetros LoRA
2. **Criptografa** → AES-256-CBC com chave de 256 bits
3. **Envia** → Parâmetros criptografados
4. **Servidor descriptografa** → Usa chave para descriptografar
5. **Agrega** → Calcula média (FedAvg)

### Armazenamento IPFS

Os parâmetros criptografados são armazenados no IPFS:

- **Content Identifier (CID)** → Hash único do conteúdo
- **Descentralizado** → Sem ponto único de falha
- **Imutável** → Conteúdo não pode ser alterado

### Fluxo de Dados

```
Empresa A → [LoRA Params] → [AES-256 Encrypt] → [IPFS Store] → CID_A
Empresa B → [LoRA Params] → [AES-256 Encrypt] → [IPFS Store] → CID_B
Empresa C → [LoRA Params] → [AES-256 Encrypt] → [IPFS Store] → CID_C

Servidor → [IPFS Get CID_A,B,C] → [AES-256 Decrypt] → [FedAvg] → Modelo Global
```

## Tecnologias Utilizadas

- **Modelo**: Qwen/Qwen2.5-1.5B-Instruct (1.5B params)
- **Framework FL**: Flower (FedAvg)
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Quantização**: BitsAndBytes (4-bit NF4)
- **Criptografia**: AES-256-CBC
- **Armazenamento**: IPFS (simulado)
- **GPU**: NVIDIA T4 (16 GB VRAM)

## Autor

**maurilobraz** - Engenharia de Segurança do Trabalho

## Data

Julho 2026
