# TCC - Aprendizado Federado com LLM para Classificação de CATs

Trabalho de Conclusão de Curso (TCC) do Curso de Engenharia de Segurança do Trabalho.

## Resumo

Este trabalho propõe uma arquitetura de Aprendizado Federado combinada com Modelos de Linguagem (LLM) e armazenamento criptografado descentralizado (IPFS + AES-256) para classificar a gravidade de acidentes de trabalho (CATs) sem compartilhar dados pessoais.

## Estrutura do Projeto

```
experimentos/
├── gerar_cats_sinteticas.py    # Geração de dados sintéticos (600 CATs)
├── zero_shot.py                # Experimento baseline (sem treino)
├── fl_50_cats.py               # FL com 50 CATs por empresa
└── tcc_final_ate_fl100.ipynb   # Notebook completo para Google Colab
```

## Como Executar

### Opção 1: Google Colab (Recomendado)

1. Abra o notebook `tcc_final_ate_fl100.ipynb` no Google Colab
2. Configure GPU: Runtime > Change runtime type > GPU: T4
3. Execute todas as células

### Opção 2: Localmente

```bash
# Instalar dependências
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets bitsandbytes peft accelerate scikit-learn matplotlib

# Gerar dados
python experimentos/gerar_cats_sinteticas.py

# Executar Zero-Shot
python experimentos/zero_shot.py

# Executar FL 50
python experimentos/fl_50_cats.py
```

## Parâmetros dos Experimentos

| Parâmetro | Valor |
|-----------|-------|
| Modelo | Qwen/Qwen2.5-1.5B-Instruct |
| Quantização | 4-bit NF4 |
| LoRA | r=8, alpha=16, dropout=0.1 |
| Rounds | 10 |
| Epochs | 1 |
| Learning Rate | 1e-4 |
| Batch Size | 4 |

## Resultados

| Experimento | CATs | Melhor Acc | F1 | Round |
|-------------|------|------------|-----|-------|
| Zero-Shot | 0 | 0.6467 | 0.5066 | N/A |
| FL 50 | 50 | 0.7622 | 0.6676 | 7 |
| FL 100 | 100 | 0.8733 | 0.8336 | 8 |

## Hipótese Confirmada

Mais CATs disponíveis para treino levam a melhor acurácia, mesmo sem compartilhar os dados diretamente.

## Tecnologias Utilizadas

- **Modelo**: Qwen/Qwen2.5-1.5B-Instruct (1.5B params)
- **Framework FL**: Flower (FedAvg)
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Quantização**: BitsAndBytes (4-bit NF4)
- **GPU**: NVIDIA T4 (16 GB VRAM)

## Autor

**maurilobraz** - Engenharia de Segurança do Trabalho

## Data

Julho 2026
