# TCC - Aprendizado Federado com LLM para Classificação de CATs

Trabalho de Conclusão de Curso (TCC) do Curso de Engenharia de Segurança do Trabalho.

## Resumo

Este trabalho propõe uma arquitetura de Aprendizado Federado combinada com Modelos de Linguagem (LLM) e armazenamento criptografado descentralizado (IPFS + AES-256) para classificar a gravidade de acidentes de trabalho (CATs) sem compartilhar dados pessoais.

## Resultados dos Experimentos

### Configuração

| Parâmetro | Valor |
|-----------|-------|
| Modelo | Qwen/Qwen2.5-1.5B-Instruct |
| Quantização | 4-bit NF4 |
| LoRA | r=8, alpha=16, dropout=0.1 |
| Rounds | 5 |
| Epochs | 1 |
| Learning Rate | 1e-4 |
| Batch Size | 4 |

### Tabela de Resultados

| Experimento | CATs | Melhor Acc | F1 | Round | Isolado | Ganho |
|-------------|------|------------|-----|-------|---------|-------|
| Zero-Shot | 0 | 0.6467 | 0.5066 | N/A | N/A | N/A |
| FL 50 | 50 | **0.7333** | **0.6386** | 5 | 0.5489 | +18.44% |
| FL 100 | 100 | **0.8433** | **0.8206** | 5 | 0.6333 | +21.00% |

### Histórico por Rodada

| Round | FL 50 Acc | FL 50 Loss | FL 100 Acc | FL 100 Loss |
|-------|-----------|------------|------------|-------------|
| 1 | 0.6689 | 1.4673 | 0.5200 | 0.8792 |
| 2 | 0.6911 | 0.2056 | 0.7733 | 0.1477 |
| 3 | 0.7089 | 0.1453 | 0.6967 | 0.0875 |
| 4 | 0.7156 | 0.1089 | 0.7100 | 0.0646 |
| **5** | **0.7333** | 0.0813 | **0.8433** | 0.0547 |

### Hipótese Confirmada

**Mais CATs → Melhor Acurácia**

- Zero-Shot: 64.67%
- FL 50: 73.33% (+8.66% sobre Zero-Shot)
- FL 100: 84.33% (+19.66% sobre Zero-Shot)

### Viabilidade para Empresa de Médio Porte

#### Simulação em PC Básico Corporativo

| Componente | Configuração |
|------------|--------------|
| Processador | Intel Core i5-4590 (4 cores, 3.3 GHz) |
| RAM | 8 GB |
| GPU | Nenhuma (apenas integrada) |
| Disco | HDD 500 GB |

#### Performance por Volume Diário

| Volume | Tempo (HDD) | Tempo (SSD) | Viável? |
|--------|-------------|-------------|---------|
| 10 CATs/dia | 2.2 minutos | 1.5 minutos | **Sim** |
| 20 CATs/dia | 4.5 minutos | 2.9 minutos | **Sim** |
| 50 CATs/dia | 11.1 minutos | 7.3 minutos | **Sim** |
| 100 CATs/dia | 22.2 minutos | 14.6 minutos | **Sim** |

#### Custo de Implementação

| Opção | Custo | Performance | Limitação |
|-------|-------|-------------|-----------|
| Sem upgrade | R$ 0 | 13.3s/CAT | Até 20 CATs/dia |
| Upgrade SSD | R$ 150-200 | 8.8s/CAT | Até 50 CATs/dia |
| Upgrade completo | R$ 1.200-1.700 | 1s/CAT | Sem limitação |

## Gráficos

Os gráficos para o TCC estão em `experimentos/graficos_tcc/`:

1. `comparativo_final.png` - Comparativo de todos os experimentos
2. `learning_curve_fl_50.png` - Curva de aprendizado FL 50
3. `learning_curve_fl_100.png` - Curva de aprendizado FL 100
4. `loss_por_rodada.png` - Evolução do loss
5. `ganho_fl_vs_isolado.png` - Ganho do FL sobre isolado
6. `matrizes_confusao.png` - Matrizes de confusão comparativas
7. `tabela_resultados.png` - Tabela resumo

## Estrutura do Projeto

```
experimentos/
├── gerar_cats_sinteticas.py    # Geração de dados sintéticos (600 CATs)
├── crypto_utils.py             # Criptografia AES-256 e IPFS
├── zero_shot.py                # Experimento baseline (sem treino)
├── fl_50_cats.py               # FL com 50 CATs por empresa
├── fl_encrypted.py             # FL com criptografia AES-256 e IPFS
├── viabilidade.py              # Análise de viabilidade para empresa
├── teste_cpu_only.py           # Teste de performance sem GPU
├── simulacao_pc_basico.py      # Simulação em PC básico corporativo
├── gerar_graficos.py           # Gerar gráficos para o TCC
├── analisar_cats.py            # Software para analisar CATs
├── tcc_final_ate_fl100.ipynb   # Notebook FL (10 rounds)
├── tcc_crypto_5rounds.ipynb    # Notebook FL + Criptografia (5 rounds)
├── tcc_5rounds.ipynb           # Notebook FL (5 rounds, sem criptografia)
└── graficos_tcc/               # Gráficos gerados para o TCC
    ├── comparativo_final.png
    ├── learning_curve_fl_50.png
    ├── learning_curve_fl_100.png
    ├── loss_por_rodada.png
    ├── ganho_fl_vs_isolado.png
    ├── matrizes_confusao.png
    └── tabela_resultados.png
```

## Conclusão

1. **Hipótese confirmada**: Mais CATs disponíveis para treino levam a melhor acurácia
2. **FL funciona**: Ganho de 18-21% sobre treino isolado
3. **Viável para empresas**: PC básico com 8 GB RAM suporta até 50 CATs/dia
4. **Acessível**: Upgrade de SSD (R$ 150-200) melhora performance em 34%
5. **Privacidade garantida**: Criptografia AES-256 + IPFS

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
