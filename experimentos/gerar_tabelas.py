import matplotlib.pyplot as plt
import numpy as np
import os

# Criar pasta para tabelas
os.makedirs("tabelas_tcc", exist_ok=True)

# ============================================================
# TABELA 1: ESTRUTURA DOS DADOS (SEM ID)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("off")

col_labels = ["Campo", "Tipo", "Exemplo"]
table_data = [
    ["Empresa", "Texto", "Empresa A"],
    ["Setor", "Texto", "Construção Civil"],
    ["Data do Acidente", "Data", "16/08/2024"],
    ["Tipo de Acidente", "Texto", "Entorse leve"],
    ["Parte do Corpo", "Texto", "Joelho"],
    ["Grau de Gravidade", "Texto", "Leve"],
    ["Dias de Afastamento", "Número", "3"],
    ["Descrição", "Texto", "Trabalhador escorregou..."]
]

table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Colorir cabecalho
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#2C3E50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Colorir linhas alternadas
for i in range(len(table_data)):
    color = "#EBF5FB" if i % 2 == 0 else "#FFFFFF"
    for j in range(len(col_labels)):
        table[i + 1, j].set_facecolor(color)

ax.set_title("Tabela 1 - Estrutura dos Dados das CATs Sintéticas", fontweight="bold", fontsize=13, pad=20)
plt.tight_layout()
plt.savefig("tabelas_tcc/tabela_estrutura_dados.png", dpi=200, bbox_inches="tight")
plt.close()
print("1. tabela_estrutura_dados.png")

# ============================================================
# TABELA 2: EXEMPLOS POR GRAVIDADE (COM ID)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis("off")

col_labels = ["ID", "Empresa", "Setor", "Data", "Tipo", "Parte", "Gravidade", "Dias", "Descrição"]
table_data = [
    ["emp_a_0001", "Empresa A", "Construção", "16/08/2024", "Entorse leve", "Joelho", "Leve", "3", "Escorregou no chão úmido..."],
    ["emp_a_0050", "Empresa A", "Construção", "22/03/2024", "Corte sup.", "Mão", "Leve", "1", "Corte na mão ao manusear furadeira..."],
    ["emp_b_0050", "Empresa B", "Metalurgia", "12/06/2024", "Queimadura 2°", "Braço", "Moderado", "20", "Queimadura por metal fundido..."],
    ["emp_c_0050", "Empresa C", "Transporte", "28/09/2024", "Entorse grave", "Coluna", "Moderado", "15", "Entorse na coluna em freada brusca..."],
    ["emp_a_0150", "Empresa A", "Construção", "03/04/2024", "Fratura exposta", "Perna", "Grave", "90", "Desabamento atingiu trabalhador..."],
    ["emp_b_0150", "Empresa B", "Metalurgia", "19/07/2024", "Amputação", "Braço", "Grave", "180", "Braço amputado em máquina sem proteção..."],
    ["emp_c_0150", "Empresa C", "Transporte", "08/12/2024", "Traumatismo", "Cabeça", "Grave", "120", "Colisão entre caminhões..."],
]

table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.6)

# Colorir cabecalho
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#2C3E50")
    table[0, j].set_text_props(color="white", fontweight="bold", fontsize=8)

# Colorir por gravidade
colors_map = {"Leve": "#E8F8F5", "Moderado": "#FEF9E7", "Grave": "#FDEDEC"}
for i in range(len(table_data)):
    gravidade = table_data[i][6]
    color = colors_map.get(gravidade, "#FFFFFF")
    for j in range(len(col_labels)):
        table[i + 1, j].set_facecolor(color)

ax.set_title("Tabela 2 - Exemplos de CATs por Nível de Gravidade", fontweight="bold", fontsize=13, pad=20)
plt.tight_layout()
plt.savefig("tabelas_tcc/tabela_exemplos_gravidade.png", dpi=200, bbox_inches="tight")
plt.close()
print("2. tabela_exemplos_gravidade.png")

# ============================================================
# TABELA 3: EXEMPLO FORMATADO (SEM ID)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("off")

col_labels = ["Campo", "Valor"]
table_data = [
    ["Empresa", "Empresa A"],
    ["Setor", "Construção Civil"],
    ["Data do Acidente", "16/08/2024"],
    ["Tipo de Acidente", "Entorse leve"],
    ["Parte do Corpo Atingida", "Joelho"],
    ["Grau de Gravidade", "Leve"],
    ["Dias de Afastamento", "3"],
    ["Descrição", "Trabalhador escorregou no chão úmido do canteiro e sofreu entorse leve no joelho."]
]

table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Colorir cabecalho
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#2C3E50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Colorir campo
for i in range(len(table_data)):
    table[i + 1, 0].set_facecolor("#3498DB")
    table[i + 1, 0].set_text_props(color="white", fontweight="bold")
    table[i + 1, 1].set_facecolor("#EBF5FB")

ax.set_title("Tabela 3 - Exemplo de CAT Formatada", fontweight="bold", fontsize=13, pad=20)
plt.tight_layout()
plt.savefig("tabelas_tcc/tabela_exemplo_formatado.png", dpi=200, bbox_inches="tight")
plt.close()
print("3. tabela_exemplo_formatado.png")

# ============================================================
# TABELA 4: DISTRIBUICAO DAS CATs
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.axis("off")

col_labels = ["Empresa", "Setor", "Leve", "Moderado", "Grave", "Total"]
table_data = [
    ["Empresa A", "Construção Civil", "100", "60", "40", "200"],
    ["Empresa B", "Metalurgia", "100", "60", "40", "200"],
    ["Empresa C", "Transporte", "100", "60", "40", "200"],
    ["Total", "", "300", "180", "120", "600"]
]

table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Colorir cabecalho
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#2C3E50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Colorir por tipo
colors = ["#3498DB", "#2ECC71", "#E74C3C"]
for i in range(3):
    table[i + 1, 0].set_facecolor(colors[i])
    table[i + 1, 0].set_text_props(color="white", fontweight="bold")
    table[i + 1, 1].set_facecolor("#ECF0F1")

# Colorir gravidade
for i in range(4):
    table[i + 1, 2].set_facecolor("#E8F8F5")  # Leve
    table[i + 1, 3].set_facecolor("#FEF9E7")  # Moderado
    table[i + 1, 4].set_facecolor("#FDEDEC")  # Grave

# Linha total
for j in range(len(col_labels)):
    table[4, j].set_facecolor("#2C3E50")
    table[4, j].set_text_props(color="white", fontweight="bold")

ax.set_title("Tabela 4 - Distribuição das CATs por Empresa e Gravidade", fontweight="bold", fontsize=13, pad=20)
plt.tight_layout()
plt.savefig("tabelas_tcc/tabela_distribuicao.png", dpi=200, bbox_inches="tight")
plt.close()
print("4. tabela_distribuicao.png")

# ============================================================
# TABELA 5: RESULTADOS DOS EXPERIMENTOS
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("off")

col_labels = ["Experimento", "CATs", "Melhor Acc", "F1", "Round", "Isolado", "Ganho"]
table_data = [
    ["Zero-Shot", "0", "0.6467", "0.5066", "N/A", "N/A", "N/A"],
    ["FL 50", "50", "0.7333", "0.6386", "5", "0.5489", "+18.44%"],
    ["FL 100", "100", "0.8433", "0.8206", "5", "0.6333", "+21.00%"]
]

table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Colorir cabecalho
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#2C3E50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Colorir linhas
colors = ["#EBF5FB", "#E8F8F5", "#FEF9E7"]
for i in range(3):
    for j in range(len(col_labels)):
        table[i + 1, j].set_facecolor(colors[i])

ax.set_title("Tabela 5 - Resultados dos Experimentos (5 Rounds)", fontweight="bold", fontsize=13, pad=20)
plt.tight_layout()
plt.savefig("tabelas_tcc/tabela_resultados.png", dpi=200, bbox_inches="tight")
plt.close()
print("5. tabela_resultados.png")

# ============================================================
# TABELA 6: VIABILIDADE
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("off")

col_labels = ["Configuração", "Tempo/CAT", "50 CATs", "100 CATs", "Custo"]
table_data = [
    ["PC básico (HDD)", "13.3s", "11.1 min", "22.2 min", "R$ 0"],
    ["PC básico (SSD)", "8.8s", "7.3 min", "14.6 min", "R$ 150-200"],
    ["PC médio (i7)", "4.5s", "3.8 min", "7.5 min", "R$ 350-500"],
    ["PC com GPU", "1.0s", "0.8 min", "1.7 min", "R$ 1.200-1.700"]
]

table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Colorir cabecalho
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#2C3E50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Colorir linhas
colors = ["#FDEDEC", "#FEF9E7", "#E8F8F5", "#EBF5FB"]
for i in range(4):
    for j in range(len(col_labels)):
        table[i + 1, j].set_facecolor(colors[i])

ax.set_title("Tabela 6 - Viabilidade por Configuração de Hardware", fontweight="bold", fontsize=13, pad=20)
plt.tight_layout()
plt.savefig("tabelas_tcc/tabela_viabilidade.png", dpi=200, bbox_inches="tight")
plt.close()
print("6. tabela_viabilidade.png")

print("\n" + "="*60)
print("TODAS AS TABELAS GERADAS!")
print("="*60)
print("\nArquivos salvos em: tabelas_tcc/")
print("1. tabela_estrutura_dados.png")
print("2. tabela_exemplos_gravidade.png")
print("3. tabela_exemplo_formatado.png")
print("4. tabela_distribuicao.png")
print("5. tabela_resultados.png")
print("6. tabela_viabilidade.png")
