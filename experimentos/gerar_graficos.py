import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Dados dos experimentos
zero_shot = {"acc": 0.6467, "f1": 0.5066}
fl_50 = {
    "acc_melhor": 0.7333,
    "f1_melhor": 0.6386,
    "melhor_round": 5,
    "isolado": 0.5489,
    "ganho": 0.1844,
    "historico_acc": [0.6689, 0.6911, 0.7089, 0.7156, 0.7333],
    "historico_f1": [0.5428, 0.5715, 0.5999, 0.5641, 0.6386],
    "historico_loss": [1.4673, 0.2056, 0.1453, 0.1089, 0.0813]
}
fl_100 = {
    "acc_melhor": 0.8433,
    "f1_melhor": 0.8206,
    "melhor_round": 5,
    "isolado": 0.6333,
    "ganho": 0.2100,
    "historico_acc": [0.52, 0.7733, 0.6967, 0.71, 0.8433],
    "historico_f1": [0.3479, 0.7198, 0.6423, 0.6177, 0.8206],
    "historico_loss": [0.8792, 0.1477, 0.0875, 0.0646, 0.0547]
}

# Matrizes de confusao
cm_zero_shot = np.array([[299, 0, 0], [153, 0, 32], [27, 0, 89]])
cm_fl_50 = np.array([[225, 0, 0], [96, 16, 24], [0, 0, 89]])
cm_fl_100 = np.array([[138, 2, 0], [24, 53, 21], [0, 0, 62]])

labels = ["Leve", "Moderado", "Grave"]

# Criar pasta para graficos
os.makedirs("graficos_tcc", exist_ok=True)

# ============================================================
# GRAFICO 1: COMPARATIVO FINAL
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
exps = ["Zero-Shot", "FL 50", "FL 100"]
accs = [zero_shot["acc"], fl_50["acc_melhor"], fl_100["acc_melhor"]]
f1s = [zero_shot["f1"], fl_50["f1_melhor"], fl_100["f1_melhor"]]

x = np.arange(len(exps))
w = 0.35
bars1 = ax.bar(x - w/2, accs, w, label="Acurácia", color="#3498DB", edgecolor="#333")
bars2 = ax.bar(x + w/2, f1s, w, label="F1 Macro", color="#E74C3C", edgecolor="#333")

ax.set_xticks(x)
ax.set_xticklabels(exps, fontsize=12)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Valor", fontsize=12)
ax.set_title("Comparativo Final - Melhor Resultado por Experimento", fontweight="bold", fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis="y")

for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f"{h:.4f}", ha="center", fontsize=10, fontweight="bold")
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f"{h:.4f}", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("graficos_tcc/comparativo_final.png", dpi=200, bbox_inches="tight")
plt.close()
print("1. comparativo_final.png")

# ============================================================
# GRAFICO 2: LEARNING CURVE FL 50
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
rounds = list(range(1, 6))
ax.plot(rounds, fl_50["historico_acc"], "o-", color="#3498DB", lw=2, markersize=8, label="Acurácia FL 50")
ax.axhline(y=zero_shot["acc"], color="#95A5A6", ls="--", lw=2, label=f"Zero-Shot ({zero_shot['acc']:.4f})")
ax.axhline(y=fl_50["isolado"], color="#E74C3C", ls=":", lw=2, label=f"Isolado ({fl_50['isolado']:.4f})")

ax.set_xlabel("Rodada", fontsize=12)
ax.set_ylabel("Acurácia", fontsize=12)
ax.set_ylim(0.4, 0.85)
ax.set_title("Curva de Aprendizado - FL 50 CATs (5 Rounds)", fontweight="bold", fontsize=14)
ax.set_xticks(rounds)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

for i, v in enumerate(fl_50["historico_acc"]):
    ax.text(i + 1, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("graficos_tcc/learning_curve_fl_50.png", dpi=200, bbox_inches="tight")
plt.close()
print("2. learning_curve_fl_50.png")

# ============================================================
# GRAFICO 3: LEARNING CURVE FL 100
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(rounds, fl_100["historico_acc"], "o-", color="#2ECC71", lw=2, markersize=8, label="Acurácia FL 100")
ax.axhline(y=zero_shot["acc"], color="#95A5A6", ls="--", lw=2, label=f"Zero-Shot ({zero_shot['acc']:.4f})")
ax.axhline(y=fl_100["isolado"], color="#E74C3C", ls=":", lw=2, label=f"Isolado ({fl_100['isolado']:.4f})")

ax.set_xlabel("Rodada", fontsize=12)
ax.set_ylabel("Acurácia", fontsize=12)
ax.set_ylim(0.4, 0.95)
ax.set_title("Curva de Aprendizado - FL 100 CATs (5 Rounds)", fontweight="bold", fontsize=14)
ax.set_xticks(rounds)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

for i, v in enumerate(fl_100["historico_acc"]):
    ax.text(i + 1, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("graficos_tcc/learning_curve_fl_100.png", dpi=200, bbox_inches="tight")
plt.close()
print("3. learning_curve_fl_100.png")

# ============================================================
# GRAFICO 4: LOSS POR RODADA
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(rounds, fl_50["historico_loss"], "o-", color="#3498DB", lw=2, markersize=8, label="Loss FL 50")
ax.plot(rounds, fl_100["historico_loss"], "s-", color="#2ECC71", lw=2, markersize=8, label="Loss FL 100")

ax.set_xlabel("Rodada", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Evolução do Loss por Rodada", fontweight="bold", fontsize=14)
ax.set_xticks(rounds)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("graficos_tcc/loss_por_rodada.png", dpi=200, bbox_inches="tight")
plt.close()
print("4. loss_por_rodada.png")

# ============================================================
# GRAFICO 5: GANHO DO FL SOBRE ISOLADO
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
exps_ganho = ["FL 50", "FL 100"]
accs_fl = [fl_50["acc_melhor"], fl_100["acc_melhor"]]
accs_iso = [fl_50["isolado"], fl_100["isolado"]]
ganhos = [fl_50["ganho"], fl_100["ganho"]]

x = np.arange(len(exps_ganho))
w = 0.25
bars1 = ax.bar(x - w, accs_fl, w, label="FL (Melhor)", color="#3498DB", edgecolor="#333")
bars2 = ax.bar(x, accs_iso, w, label="Isolado", color="#95A5A6", edgecolor="#333")
bars3 = ax.bar(x + w, ganhos, w, label="Ganho", color="#2ECC71", edgecolor="#333")

ax.set_xticks(x)
ax.set_xticklabels(exps_ganho, fontsize=12)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Valor", fontsize=12)
ax.set_title("Ganho do FL sobre Treino Isolado", fontweight="bold", fontsize=14)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis="y")

for bar in bars3:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f"+{h:.1%}", ha="center", fontsize=11, fontweight="bold", color="#27AE60")

plt.tight_layout()
plt.savefig("graficos_tcc/ganho_fl_vs_isolado.png", dpi=200, bbox_inches="tight")
plt.close()
print("5. ganho_fl_vs_isolado.png")

# ============================================================
# GRAFICO 6: MATRIZES DE CONFUSAO
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

matrizes = [
    (cm_zero_shot, "Zero-Shot"),
    (cm_fl_50, "FL 50 CATs"),
    (cm_fl_100, "FL 100 CATs")
]

for ax, (cm, titulo) in zip(axes, matrizes):
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_title(titulo, fontweight="bold", fontsize=13)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predito", fontsize=11)
    ax.set_ylabel("Real", fontsize=11)
    
    thresh = cm.max() / 2.0
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")

plt.suptitle("Matrizes de Confusão - Comparativo", fontweight="bold", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("graficos_tcc/matrizes_confusao.png", dpi=200, bbox_inches="tight")
plt.close()
print("6. matrizes_confusao.png")

# ============================================================
# GRAFICO 7: TABELA RESUMO (IMAGEM)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")

col_labels = ["Experimento", "CATs", "Melhor Acc", "F1", "Round", "Isolado", "Ganho"]
table_data = [
    ["Zero-Shot", "0", f"{zero_shot['acc']:.4f}", f"{zero_shot['f1']:.4f}", "N/A", "N/A", "N/A"],
    ["FL 50", "50", f"{fl_50['acc_melhor']:.4f}", f"{fl_50['f1_melhor']:.4f}", str(fl_50['melhor_round']), f"{fl_50['isolado']:.4f}", f"+{fl_50['ganho']:.1%}"],
    ["FL 100", "100", f"{fl_100['acc_melhor']:.4f}", f"{fl_100['f1_melhor']:.4f}", str(fl_100['melhor_round']), f"{fl_100['isolado']:.4f}", f"+{fl_100['ganho']:.1%}"]
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

ax.set_title("Tabela de Resultados - TCC", fontweight="bold", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig("graficos_tcc/tabela_resultados.png", dpi=200, bbox_inches="tight")
plt.close()
print("7. tabela_resultados.png")

print("\n" + "="*60)
print("TODOS OS GRAFICOS GERADOS!")
print("="*60)
print("\nArquivos salvos em: graficos_tcc/")
print("1. comparativo_final.png")
print("2. learning_curve_fl_50.png")
print("3. learning_curve_fl_100.png")
print("4. loss_por_rodada.png")
print("5. ganho_fl_vs_isolado.png")
print("6. matrizes_confusao.png")
print("7. tabela_resultados.png")
