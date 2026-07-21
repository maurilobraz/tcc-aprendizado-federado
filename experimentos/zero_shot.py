import csv
import json
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), "01_geracao_dados")
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42
MAX_NEW_TOKENS = 10

LABELS = ["leve", "moderado", "grave"]

SYSTEM_PROMPT = (
    "Voce e um classificador de gravidade de acidentes de trabalho. "
    "Responda APENAS com uma das opcoes: leve, moderado, grave. "
    "Nao inclua nenhuma outra palavra."
)


def load_dataset():
    path = os.path.join(DATA_DIR, "dataset_todos.csv")
    registros = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            registros.append(row)
    return registros


def load_model():
    print("Carregando Phi-4-mini (4-bit quantizado)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    model.eval()
    print(f"Modelo carregado em: {model.device}")
    return model, tokenizer


def classificar(model, tokenizer, descricao):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Classifique a gravidade deste acidente de trabalho:\n\n{descricao}"}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            top_p=1.0
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    resposta = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()

    for label in LABELS:
        if label in resposta:
            return label
    return resposta


def plot_confusion_matrix(cm, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Matriz de Confusao - Zero-Shot", fontweight="bold", fontsize=13)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(LABELS))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(LABELS)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predito", fontsize=11)
    ax.set_ylabel("Real", fontsize=11)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_metricas_por_classe(report_dict, output_path):
    classes = ["leve", "moderado", "grave"]
    prec = [report_dict[c]["precision"] for c in classes]
    rec = [report_dict[c]["recall"] for c in classes]
    f1 = [report_dict[c]["f1-score"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, prec, width, label="Precisao", color="#3498DB", edgecolor="#333")
    ax.bar(x, rec, width, label="Recall", color="#2ECC71", edgecolor="#333")
    ax.bar(x + width, f1, width, label="F1-Score", color="#E74C3C", edgecolor="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Valor")
    ax.set_title("Metricas por Classe - Zero-Shot", fontweight="bold", fontsize=13)
    ax.legend()
    for i in range(len(classes)):
        ax.text(x[i] - width, prec[i] + 0.02, f"{prec[i]:.2f}", ha="center", fontsize=8)
        ax.text(x[i], rec[i] + 0.02, f"{rec[i]:.2f}", ha="center", fontsize=8)
        ax.text(x[i] + width, f1[i] + 0.02, f"{f1[i]:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    registros = load_dataset()
    y_true = [r["grau_gravidade"] for r in registros]
    print(f"Dataset carregado: {len(registros)} registros")

    model, tokenizer = load_model()

    print(f"\nIniciando classificacao zero-shot de {len(registros)} CATs...")
    t_start = time.time()

    y_pred = []
    acertos = 0
    for i, r in enumerate(registros):
        pred = classificar(model, tokenizer, r["descricao"])
        y_pred.append(pred)
        if pred == r["grau_gravidade"]:
            acertos += 1
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            acc_parcial = acertos / (i + 1)
            print(f"  [{i+1}/{len(registros)}] Acc parcial: {acc_parcial:.1%} | Tempo: {elapsed:.0f}s")

    t_total = time.time() - t_start
    tempo_medio = t_total / len(registros)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_por_classe = f1_score(y_true, y_pred, average=None, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)

    print(f"\n{'='*50}")
    print(f"RESULTADOS ZERO-SHOT")
    print(f"{'='*50}")
    print(f"Acuracia: {acc:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")
    print(f"Tempo total: {t_total:.1f}s | Tempo medio: {tempo_medio:.2f}s/CAT")
    print(f"\nMatriz de Confusao:")
    print(cm)
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

    plot_confusion_matrix(cm, os.path.join(OUTPUT_DIR, "matriz_confusao_zero_shot.png"))
    plot_metricas_por_classe(report, os.path.join(OUTPUT_DIR, "metricas_por_classe_zero_shot.png"))

    resultados = {
        "experimento": "zero_shot",
        "modelo": MODEL_ID,
        "quantizacao": "4-bit NF4",
        "total_amostras": len(registros),
        "acuracia": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "f1_leve": round(f1_por_classe[0], 4),
        "f1_moderado": round(f1_por_classe[1], 4),
        "f1_grave": round(f1_por_classe[2], 4),
        "tempo_total_seg": round(t_total, 1),
        "tempo_medio_seg": round(tempo_medio, 2),
        "matriz_confusao": cm.tolist(),
        "seed": SEED,
        "gpu": "NVIDIA GeForce GTX 1660 SUPER (6 GB VRAM)"
    }
    with open(os.path.join(OUTPUT_DIR, "resultados_zero_shot.json"), "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUTPUT_DIR, "metricas_detalhadas.txt"), "w", encoding="utf-8") as f:
        f.write("EXPERIMENTO: ZERO-SHOT BASELINE\n")
        f.write(f"Modelo: {MODEL_ID}\n")
        f.write(f"Quantizacao: 4-bit NF4\n")
        f.write(f"Total de amostras: {len(registros)}\n")
        f.write(f"Semente: {SEED}\n\n")
        f.write(f"Acuracia: {acc:.4f}\n")
        f.write(f"F1 Macro: {f1_macro:.4f}\n")
        f.write(f"F1 Weighted: {f1_weighted:.4f}\n")
        f.write(f"Tempo total: {t_total:.1f}s\n")
        f.write(f"Tempo medio: {tempo_medio:.2f}s/CAT\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
        f.write(f"\nMATRIZ DE CONFUSAO:\n")
        f.write(str(cm))

    with open(os.path.join(OUTPUT_DIR, "tempo_processamento.txt"), "w", encoding="utf-8") as f:
        f.write(f"Experimento: Zero-Shot\n")
        f.write(f"Tempo total: {t_total:.1f} segundos\n")
        f.write(f"Tempo medio por CAT: {tempo_medio:.2f} segundos\n")
        f.write(f"Total de amostras: {len(registros)}\n")
        f.write(f"GPU: NVIDIA GeForce GTX 1660 SUPER (6 GB VRAM)\n")

    predicoes_path = os.path.join(OUTPUT_DIR, "predicoes_zero_shot.csv")
    with open(predicoes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "empresa", "grau_gravidade", "predito", "correto"])
        writer.writeheader()
        for r, pred in zip(registros, y_pred):
            writer.writerow({
                "id": r["id"],
                "empresa": r["empresa"],
                "grau_gravidade": r["grau_gravidade"],
                "predito": pred,
                "correto": "sim" if pred == r["grau_gravidade"] else "nao"
            })

    print(f"\nArquivos salvos em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
