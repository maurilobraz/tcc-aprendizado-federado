import csv
import json
import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from crypto_utils import AES256Crypto, IPFSStorage, store_params_ipfs, retrieve_params_ipfs

# Configuracao
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LABELS = ["leve", "moderado", "grave"]
CLIENTS = ["empresa_a", "empresa_b", "empresa_c"]
ROUNDS = 10
EPOCHS = 1
LR = 1e-4
LORA_R = 8
LORA_DROPOUT = 0.1
BATCH = 4
MAXLEN = 256
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Inicializar criptografia e IPFS
crypto = AES256Crypto()
ipfs = IPFSStorage()

print("="*60)
print("FL COM CRIPTOGRAFIA AES-256 E IPFS")
print("="*60)
print(f"Chave AES-256: {crypto.get_key_hex()[:16]}...")
print(f"IPFS: Simulado (em memoria)")


def load_data(eid, n_train, n_test):
    rows = []
    with open(f"dados/dataset_{eid}.csv", "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    random.shuffle(rows)
    return rows[:n_train], rows[n_train:n_train+n_test]


def make_model():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                                device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
    mdl.gradient_checkpointing_enable()
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=16, lora_dropout=LORA_DROPOUT,
                     target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], bias="none")
    mdl = get_peft_model(mdl, cfg)
    return mdl, tok


def train_epoch(mdl, tok, data, opt, dev):
    mdl.train()
    tot = 0
    nb = 0
    random.shuffle(data)
    for i in range(0, len(data), BATCH):
        batch = data[i:i+BATCH]
        texts = [f"Classifique a gravidade deste acidente: {r['descricao']}\nResposta: {LABELS.index(r['grau_gravidade'])}" for r in batch]
        enc = tok(texts, truncation=True, padding="max_length", max_length=MAXLEN, return_tensors="pt").to(dev)
        out = mdl(**enc, labels=enc["input_ids"])
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        tot += out.loss.item()
        nb += 1
    return tot / max(nb, 1)


def evaluate(mdl, tok, data, dev):
    mdl.eval()
    yt = []
    yp = []
    with torch.no_grad():
        for r in data:
            msgs = [{"role":"system","content":"Responda APENAS com: leve, moderado ou grave."},
                    {"role":"user","content":f"Classifique a gravidade deste acidente: {r['descricao']}"}]
            inp = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                      return_tensors="pt", truncation=True, max_length=MAXLEN).to(dev)
            out = mdl.generate(**inp, max_new_tokens=10, do_sample=False)
            resp = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
            pred = 1
            for j, l in enumerate(LABELS):
                if l in resp:
                    pred = j
                    break
            yt.append(LABELS.index(r["grau_gravidade"]))
            yp.append(pred)
    return accuracy_score(yt, yp), f1_score(yt, yp, average="macro", zero_division=0), \
           confusion_matrix(yt, yp, labels=[0,1,2]), yt, yp


def get_state(m):
    return {k: v.cpu().clone() for k,v in m.state_dict().items() if "lora" in k and "base_layer" not in k}


def set_state(m, s):
    model_keys = {k for k in m.state_dict().keys() if "lora" in k and "base_layer" not in k}
    filtered_s = {k: v for k, v in s.items() if k in model_keys}
    m.load_state_dict(filtered_s, strict=False)


def avg_states(states):
    keys = [k for k in states[0].keys() if "base_layer" not in k]
    return {k: torch.stack([s[k] for s in states]).float().mean(0).to(states[0][k].dtype) for k in keys}


def plot_cm(cm, path, title):
    fig, ax = plt.subplots(figsize=(6,5))
    ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(LABELS)
    ax.set_yticks([0,1,2])
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predito", fontsize=10)
    ax.set_ylabel("Real", fontsize=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j]>cm.max()/2 else "black", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()


def plot_curve(rounds, accs, path, title):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(rounds, accs, "o-", color="#3498DB", lw=2, markersize=6, label="Acurácia")
    ax.set_xlabel("Rodada", fontsize=11)
    ax.set_ylabel("Acurácia", fontsize=11)
    ax.set_ylim(0.4, 0.9)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()


def run_fl_experiment_encrypted(n_train, exp_name):
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO: {exp_name} ({n_train} CATs/empresa) COM CRIPTOGRAFIA")
    print(f"{'='*60}")
    
    train_d, test_d = {}, {}
    for c in CLIENTS:
        tr, te = load_data(c, n_train, 200-n_train)
        train_d[c] = tr
        test_d[c] = te
        print(f"  {c}: {len(tr)} treino, {len(te)} teste")
    
    print("\nCarregando modelo...")
    mdl, tok = make_model()
    dev = "cuda"
    print(f"Modelo carregado: {MODEL}")
    
    t0 = time.time()
    hist_r, hist_a, hist_f, hist_l = [], [], [], []
    ipfs_cids = []  # Armazenar CIDs do IPFS
    
    print(f"\n--- TREINAMENTO FEDERADO COM CRIPTOGRAFIA ({ROUNDS} rounds) ---")
    for rnd in range(1, ROUNDS+1):
        print(f"\nRound {rnd}/{ROUNDS}")
        states = []
        rloss = 0
        
        for ci, c in enumerate(CLIENTS):
            print(f"  Treinando {c} ({ci+1}/3)...")
            opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, mdl.parameters()), lr=LR, weight_decay=0.01)
            loss = 0
            for ep in range(EPOCHS):
                loss += train_epoch(mdl, tok, train_d[c], opt, dev)
            loss /= EPOCHS
            
            # Obter parametros LoRA
            lora_params = get_state(mdl)
            
            # CRIPTOGRAFAR parametros antes de enviar
            print(f"    Criptografando parametros com AES-256...")
            cid = store_params_ipfs(lora_params, crypto, ipfs)
            ipfs_cids.append({"round": rnd, "client": c, "cid": cid})
            print(f"    Armazenado no IPFS: CID={cid[:16]}...")
            
            # Recuperar parametros descriptografados
            recovered_params = retrieve_params_ipfs(cid, crypto, ipfs)
            states.append(recovered_params)
            
            rloss += loss
            print(f"    {c}: loss={loss:.4f}")
            torch.cuda.empty_cache()
        
        print("  Agregando (FedAvg)...")
        set_state(mdl, avg_states(states))
        
        print("  Avaliando modelo global...")
        acc, f1, cm, yt, yp = evaluate(mdl, tok, sum([test_d[c] for c in CLIENTS], []), dev)
        torch.cuda.empty_cache()
        
        hist_r.append(rnd)
        hist_a.append(acc)
        hist_f.append(f1)
        hist_l.append(rloss/3)
        print(f"  Resultado: Acc={acc:.4f} F1={f1:.4f} Loss={rloss/3:.4f} ({time.time()-t0:.0f}s)")
    
    print(f"\n--- TREINAMENTO ISOLADO (baseline) ---")
    accs_iso = []
    for ci, c in enumerate(CLIENTS):
        print(f"  Treinando {c} isolado ({ci+1}/3)...")
        m2, t2 = make_model()
        opt2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, m2.parameters()), lr=LR, weight_decay=0.01)
        for ep in range(EPOCHS*ROUNDS):
            train_epoch(m2, t2, train_d[c], opt2, dev)
        a, _, _, _, _ = evaluate(m2, t2, test_d[c], dev)
        accs_iso.append(a)
        print(f"    {c}: acc={a:.4f}")
        del m2
        torch.cuda.empty_cache()
    acc_iso = np.mean(accs_iso)
    print(f"  Média isolado: {acc_iso:.4f}")
    
    print(f"\n--- GERANDO GRÁFICOS ---")
    plot_cm(cm, f"matriz_confusao_{exp_name}.png", f"Matriz de Confusão - {exp_name}")
    plot_curve(hist_r, hist_a, f"learning_curve_{exp_name}.png", f"Curva de Aprendizado - {exp_name}")
    
    tt = time.time()-t0
    melhor_round = hist_r[np.argmax(hist_a)]
    melhor_acc = max(hist_a)
    melhor_f1 = hist_f[np.argmax(hist_a)]
    
    res = {
        "experimento": exp_name,
        "modelo": MODEL,
        "criptografia": "AES-256",
        "ipfs": "Simulado",
        "lora_r": LORA_R,
        "rounds": ROUNDS,
        "epochs": EPOCHS,
        "n_train": n_train,
        "n_test": 200-n_train,
        "batch_size": BATCH,
        "learning_rate": LR,
        "melhor_acuracia": round(melhor_acc, 4),
        "melhor_f1": round(melhor_f1, 4),
        "melhor_round": melhor_round,
        "acuracia_final": round(hist_a[-1], 4),
        "f1_macro_final": round(hist_f[-1], 4),
        "acuracia_isolado": round(acc_iso, 4),
        "ganho_fl": round(melhor_acc-acc_iso, 4),
        "tempo_total_seg": round(tt, 1),
        "historico_acuracia": [round(a, 4) for a in hist_a],
        "historico_f1": [round(x, 4) for x in hist_f],
        "historico_loss": [round(x, 4) for x in hist_l],
        "matriz_confusao": cm.tolist(),
        "ipfs_cids": ipfs_cids,
        "gpu": "NVIDIA T4 15GB",
        "semente": SEED
    }
    with open(f"resultados_{exp_name}.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    
    print(f"\n{exp_name} CONCLUÍDO!")
    print(f"  Melhor Acc={melhor_acc:.4f} (Round {melhor_round}) F1={melhor_f1:.4f}")
    print(f"  Isolado={acc_iso:.4f} Ganho={melhor_acc-acc_iso:.4f} Tempo={tt:.0f}s")
    print(f"  Parâmetros criptografados armazenados no IPFS: {len(ipfs_cids)} CIDs")
    return res


if __name__ == "__main__":
    print("\n" + "="*60)
    print("INICIANDO EXPERIMENTOS COM CRIPTOGRAFIA")
    print("="*60)
    
    # Executar FL 50
    resultados_fl_50 = run_fl_experiment_encrypted(50, "fl_50_crypto")
    
    print("\n" + "="*60)
    print("EXPERIMENTO FL 50 COM CRIPTOGRAFIA CONCLUÍDO")
    print("="*60)
    print(f"Melhor Acc: {resultados_fl_50['melhor_acuracia']}")
    print(f"IPFS CIDs: {len(resultados_fl_50['ipfs_cids'])}")
