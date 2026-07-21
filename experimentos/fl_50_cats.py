import csv, json, os, time, random, numpy as np, torch
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(OUT), "01_geracao_dados")
PROG = os.path.join(OUT, "progresso.txt")
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LABELS = ["leve", "moderado", "grave"]
CLIENTS = ["empresa_a", "empresa_b", "empresa_c"]
N_TRAIN, N_TEST = 50, 150
ROUNDS, EPOCHS, LR, LORA_R = 10, 3, 2e-4, 4
BATCH, MAXLEN = 2, 256

random.seed(42); np.random.seed(42); torch.manual_seed(42)


def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    with open(PROG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_data(eid):
    rows = []
    with open(os.path.join(DATA, f"dataset_{eid}.csv"), "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    random.shuffle(rows)
    return rows[:N_TRAIN], rows[N_TRAIN:N_TRAIN+N_TEST]


def make_model():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb,
                                                device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
    mdl.gradient_checkpointing_enable()
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=16, lora_dropout=0.05,
                     target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], bias="none")
    mdl = get_peft_model(mdl, cfg)
    return mdl, tok


def train_epoch(mdl, tok, data, opt, dev):
    mdl.train(); tot = 0; nb = 0
    random.shuffle(data)
    for i in range(0, len(data), BATCH):
        batch = data[i:i+BATCH]
        texts = [f"Classifique a gravidade deste acidente: {r['descricao']}\nResposta: {LABELS.index(r['grau_gravidade'])}" for r in batch]
        enc = tok(texts, truncation=True, padding="max_length", max_length=MAXLEN, return_tensors="pt").to(dev)
        out = mdl(**enc, labels=enc["input_ids"])
        opt.zero_grad(); out.loss.backward(); opt.step()
        tot += out.loss.item(); nb += 1
    return tot / max(nb, 1)


def evaluate(mdl, tok, data, dev):
    mdl.eval(); yt = []; yp = []
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
                if l in resp: pred = j; break
            yt.append(LABELS.index(r["grau_gravidade"])); yp.append(pred)
    return accuracy_score(yt, yp), f1_score(yt, yp, average="macro", zero_division=0), \
           confusion_matrix(yt, yp, labels=[0,1,2]), yt, yp


def get_state(m): return {k: v.cpu().clone() for k,v in m.state_dict().items() if "lora" in k}
def set_state(m, s):
    d = m.state_dict(); d.update(s); m.load_state_dict(d)
def avg_states(states):
    return {k: torch.stack([s[k] for s in states]).float().mean(0).to(states[0][k].dtype) for k in states[0]}


def plot_cm(cm, path, title):
    fig, ax = plt.subplots(figsize=(6,5))
    ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_title(title, fontweight="bold"); ax.set_xticks([0,1,2]); ax.set_xticklabels(LABELS)
    ax.set_yticks([0,1,2]); ax.set_yticklabels(LABELS); ax.set_xlabel("Predito"); ax.set_ylabel("Real")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j]>cm.max()/2 else "black", fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_curve(rounds, accs, f1s, path):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(rounds, accs, "o-", color="#3498DB", lw=2, label="Acuracia")
    ax.plot(rounds, f1s, "s-", color="#E74C3C", lw=2, label="F1 Macro")
    ax.set_xlabel("Rodada"); ax.set_ylabel("Valor"); ax.set_ylim(0,1.05)
    ax.set_title(f"Curva FL ({N_TRAIN} CATs/empresa)", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_comp(accs_fl, acc_iso, path):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(range(1,len(accs_fl)+1), accs_fl, "o-", color="#3498DB", lw=2, label="FL (FedAvg)")
    ax.axhline(acc_iso, color="#E74C3C", ls="--", lw=2, label=f"Isolado ({acc_iso:.1%})")
    ax.set_xlabel("Rodada"); ax.set_ylabel("Acuracia"); ax.set_ylim(0,1.05)
    ax.set_title(f"FL vs. Isolado ({N_TRAIN} CATs/empresa)", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    with open(PROG, "w", encoding="utf-8") as f:
        f.write(f"=== FL {N_TRAIN} CATs ===\nInicio: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    log(f"[0%] FL {N_TRAIN} CATs | {ROUNDS} rounds | LoRA r={LORA_R}")

    train_d, test_d = {}, {}
    for c in CLIENTS:
        tr, te = load_data(c); train_d[c] = tr; test_d[c] = te
        log(f"  {c}: {len(tr)} treino, {len(te)} teste")

    log("[10%] Carregando modelo...")
    mdl, tok = make_model()
    dev = "cuda"
    log("[15%] Modelo carregado")

    t0 = time.time()
    hist_r, hist_a, hist_f, hist_l = [], [], [], []

    for rnd in range(1, ROUNDS+1):
        pct = 15 + (rnd-1)*7
        log(f"[{pct}%] Round {rnd}/{ROUNDS}")
        states = []; rloss = 0

        for ci, c in enumerate(CLIENTS):
            log(f"  [{pct}%] {c} ({ci+1}/3)...")
            opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, mdl.parameters()), lr=LR)
            loss = 0
            for ep in range(EPOCHS):
                loss += train_epoch(mdl, tok, train_d[c], opt, dev)
            loss /= EPOCHS
            states.append(get_state(mdl)); rloss += loss
            log(f"  {c}: loss={loss:.4f}")
            torch.cuda.empty_cache()

        log(f"  [{pct+3}%] FedAvg...")
        set_state(mdl, avg_states(states))

        log(f"  [{pct+4}%] Avaliando...")
        acc, f1, cm, yt, yp = evaluate(mdl, tok, sum([test_d[c] for c in CLIENTS], []), dev)
        torch.cuda.empty_cache()

        hist_r.append(rnd); hist_a.append(acc); hist_f.append(f1); hist_l.append(rloss/3)
        log(f"  [{pct+7}%] Acc={acc:.4f} F1={f1:.4f} Loss={rloss/3:.4f} ({time.time()-t0:.0f}s)")

        with open(os.path.join(OUT, f"resultados_fl_{N_TRAIN}_parcial.json"), "w") as f:
            json.dump({"rounds":rnd,"acc":[round(a,4) for a in hist_a],"f1":[round(x,4) for x in hist_f],
                        "loss":[round(x,4) for x in hist_l],"tempo":round(time.time()-t0,1)}, f, indent=2)

    log("[85%] Treinamento isolado...")
    accs_iso = []
    for ci, c in enumerate(CLIENTS):
        log(f"  [{85+ci*3}%] {c} isolado...")
        m2, t2 = make_model()
        opt2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, m2.parameters()), lr=LR)
        for _ in range(EPOCHS*ROUNDS):
            train_epoch(m2, t2, train_d[c], opt2, dev)
        a, _, _, _, _ = evaluate(m2, t2, test_d[c], dev)
        accs_iso.append(a); log(f"  {c}: acc={a:.4f}")
        del m2; torch.cuda.empty_cache()
    acc_iso = np.mean(accs_iso)

    log("[95%] Gerando graficos...")
    plot_cm(cm, os.path.join(OUT, f"matriz_confusao_fl_{N_TRAIN}.png"), f"FL {N_TRAIN} CATs")
    plot_curve(hist_r, hist_a, hist_f, os.path.join(OUT, f"learning_curve_fl_{N_TRAIN}.png"))
    plot_comp(hist_a, acc_iso, os.path.join(OUT, f"comparativo_fl_vs_isolado_{N_TRAIN}.png"))

    tt = time.time()-t0
    res = {"exp":f"fl_{N_TRAIN}","acc_final":round(hist_a[-1],4),"f1_final":round(hist_f[-1],4),
           "acc_iso":round(acc_iso,4),"ganho":round(hist_a[-1]-acc_iso,4),"tempo":round(tt,1),
           "hist_acc":[round(a,4) for a in hist_a],"hist_f1":[round(x,4) for x in hist_f],
           "hist_loss":[round(x,4) for x in hist_l],"cm":cm.tolist(),"gpu":"GTX 1660 SUPER"}
    with open(os.path.join(OUT, f"resultados_fl_{N_TRAIN}.json"), "w") as f:
        json.dump(res, f, indent=2)

    with open(os.path.join(OUT, "metricas_detalhadas.txt"), "w") as f:
        f.write(f"FL {N_TRAIN} CATs\nAcc: {hist_a[-1]:.4f}\nF1: {hist_f[-1]:.4f}\nIso: {acc_iso:.4f}\nGanho: {hist_a[-1]-acc_iso:.4f}\n")
        f.write(classification_report(yt, yp, target_names=LABELS, zero_division=0))

    log(f"[100%] CONCLUIDO! Acc={hist_a[-1]:.4f} F1={hist_f[-1]:.4f} Iso={acc_iso:.4f} Ganho={hist_a[-1]-acc_iso:.4f} Tempo={tt:.0f}s")

if __name__ == "__main__":
    main()
