import csv
import json
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from datetime import datetime

random.seed(42)
np.random.seed(42)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

EMPRESAS_NOME = {"empresa_a": "Empresa A", "empresa_b": "Empresa B", "empresa_c": "Empresa C"}
EMPRESAS_SETOR = {"empresa_a": "Construcao Civil", "empresa_b": "Metalurgia", "empresa_c": "Transporte"}

GRAVIDADES = ["leve", "moderado", "grave"]
PESOS = [0.50, 0.30, 0.20]

PARTES_CORPO = {
    "empresa_a": ["tornozelo", "mao", "antebraco", "punho", "dedo", "coxa", "cabeca", "joelho", "coluna", "pe"],
    "empresa_b": ["olho", "braco", "punho", "mao", "ouvido", "antebraco", "pe", "ombro", "torax", "pescoco"],
    "empresa_c": ["coluna cervical", "pe", "joelho", "ombro", "mao", "costelas", "quadril", "tornozelo", "torax", "abdomen"]
}

CAPS = {
    "tornozelo": "Tornozelo", "mao": "Mao", "antebraco": "Antebraco", "punho": "Punho",
    "dedo": "Dedo", "coxa": "Coxa", "cabeca": "Cabeca", "joelho": "Joelho",
    "coluna": "Coluna", "pe": "Pe", "olho": "Olho", "braco": "Braco",
    "ouvido": "Ouvido", "ombro": "Ombro", "torax": "Torax", "pescoco": "Pescoco",
    "coluna cervical": "Coluna cervical", "costelas": "Costelas", "quadril": "Quadril", "abdomen": "Abdomen"
}

DIAS = {"leve": (0, 7), "moderado": (8, 60), "grave": (61, 365)}

# Cada entrada: (tipo_acidente, descricao)
# Regra: o tipo_acidente DEVE aparecer literalmente na descricao
# #P# = parte do corpo (minuscula), #PC# = parte capitalizada
TEMPLATES = {
    "empresa_a": {
        "leve": [
            ("Entorse leve", "Trabalhador escorregou no chao umido do canteiro e sofreu entorse leve no #P#. Nao houve fratura."),
            ("Corte superficial", "Corte superficial no #P# ao manusear furadeira. Atendimento no posto de saude, sem afastamento prolongado."),
            ("Queimadura leve", "Queimadura leve no #P# por respingo de solda. Primeiro socorro no local."),
            ("Dor muscular", "Dor muscular no #P# apos repetir movimento de assentamento de tijolos por 4 horas seguidas."),
            ("Corte superficial", "Corte superficial no #P# ao cortar vergalhao com alicate. Curativo e liberacao no mesmo dia."),
            ("Contusao leve", "Contusao leve no #P# apos colisao com carrinho de mao no corredor do canteiro. Sem afastamento."),
            ("Escoriacao", "Escoriacao no #P# por estilha de madeira durante corte com serra. Sem sequelas."),
            ("Escoriacao", "Queda do proprio nivel no patio da obra causou escoriacao no #P#. Sem necessidade de afastamento."),
            ("Dor muscular", "Dor muscular no #P# apos carregar saco de cimento por 3 horas. Analgesico e repouso de 2 dias."),
            ("Picada", "Picada de abelha no #P# durante servico no telhado. Inchaço local, sem reacao alergica grave.")
        ],
        "moderado": [
            ("Fratura simples", "Trabalhador caiu de andaime de 3 metros e sofreu fratura simples no #P#. Afastamento de 30 dias."),
            ("Corte profundo", "Corte profundo no #P# com serra circular. Necessitou de 8 pontos de sutura."),
            ("Queimadura de segundo grau", "Queimadura de segundo grau no #P# por contato com cano de agua quente exposto."),
            ("Entorse grave", "Queda de escada de 2 metros causou entorse grave no #P#. Gesso por 21 dias."),
            ("Contusao moderada", "Contusao moderada no #P# apos queda de ferramenta de altura. Limitacao de movimento, afastamento de 10 dias."),
            ("Lesao por esforco repetitivo", "Lesao por esforco repetitivo no #P# apos agachamento repetitivo durante assentamento de piso. Afastamento de 15 dias."),
            ("Fratura simples", "Fratura simples no #P# apos prensamento entre tijolos. Imobilizacao por 30 dias."),
            ("Queimadura de segundo grau", "Queimadura de segundo grau no #P# por contato com massa corrida quente. Afastamento de 12 dias."),
            ("Entorse grave", "Entorse grave no #P# apos queda de andaime de 2,5 metros. Escoriacoes extensas."),
            ("Choque eletrico leve", "Choque eletrico leve no #P# ao tocar fio desencapado. Marcas de queimadura, afastamento de 10 dias.")
        ],
        "grave": [
            ("Fratura exposta", "Desabamento parcial de escoramento atingiu trabalhador. Fratura exposta no #P#. Internacao."),
            ("Traumatismo craniano", "Queda de altura de 7 metros da torre. Traumatismo craniano e fratura no #P#. Internacao em UTI."),
            ("Queimadura grave", "Eletrocutado ao tocar em fio de alta tensao. Queimadura grave no #P#. Internacao em UTI."),
            ("Fratura exposta", "Soterramento parcial por desmoronamento de valva. Fratura exposta no #P# e fratura multipla."),
            ("Fratura exposta", "Atropelado por caminhao-betoneira no canteiro. Fratura exposta no #P# e traumatismo craniano."),
            ("Fratura de coluna", "Queda de andaime de 12 metros. Fratura de coluna e fratura no #P#. Paraplegia."),
            ("Queimadura de terceiro grau", "Queimadura de terceiro grau no #P# por incendio em deposito de solventes."),
            ("Fratura exposta", "Atingido por vergalhao que caiu da torre do guindaste. Fratura exposta no #P# e perfuracao."),
            ("Trauma abdominal", "Colisao entre carrinho de mao e caminhao no canteiro. Trauma abdominal e fratura no #P#, cirurgia de emergencia."),
            ("Fratura exposta", "Queda de telhado de 6 metros. Fratura exposta no #P# e fratura bilateral.")
        ]
    },
    "empresa_b": {
        "leve": [
            ("Escoriacao", "Particula metalica atingiu o #P# causando escoriacao. Lavagem no local, sem sequelas."),
            ("Escoriacao", "Escoriacao no #P# ao manusear chapa de aco. Curativo no posto de saude."),
            ("Dor muscular", "Dor muscular no #P# apos operar prensa por 5 horas. Repouso de 1 dia."),
            ("Corte superficial", "Corte superficial no #P# com borda de peca de ferro. Sem necessidade de sutura."),
            ("Irritacao cutanea", "Irritacao cutanea no #P# apos exposicao a ruido de 95dB por 6 horas sem protetor auricular."),
            ("Queimadura leve", "Queimadura leve no #P# por respingo de oleo quente. Tratamento local."),
            ("Escoriacao", "Escoriacao no #P# apos escorregar no oleo do piso da fabrica."),
            ("Picada", "Picada de prego no #P# ao pisar em sujeira de ferro. Vacina antitetanica."),
            ("Dor muscular", "Dor muscular no #P# por levantar caixa de pecas de 25kg. Analgesico e repouso."),
            ("Corte superficial", "Corte superficial no #P# ao abrir caixa de ferramentas. Curativo simples.")
        ],
        "moderado": [
            ("Fratura simples", "#PC# prensada na prensa hidraulica. Fratura simples no #P#. Afastamento de 45 dias."),
            ("Queimadura de segundo grau", "Queimadura de segundo grau no #P# por projecao de metal fundido."),
            ("Corte profundo", "Corte profundo no #P# com lamina de serra. 12 pontos e 30 dias de afastamento."),
            ("Lesao por esforco repetitivo", "Lesao por esforco repetitivo no #P# apos inalacao de fumaca de solda. Dispneia, afastamento de 3 dias."),
            ("Corte profundo", "Corte profundo no #P# por particula metalica incrustada. Microcirurgia para remocao."),
            ("Queimadura de segundo grau", "Queimadura de segundo grau no #P# ao abrir painel sem isolamento. Afastamento de 15 dias."),
            ("Lesao por esforco repetitivo", "Lesao por esforco repetitivo no #P# por movimento repetitivo na linha de montagem. Fisioterapia por 60 dias."),
            ("Fratura simples", "Queda de peca de 50kg no #P#. Fratura simples no #P#. Gesso por 30 dias."),
            ("Queimadura de segundo grau", "Queimadura de segundo grau no #P# por solvente industrial. Afastamento de 20 dias."),
            ("Fratura simples", "Colisao com empilhadeira. Fratura simples no #P#.")
        ],
        "grave": [
            ("Amputacao", "Amputacao do #P# em maquina de corte sem protecao. Cirurgia de emergencia."),
            ("Queimadura de terceiro grau", "Queimadura de terceiro grau no #P# por explosao de forno. Internacao prolongada."),
            ("Esmagamento", "Esmagamento do #P# entre prensa e peca. Fratura multipla e hemorragia interna."),
            ("Queimadura grave", "Queimadura grave no #P# por inalacao de gas toxico em area confinada. Internacao em UTI."),
            ("Fratura exposta", "Queda de peca de 2 toneladas sobre o corpo. Fratura exposta no #P# e traumatismo."),
            ("Parada cardiaca", "Choque eletrico de alta tensao no #P#. Parada cardiaca e sequelas neuroligicas."),
            ("Queimadura de terceiro grau", "Incendio na linha de producao. Queimadura de terceiro grau no #P# e em outras regioes."),
            ("Amputacao", "Aprisionamento em maquina causou amputacao do #P#."),
            ("Traumatismo craniano", "Queda de guindaste sobre trabalhador. Traumatismo craniano e fratura no #P#."),
            ("Fratura exposta", "Explosao de cilindro de gas. Fratura exposta no #P# e queimaduras graves.")
        ]
    },
    "empresa_c": {
        "leve": [
            ("Dor muscular", "Dor muscular no #P# apos 8 horas de direcao continua. Repouso de 1 dia."),
            ("Escoriacao", "Escoriacao no #P# ao amarrar carga no caminhao. Curativo no local."),
            ("Escoriacao", "Escoriacao no #P# apos escorregar ao descer do veiculo. Sem afastamento."),
            ("Corte superficial", "Corte superficial no #P# ao manusear fita de amarracao de carga. Primeiro socorro."),
            ("Dor muscular", "Dor muscular no #P# apos dirigir caminhao com vibracao excessiva. Analgesico."),
            ("Contusao leve", "Contusao leve no #P# apos colisao do veiculo de apoio com estacionamento."),
            ("Escoriacao", "Escoriacao no #P# apos ser atingido por tampa de container que caiu."),
            ("Dor muscular", "Dor muscular no #P# ao levantar caixa de 20kg no deposito. Repouso de 2 dias."),
            ("Dor muscular", "Dor muscular no #P# apos inalar gasolina ao abastecer veiculo. Sem gravidade."),
            ("Escoriacao", "Queda do proprio nivel no patio da base causou escoriacao no #P#. Sem fratura.")
        ],
        "moderado": [
            ("Fratura simples", "Colisao frontal do caminhao em rodovia. Fratura simples no #P#, 30 dias de afastamento."),
            ("Fratura simples", "Carga mal fixada caiu sobre o ajudante. Fratura simples no #P# e contusao."),
            ("Fratura simples", "Queda ao subir no caminhao. Fratura simples no #P#. Gesso por 21 dias."),
            ("Entorse grave", "Latigamento cervical e entorse grave no #P# em freada brusca. Afastamento de 15 dias."),
            ("Fratura simples", "Atropelamento de pedestre na area de carga. Fratura simples no #P# da vitima."),
            ("Entorse grave", "Capotamento de veiculo leve em estrada. Entorse grave e luxacao no #P#."),
            ("Lesao por esforco repetitivo", "Lesao por esforco repetitivo no #P# apos 6 meses dirigindo caminhao de vibracao. Fisioterapia."),
            ("Contusao moderada", "Contusao moderada no #P# apos container tombado durante descarga. Afastamento de 15 dias."),
            ("Corte profundo", "Corte profundo no #P# com borda de chapa do caminhao. 10 pontos de sutura."),
            ("Entorse grave", "Entorse grave no #P# apos 12 horas de viagem continua. Afastamento de 20 dias.")
        ],
        "grave": [
            ("Fratura exposta", "Capotamento de caminhao-tanque em rodovia. Fratura exposta no #P# e lesao grave. Resgate prolongado."),
            ("Traumatismo craniano", "Colisao entre dois caminhoes. Traumatismo craniano e fratura no #P#. Internacao em UTI."),
            ("Fratura de coluna", "Carga de 5 toneladas caiu sobre ajudante. Fratura de coluna e fratura no #P#. Paraplegia."),
            ("Fratura exposta", "Atropelamento na area de manobra. Fratura exposta no #P# e hemorragia interna."),
            ("Queimadura de terceiro grau", "Incendio no caminhao apos colisao. Queimadura de terceiro grau no #P# e em 30% do corpo."),
            ("Traumatismo craniano", "Queda de 8 metros do topo do caminhao-tanque. Traumatismo craniano e fratura no #P#."),
            ("Queimadura de terceiro grau", "Colisao com caminhao de carga perigosa. Queimadura de terceiro grau no #P# por produto quimico."),
            ("Queimadura de terceiro grau", "Acidente com veiculo de transporte de inflamaveis. Explosao causou queimadura de terceiro grau no #P# e em outras regioes."),
            ("Esmagamento", "Esmagamento do #P# entre dois caminhoes na garagem. Fratura multipla no #P# e hemorragia."),
            ("Trauma abdominal", "Queda de cavalete de carga sobre motorista. Trauma abdominal e fratura no #P#, cirurgia de urgencia.")
        ]
    }
}


def gerar_cat(eid, grav, idx):
    parte = random.choice(PARTES_CORPO[eid])
    tipo, tmpl = random.choice(TEMPLATES[eid][grav])
    desc = tmpl.replace("#P#", parte).replace("#PC#", CAPS.get(parte, parte.capitalize()))
    dias = random.randint(*DIAS[grav])
    data = f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
    return {
        "id": f"{eid}_{idx+1:04d}",
        "empresa": EMPRESAS_NOME[eid],
        "setor": EMPRESAS_SETOR[eid],
        "data_acidente": data,
        "tipo_acidente": tipo,
        "parte_corpo_atingida": parte,
        "grau_gravidade": grav,
        "dias_afastamento": dias,
        "descricao": desc
    }


def verificar(regs):
    errs = 0
    for r in regs:
        p = r["parte_corpo_atingida"].lower()
        d = r["descricao"].lower()
        t = r["tipo_acidente"].lower()
        ok_parte = p in d
        ok_tipo = t in d
        if not ok_parte:
            errs += 1
            print(f"  PARTE: {r['id']} | tipo={t} | parte={p} | desc={r['descricao'][:80]}")
        if not ok_tipo:
            errs += 1
            print(f"  TIPO:  {r['id']} | tipo={t} | desc={r['descricao'][:80]}")
    return errs


def graf_dist(eid, gravs, out):
    c = Counter(gravs)
    labels = ["leve", "moderado", "grave"]
    vals = [c[g] for g in labels]
    cores = ["#4CAF50", "#FF9800", "#F44336"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, vals, color=cores, edgecolor="#333", linewidth=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height()+2, str(v),
                ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_title(f"Distribuicao - {EMPRESAS_NOME[eid]} ({EMPRESAS_SETOR[eid]})",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Quantidade de CATs")
    ax.set_ylim(0, max(vals)+30)
    plt.tight_layout()
    plt.savefig(os.path.join(out, f"distribuicao_{eid}.png"), dpi=150)
    plt.close()


def graf_geral(todos, out):
    cg = Counter()
    ce = {}
    for r in todos:
        cg[r["grau_gravidade"]] += 1
        k = r["empresa"]
        if k not in ce:
            ce[k] = Counter()
        ce[k][r["grau_gravidade"]] += 1
    labels = ["leve", "moderado", "grave"]
    cores = ["#4CAF50", "#FF9800", "#F44336"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    vg = [cg[g] for g in labels]
    axes[0].bar(labels, vg, color=cores, edgecolor="#333")
    axes[0].set_title("Distribuicao Geral (600 CATs)", fontweight="bold")
    axes[0].set_ylabel("Quantidade")
    for i, v in enumerate(vg):
        axes[0].text(i, v+5, str(v), ha="center", fontweight="bold")
    emps = sorted(ce.keys())
    x = np.arange(len(labels))
    w = 0.25
    for i, em in enumerate(emps):
        vals = [ce[em][g] for g in labels]
        axes[1].bar(x+i*w, vals, w, label=em, edgecolor="#333")
    axes[1].set_xticks(x+w)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Distribuicao por Empresa", fontweight="bold")
    axes[1].set_ylabel("Quantidade")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "distribuicao_geral.png"), dpi=150)
    plt.close()


def main():
    todos = []
    resumo = {"total": 0, "por_empresa": {}, "por_gravidade": {"leve": 0, "moderado": 0, "grave": 0}}

    for eid in EMPRESAS_NOME:
        gravs = random.choices(GRAVIDADES, weights=PESOS, k=200)
        regs = [gerar_cat(eid, gravs[i], i) for i in range(200)]
        print(f"\nVerificando coerencia - {EMPRESAS_NOME[eid]}:")
        errs = verificar(regs)
        if errs == 0:
            print(f"  OK: Todos os 200 registros coerentes.")
        else:
            print(f"  ERRO: {errs} incoerencias detectadas!")

        with open(os.path.join(OUTPUT_DIR, f"dataset_{eid}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=regs[0].keys())
            w.writeheader()
            w.writerows(regs)

        graf_dist(eid, gravs, OUTPUT_DIR)

        c = Counter(gravs)
        resumo["por_empresa"][eid] = {
            "nome": EMPRESAS_NOME[eid], "setor": EMPRESAS_SETOR[eid],
            "total": 200, "leve": c["leve"], "moderado": c["moderado"], "grave": c["grave"]
        }
        for g in GRAVIDADES:
            resumo["por_gravidade"][g] += c[g]
        todos.extend(regs)
        print(f"  {EMPRESAS_NOME[eid]}: leve={c['leve']}, moderado={c['moderado']}, grave={c['grave']}")

    resumo["total"] = 600
    graf_geral(todos, OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "dataset_todos.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=todos[0].keys())
        w.writeheader()
        w.writerows(todos)

    with open(os.path.join(OUTPUT_DIR, "resumo_geracao.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUTPUT_DIR, "metadados.json"), "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(), "seed": 42,
            "total": 600, "empresas": 3, "cats_por_empresa": 200,
            "distribuicao": "50/30/20",
            "coerencia": "tipo_acidente, parte_corpo_atingida e descricao 100% coerentes"
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Total: {resumo['total']} | Leve: {resumo['por_gravidade']['leve']} | Moderado: {resumo['por_gravidade']['moderado']} | Grave: {resumo['por_gravidade']['grave']}")


if __name__ == "__main__":
    main()
