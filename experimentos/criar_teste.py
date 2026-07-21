import csv
import os
import random

# Criar pasta de teste
os.makedirs("teste_cats", exist_ok=True)

# 20 CATs de exemplo (descrições realistas)
cats_exemplo = [
    "Trabalhador escorregou no chao umido do canteiro e torceu o tornozelo. Nao houve fratura.",
    "Corte superficial na mao ao manusear furadeira. Atendimento no posto de saude.",
    "Queimadura leve no antebraco por respingo de solda. Primeiro socorro no local.",
    "Trabalhador sentiu dor no punho apos repetir movimento de assentamento de tijolos.",
    "Pequeno corte no dedo ao cortar vergalhao com alicate. Curativo e liberacao no mesmo dia.",
    "Hematoma na coxa apos colisao com carrinho de mao no corredor do canteiro.",
    "Escoriacao no braco por estilha de madeira durante corte com serra. Sem sequelas.",
    "Queda do proprio nivel no patio da obra. Joelho escoriado, sem necessidade de afastamento.",
    "Dor lombar apos carregar saco de cimento por 3 horas. Analgesico e repouso de 2 dias.",
    "Picada de abelha no braco durante servico no telhado. Inchaço local.",
    "Trabalhador caiu de andaime de 3 metros. Fratura no punho esquerdo. Afastamento de 30 dias.",
    "Corte profundo na mao direita com serra circular. Necessitou de 8 pontos de sutura.",
    "Queimadura de segundo grau no braco por contato com cano de agua quente exposto.",
    "Queda de escada de 2 metros. Entorse grave no joelho. Gesso por 21 dias.",
    "Contusao no ombro apos queda de ferramenta de altura. Limitacao de movimento.",
    "Lesao no joelho apos agachamento repetitivo durante assentamento de piso. Afastamento de 15 dias.",
    "Fratura no dedo apos prensamento entre tijolos. Imobilizacao por 30 dias.",
    "Choque eletrico no antebraco ao tocar fio desencapado. Afastamento de 10 dias.",
    "Desabamento parcial de escoramento atingiu trabalhador. Fratura exposta na perna. Internacao.",
    "Queda de altura de 7 metros da torre de edificio em construcao. Traumatismo craniano grave."
]

# Classificações esperadas (para verificação)
classificacoes_esperadas = [
    "leve", "leve", "leve", "leve", "leve",
    "leve", "leve", "leve", "leve", "leve",
    "moderado", "moderado", "moderado", "moderado", "moderado",
    "moderado", "moderado", "moderado",
    "grave", "grave"
]

# Criar arquivo CSV
with open("teste_cats/cats_exemplo.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "descricao", "classificacao_esperada"])
    for i, (cat, cls) in enumerate(zip(cats_exemplo, classificacoes_esperadas), 1):
        writer.writerow([f"CAT_{i:03d}", cat, cls])

print(f"Arquivo criado: teste_cats/cats_exemplo.csv")
print(f"Total de CATs: {len(cats_exemplo)}")
print(f"\nClassificações esperadas:")
print(f"  Leve: {classificacoes_esperadas.count('leve')}")
print(f"  Moderado: {classificacoes_esperadas.count('moderado')}")
print(f"  Grave: {classificacoes_esperadas.count('grave')}")
print(f"\nPrimeiras 5 CATs:")
for i, (cat, cls) in enumerate(zip(cats_exemplo[:5], classificacoes_esperadas[:5]), 1):
    print(f"  {i}. [{cls.upper()}] {cat[:80]}...")
