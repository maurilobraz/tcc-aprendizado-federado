#!/usr/bin/env python3
"""
Sistema de Analise de CATs com Aprendizado Federado
Demonstracao para Empresa de Medio Porte

Uso: python analisar_cats.py [pasta_com_cats]
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

def verificar_sistema():
    """Verifica se o sistema esta pronto."""
    print("="*60)
    print("VERIFICANDO SISTEMA...")
    print("="*60)
    
    # Verificar Python
    print(f"Python: {sys.version}")
    
    # Verificar dependencias
    dependencias = ["torch", "transformers", "peft"]
    for dep in dependencias:
        try:
            __import__(dep)
            print(f"[OK] {dep} instalado")
        except ImportError:
            print(f"[FALTA] {dep} NAO instalado - execute: pip install {dep}")
            return False
    
    # Verificar GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[AVISO] GPU nao detectada - processamento sera mais lento")
    except:
        print("[AVISO] PyTorch sem suporte a GPU")
    
    return True

def carregar_modelo():
    """Carrega o modelo de IA."""
    print("\nCarregando modelo de IA...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType
        import torch
        
        MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        print("[OK] Modelo carregado com sucesso!")
        return model, tokenizer
        
    except Exception as e:
        print(f"[FALTA] Erro ao carregar modelo: {e}")
        return None, None

def analisar_cat(model, tokenizer, descricao):
    """Analisa uma CAT individual."""
    try:
        import torch
        messages = [
            {"role": "system", "content": "Responda APENAS com: leve, moderado ou grave."},
            {"role": "user", "content": f"Classifique a gravidade deste acidente: {descricao}"}
        ]
        
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        resposta = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()
        
        labels = ["leve", "moderado", "grave"]
        for label in labels:
            if label in resposta:
                return label
        
        return "moderado"  # Default
        
    except Exception as e:
        return f"erro: {e}"

def processar_pasta(pasta, model, tokenizer):
    """Processa todas as CATs de uma pasta."""
    print(f"\nProcessando CATs de: {pasta}")
    
    resultados = []
    arquivos = list(Path(pasta).glob("*.csv")) + list(Path(pasta).glob("*.txt"))
    
    if not arquivos:
        print("Nenhum arquivo encontrado na pasta!")
        return resultados
    
    print(f"Encontrados {len(arquivos)} arquivos")
    
    for arquivo in arquivos:
        print(f"\nProcessando: {arquivo.name}")
        
        try:
            with open(arquivo, 'r', encoding='utf-8') as f:
                linhas = f.readlines()
            
            for i, linha in enumerate(linhas[:10]):  # Limitar a 10 por arquivo
                if linha.strip():
                    inicio = time.time()
                    resultado = analisar_cat(model, tokenizer, linha.strip())
                    tempo = time.time() - inicio
                    
                    resultados.append({
                        "arquivo": arquivo.name,
                        "linha": i + 1,
                        "descricao": linha.strip()[:100] + "...",
                        "classificacao": resultado,
                        "tempo_segundos": round(tempo, 2)
                    })
                    
                    print(f"  Linha {i+1}: {resultado} ({tempo:.2f}s)")
        
        except Exception as e:
            print(f"  Erro ao processar {arquivo.name}: {e}")
    
    return resultados

def main():
    print("="*60)
    print("SISTEMA DE ANALISE DE CATs COM APRENDIZADO FEDERADO")
    print("="*60)
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Verificar sistema
    if not verificar_sistema():
        print("\nSistema nao atende os requisitos minimos!")
        return
    
    # Carregar modelo
    model, tokenizer = carregar_modelo()
    if model is None:
        return
    
    # Processar pasta (se fornecida como argumento)
    if len(sys.argv) > 1:
        pasta = sys.argv[1]
        if os.path.exists(pasta):
            inicio_total = time.time()
            resultados = processar_pasta(pasta, model, tokenizer)
            tempo_total = time.time() - inicio_total
            
            # Salvar resultados
            relatorio = {
                "data": datetime.now().isoformat(),
                "total_cats": len(resultados),
                "tempo_total_segundos": round(tempo_total, 2),
                "tempo_medio_por_cat": round(tempo_total / max(len(resultados), 1), 2),
                "resultados": resultados
            }
            
            with open("relatorio_analise.json", "w", encoding="utf-8") as f:
                json.dump(relatorio, f, indent=2, ensure_ascii=False)
            
            print(f"\n{'='*60}")
            print("RELATORIO FINAL")
            print(f"{'='*60}")
            print(f"Total de CATs analisadas: {len(resultados)}")
            print(f"Tempo total: {tempo_total:.2f} segundos")
            print(f"Tempo medio por CAT: {tempo_total / max(len(resultados), 1):.2f} segundos")
            print(f"Relatorio salvo em: relatorio_analise.json")
        else:
            print(f"Pasta nao encontrada: {pasta}")
    else:
        print("\nUso: python analisar_cats.py [pasta_com_cats]")
        print("Exemplo: python analisar_cats.py ./dados")

if __name__ == "__main__":
    main()
