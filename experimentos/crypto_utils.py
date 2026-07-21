import os
import json
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

class AES256Crypto:
    """Classe para criptografia AES-256 dos parametros LoRA."""
    
    def __init__(self, key=None):
        """Inicializa com uma chave de 256 bits (32 bytes)."""
        if key is None:
            self.key = os.urandom(32)  # Gera chave aleatoria
        else:
            self.key = key
    
    def encrypt(self, data: bytes) -> bytes:
        """Criptografa dados com AES-256-CBC."""
        # Adicionar padding
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()
        
        # Gerar IV aleatorio
        iv = os.urandom(16)
        
        # Criptografar
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        # Retornar IV + ciphertext
        return iv + ciphertext
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Descriptografa dados com AES-256-CBC."""
        # Extrair IV
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]
        
        # Descriptografar
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remover padding
        unpadder = padding.PKCS7(128).unpadder()
        data = unpadder.update(padded_data) + unpadder.finalize()
        
        return data
    
    def encrypt_dict(self, data: dict) -> str:
        """Criptografa um dicionario e retorna como string base64."""
        json_bytes = json.dumps(data).encode('utf-8')
        encrypted = self.encrypt(json_bytes)
        return base64.b64encode(encrypted).decode('utf-8')
    
    def decrypt_dict(self, encrypted_str: str) -> dict:
        """Descriptografa uma string base64 e retorna o dicionario."""
        encrypted_bytes = base64.b64decode(encrypted_str)
        decrypted = self.decrypt(encrypted_bytes)
        return json.loads(decrypted.decode('utf-8'))
    
    def get_key_hex(self) -> str:
        """Retorna a chave em formato hexadecimal."""
        return self.key.hex()
    
    @classmethod
    def from_hex(cls, hex_key: str):
        """Cria uma instancia a partir de uma chave hexadecimal."""
        key = bytes.fromhex(hex_key)
        return cls(key)


class IPFSStorage:
    """Classe simulada para armazenamento IPFS."""
    
    def __init__(self):
        self.storage = {}
        self.cid_counter = 0
    
    def add(self, data: bytes) -> str:
        """Adiciona dados e retorna um CID (Content Identifier)."""
        import hashlib
        cid = hashlib.sha256(data).hexdigest()[:16]
        self.storage[cid] = data
        return cid
    
    def get(self, cid: str) -> bytes:
        """Recupera dados pelo CID."""
        if cid not in self.storage:
            raise ValueError(f"CID {cid} nao encontrado")
        return self.storage[cid]
    
    def list_cids(self) -> list:
        """Lista todos os CIDs armazenados."""
        return list(self.storage.keys())


def encrypt_lora_params(params: dict, crypto: AES256Crypto) -> str:
    """Criptografa parametros LoRA para transporte."""
    return crypto.encrypt_dict(params)


def decrypt_lora_params(encrypted_str: str, crypto: AES256Crypto) -> dict:
    """Descriptografa parametros LoRA recebidos."""
    return crypto.decrypt_dict(encrypted_str)


def store_params_ipfs(params: dict, crypto: AES256Crypto, ipfs: IPFSStorage) -> str:
    """Criptografa e armazena parametros no IPFS."""
    encrypted = crypto.encrypt(json.dumps(params).encode('utf-8'))
    cid = ipfs.add(encrypted)
    return cid


def retrieve_params_ipfs(cid: str, crypto: AES256Crypto, ipfs: IPFSStorage) -> dict:
    """Recupera e descriptografa parametros do IPFS."""
    encrypted = ipfs.get(cid)
    decrypted = crypto.decrypt(encrypted)
    return json.loads(decrypted.decode('utf-8'))


if __name__ == "__main__":
    # Teste da criptografia
    print("Testando criptografia AES-256...")
    
    # Criar instancia
    crypto = AES256Crypto()
    print(f"Chave gerada: {crypto.get_key_hex()[:16]}...")
    
    # Dados de teste
    dados_teste = {"param1": [1.0, 2.0, 3.0], "param2": [4.0, 5.0, 6.0]}
    print(f"Dados originais: {dados_teste}")
    
    # Criptografar
    encrypted = crypto.encrypt_dict(dados_teste)
    print(f"Dados criptografados: {encrypted[:50]}...")
    
    # Descriptografar
    decrypted = crypto.decrypt_dict(encrypted)
    print(f"Dados descriptografados: {decrypted}")
    
    # Verificar
    assert dados_teste == decrypted, "ERRO: Dados nao coincidem!"
    print("OK: Criptografia funcionando corretamente!")
    
    # Teste IPFS
    print("\nTestando IPFS simulado...")
    ipfs = IPFSStorage()
    cid = store_params_ipfs(dados_teste, crypto, ipfs)
    print(f"CID: {cid}")
    
    recovered = retrieve_params_ipfs(cid, crypto, ipfs)
    print(f"Dados recuperados: {recovered}")
    assert dados_teste == recovered, "ERRO: Dados IPFS nao coincidem!"
    print("OK: IPFS + Criptografia funcionando!")
