"""Fernet E2E encryption for colab-autopilot."""

import json
from cryptography.fernet import Fernet


def generate_key() -> bytes:
    return Fernet.generate_key()


def encrypt(key: bytes, data: dict) -> bytes:
    f = Fernet(key)
    plaintext = json.dumps(data).encode("utf-8")
    return f.encrypt(plaintext)


def decrypt(key: bytes, ciphertext: bytes) -> dict:
    f = Fernet(key)
    plaintext = f.decrypt(ciphertext)
    return json.loads(plaintext.decode("utf-8"))
