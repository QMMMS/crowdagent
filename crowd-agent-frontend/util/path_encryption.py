#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import hashlib


def simple_encrypt(text, key):
    key_hash = int(hashlib.md5(key.encode('utf-8')).hexdigest()[:8], 16) % 256
    text_bytes = text.encode('utf-8')
    encrypted_bytes = bytearray()
    
    for byte in text_bytes:
        encrypted_byte = (byte + key_hash) % 256
        encrypted_bytes.append(encrypted_byte)
    
    return base64.b64encode(bytes(encrypted_bytes)).decode('utf-8')


def simple_decrypt(encrypted_text, key):
    key_hash = int(hashlib.md5(key.encode('utf-8')).hexdigest()[:8], 16) % 256
    
    try:
        encrypted_bytes = base64.b64decode(encrypted_text.encode('utf-8'))
    except Exception as e:
        return f"Decryption failed: Invalid encrypted string - {str(e)}"
    
    decrypted_bytes = bytearray()
    for byte in encrypted_bytes:
        decrypted_byte = (byte - key_hash) % 256
        decrypted_bytes.append(decrypted_byte)
    
    try:
        return bytes(decrypted_bytes).decode('utf-8')
    except UnicodeDecodeError:
        return "Decryption failed: Encoding error"


def encrypt_path(path, key):
    return simple_encrypt(path, key)


def decrypt_path(encrypted_path, key):
    return simple_decrypt(encrypted_path, key)


if __name__ == "__main__":
    original_path = "xxx/xxx/a.jpg"
    secret_key = "xxxyyyzzz"
    
    print("=== Path Encryption and Decryption Test ===")
    print(f"Original Path: {original_path}")
    print(f"Secret Key: {secret_key}")
    print()
    
    encrypted_path = encrypt_path(original_path, secret_key)
    print(f"Encrypted: {encrypted_path}")
    print()
    
    decrypted_path = decrypt_path(encrypted_path, secret_key)
    print(f"Decrypted: {decrypted_path}")
    print()
    
    if original_path == decrypted_path:
        print("✅ Encryption and decryption successful!")
    else:
        print("❌ Encryption and decryption failed!")
    
    print("\n=== Other Tests ===")
    
    test_strings = [
        "Hello World!",
        "C:\\Users\\username\\Documents\\file.pdf",
        "https://example.com/api/endpoint?param=value",
        "中文测试",
        "/home/user/Documents/file.txt"
    ]
    
    for test_str in test_strings:
        encrypted = simple_encrypt(test_str, secret_key)
        decrypted = simple_decrypt(encrypted, secret_key)
        print(f"Original: {test_str}")
        print(f"Encrypted: {encrypted}")
        print(f"Decrypted: {decrypted}")
        print(f"Verification: {'✅' if test_str == decrypted else '❌'}")
        print("-" * 50) 