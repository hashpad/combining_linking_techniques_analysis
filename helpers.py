import hashlib

def hashStringSha256(input_string):
    input_string = input_string.replace('\"', '\\\"')
    try:
        digest = hashlib.sha256()
        digest.update(input_string.encode())
        hash_bytes = digest.digest()
        hex_string = ''.join(format(b, '02x') for b in hash_bytes)
        return hex_string
    except Exception as e:
        print(e)
        return None
