import json
import hashlib


def dict_hash(input: dict):
    return hashlib.sha1(json.dumps(input, sort_keys=True).encode('utf-8')).hexdigest()
