"""Immutable parameter certificate, written before any evaluation-only DFT."""
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _freeze_unlocked(result_file, certificate):
    result_file, certificate = Path(result_file), Path(certificate)
    raw = result_file.read_bytes()
    result = json.loads(raw)
    if not result.get('completed') or len(result['vector']) != 7 or result['trials'] > 3:
        raise ValueError('Only a stopped, bounded seven-parameter fit can be frozen')
    vector_hash = hashlib.sha256(canonical(result['vector'])).hexdigest()
    stopped = {k: result[k] for k in
               ['vector', 'training_materials', 'accepted_updates', 'trials', 'states', 'history']}
    signature = hashlib.sha256(canonical(stopped)).hexdigest()
    if certificate.exists():
        prior = json.loads(certificate.read_text())
        if prior['vector_sha256'] != vector_hash or prior['training_stop_sha256'] != signature:
            raise ValueError('Frozen parameter/training state changed')
        return prior
    record = dict(frozen_at=dt.datetime.now().astimezone().isoformat(),
                  result_file=str(result_file.resolve()), result_file_sha256_at_freeze=hashlib.sha256(raw).hexdigest(),
                  vector=result['vector'], vector_sha256=vector_hash,
                  serialization='UTF-8 compact JSON7vector, sort_keys=True, allow_nan=False',
                  training_stop_sha256=signature, training_materials=result['training_materials'],
                  accepted_updates=result['accepted_updates'], trials=result['trials'],
                  evaluation_can_change_fit=False, validation_used_for_updates=False)
    certificate.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects the freeze against accidental replacement.
    with certificate.open('x') as f:
        json.dump(record, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    return record


def freeze_result(result_file, certificate):
    certificate = Path(certificate)
    certificate.parent.mkdir(parents=True, exist_ok=True)
    with certificate.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _freeze_unlocked(result_file, certificate)


def guard_LOO_validation(tag, results_root, vector):
    """Applied only to the E7 LOO validation worker tags, before GPU work."""
    for name in ['LOO_Si', 'LOO_BN', 'LOO_diamond']:
        if tag.startswith(name + '_validation'):
            root = Path(results_root).parent
            cert = freeze_result(root / (name + '_result.json'), root / (name + '_parameter_freeze.json'))
            if hashlib.sha256(canonical(vector)).hexdigest() != cert['vector_sha256']:
                raise ValueError('Validation vector differs from frozen fold')
            return cert
    return None
