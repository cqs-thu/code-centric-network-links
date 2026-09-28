import numpy as np
import stim
import pymatching
from tqdm import tqdm
import time
from qldpc import decoder, codes

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from circuits.surface import *
from circuits.steane import *
from circuits.DEJMPS import *

def cal_DEJMPS_detection_per_basis(depth, p_local=0.001, p_trans=0.001, shots=1000000, batch=2_000_000):
    """Acceptance probability and conditional logical errors of the distilled pair, read out without noise.
    Returns (acceptance, zz, xx): zz (xx) is the probability that the ZZ (XX) parity is wrong among accepted
    runs, i.e. the error of a |0> (|+>) teleported through the pair and read out in the Z (X) basis.
    The reported logical error rate is the average (zz + xx) / 2, as for the surface and Steane codes."""
    circuit = epr_distillation(n=depth, p_local=p_local, p_trans=p_trans, fidelity_basis="ZX", noisy_readout=False)
    sampler = circuit.compile_detector_sampler()
    n = accepted = f_zz = f_xx = 0
    while n < shots:
        m = min(batch, shots - n)
        det, obs = sampler.sample(m, separate_observables=True)
        keep = ~det.any(axis=1)
        accepted += int(keep.sum())
        f_zz += int(obs[keep, 1].sum())   # observable 1 is the ZZ parity
        f_xx += int(obs[keep, 0].sum())   # observable 0 is the XX parity
        n += m
    return accepted / shots, f_zz / accepted, f_xx / accepted

# ---------------------------------------------------------------------------
# Error detection with the acceptance rule of the paper (appendix D.1):
# a run is accepted only if Alice's and Bob's outcomes agree in all syndrome
# comparisons performed by the protocol. Detectors that use the final destructive
# measurement of the data qubits exist only to evaluate the logical observable
# and are not used for acceptance. Accepted runs are then decoded with the
# decoder of the error-correction experiments, using all detectors.
# ---------------------------------------------------------------------------
_MEAS_GATES = {'M', 'MR', 'MX', 'MY', 'MZ', 'MRX', 'MRY', 'MRZ', 'MXX', 'MYY', 'MZZ'}


def protocol_detector_mask(circuit):
    """Boolean mask over the detectors of `circuit`: True for detectors that compare
    syndrome outcomes inside the protocol, False for detectors that use the final
    measurement instruction of the circuit (the destructive data readout)."""
    rec_instr = []
    det_records = []
    for k, inst in enumerate(circuit.flattened()):
        if inst.name == 'MPP':
            raise NotImplementedError('MPP measurements are not supported')
        if inst.name in _MEAS_GATES:
            rec_instr.extend([k] * len(inst.targets_copy()))
        elif inst.name == 'DETECTOR':
            det_records.append([len(rec_instr) + t.value for t in inst.targets_copy()])
    final = {i for i, k in enumerate(rec_instr) if k == rec_instr[-1]}
    return np.array([not any(r in final for r in recs) for recs in det_records], dtype=bool)


def _detection_with_decoding(circuit, decode, shots, min_failures, max_shots, batch, seed):
    """Sample in batches until at least `shots` runs and `min_failures` logical failures
    among accepted runs (or `max_shots` runs). Returns (acceptance, conditional LER)."""
    prot = protocol_detector_mask(circuit)
    max_shots = max_shots if max_shots is not None else max(shots, 200_000_000)
    n = accepted = failures = 0
    b = 0
    while True:
        m = min(batch, max_shots - n)
        sampler = circuit.compile_detector_sampler(seed=None if seed is None else seed + b)
        det, obs = sampler.sample(m, separate_observables=True)
        keep = ~det[:, prot].any(axis=1)
        wrong = decode(det[keep]) ^ obs[keep].astype(np.uint8)
        n += m
        accepted += int(keep.sum())
        failures += int(wrong.any(axis=1).sum())
        b += 1
        if (n >= shots and failures >= min_failures) or n >= max_shots:
            break
    return accepted / n, failures / accepted if accepted > 0 else 1


def _surface_decoder(circuit):
    model = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(model)
    n_obs = circuit.num_observables

    def decode(det):
        if len(det) == 0:
            return np.zeros((0, n_obs), dtype=np.uint8)
        return np.asarray(matching.decode_batch(det), dtype=np.uint8).reshape(len(det), n_obs)
    return decode


def _ml_lookup_decoder(circuit):
    """Exact maximum-likelihood decoder for small circuits (one logical observable).
    The detector error model of the circuit lists every circuit-level fault mechanism (gate, reset,
    measurement, idle and transmission faults in every round) with its probability and the detectors
    and observable it flips. Because the mechanisms are independent, the joint distribution of
    (syndrome, observable flip) is obtained exactly by combining them one at a time; for each syndrome
    the decoder returns the more likely observable value."""
    dem = circuit.detector_error_model(decompose_errors=False).flattened()
    nd = circuit.num_detectors
    assert circuit.num_observables == 1 and nd <= 22, "lookup decoder is meant for small circuits"
    dist = np.zeros(1 << (nd + 1))
    dist[0] = 1.0
    idx = np.arange(1 << (nd + 1))
    for inst in dem:
        if inst.type != "error":
            continue
        prob = inst.args_copy()[0]
        mask = 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                mask ^= 1 << t.val
            elif t.is_logical_observable_id():
                mask ^= 1 << nd
        dist = (1 - prob) * dist + prob * dist[idx ^ mask]
    table = (dist[1 << nd:] > dist[:1 << nd]).astype(np.uint8)   # P(s, flip) > P(s, no flip)
    weights = 1 << np.arange(nd)

    def decode(det):
        det = np.asarray(det, dtype=np.int64)
        if len(det) == 0:
            return np.zeros((0, 1), dtype=np.uint8)
        return table[det @ weights].reshape(-1, 1)
    return decode


def _ml_correction(circuit, shots, batch=2_000_000, seed=None):
    """Logical error rate of error correction with the exact maximum-likelihood lookup decoder."""
    decode = _ml_lookup_decoder(circuit)
    sampler = circuit.compile_detector_sampler(seed=seed)
    failures = done = 0
    while done < shots:
        m = min(batch, shots - done)
        det, obs = sampler.sample(m, separate_observables=True)
        failures += int((decode(det)[:, 0] ^ obs[:, 0]).sum())
        done += m
    return failures / shots


def _correction_in_batches(circuit, shots, batch):
    """Logical error rate of MWPM decoding, sampled in batches to bound memory.
    Every circuit declares one logical observable."""
    model = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(model)
    sampler = circuit.compile_detector_sampler()
    failures = 0
    done = 0
    while done < shots:
        m = min(batch, shots - done)
        syndrome_matrix, actual_observables = sampler.sample(shots=m, separate_observables=True)
        predicted_observables = matching.decode_batch(syndrome_matrix)
        failures += int((predicted_observables ^ actual_observables).any(axis=1).sum())
        done += m
    return failures / shots


def cal_surface_correction(d = 5, num_layer = 5, p_local = 0.001, p_trans = 0.01, shots = 3000000, protocol_type="encoding", batch = 2_000_000, basis = "Z"):
    # basis: "Z" (|0>_L or Z_L Z_L, default) or "X" (|+>_L or X_L X_L); noiseless final data readout in that basis
    if protocol_type == "encoding":
        # QECC Protocal ZZ measurement -> Send to Bob 
        circuit = EPR_QECC_Protocal_Circuit(d = d, num_layer = num_layer, p_local = p_local, p_trans = p_trans, basis = basis)
        
    elif protocol_type == "distillation":
        # LOCC Protocal EPR pairs -> Send to Bob 
        circuit = EPR_LOCC_Protocal_Circuit(d = d, num_layer = num_layer, p_local = p_local, p_trans = p_trans, basis = basis)
    
    elif protocol_type == "QECC":
        circuit = Direct_QECC_Circuit(d = d, num_layer = num_layer, p_local = p_local, p_trans = p_trans, basis = basis)
    
    else:
        raise ValueError("Invalid protocol type.")

    return _correction_in_batches(circuit, shots, batch)
    
def cal_surface_detection(d = 5, num_layer = 5, p_local = 0.001, p_trans = 0.01, shots = 10000, protocol_type="encoding",
                          min_failures = 0, max_shots = None, batch = 1_000_000, seed = None, basis = "Z"):
    # basis: "Z" (|0>_L or Z_L Z_L, default) or "X" (|+>_L or X_L X_L); noiseless final data readout in that basis
    if protocol_type == "encoding":
        # QECC Protocal ZZ measurement -> Send to Bob 
        circuit = EPR_QECC_Protocal_Circuit(d = d, num_layer = num_layer, p_local = p_local, p_trans = p_trans, basis = basis)
        
    elif protocol_type == "distillation":
        # LOCC Protocal EPR pairs -> Send to Bob 
        circuit = EPR_LOCC_Protocal_Circuit(d = d, num_layer = num_layer, p_local = p_local, p_trans = p_trans, basis = basis)
    
    elif protocol_type == "QECC":
        circuit = Direct_QECC_Circuit(d, num_layer, p_local = p_local, p_trans = p_trans, basis = basis)
    
    else:
        raise ValueError("Invalid protocol type.")

    return _detection_with_decoding(circuit, _surface_decoder(circuit), shots, min_failures, max_shots, batch, seed)

def cal_2g_surface_correction(d = 5, p_local = 0.001, p_trans = 0.1, n = 10000, batch = 2_000_000, basis = "Z"):
    # basis: "Z" (|0>_L or Z_L Z_L, default) or "X" (|+>_L or X_L X_L); noiseless final data readout in that basis
    circuit_qecc = EE_Surface_Transversal_CNOT_Teleportation(d = d, p_local = p_local, p_trans = p_trans, basis = basis)
    return _correction_in_batches(circuit_qecc, n, batch)

def cal_2g_surface_detection(d = 5, p_local = 0.001, p_trans = 0.1, n = 10000):
    circuit_qecc = EE_Surface_Transversal_CNOT_Teleportation(d = d, p_local = p_local, p_trans = p_trans)
    sampler = circuit_qecc.compile_detector_sampler()
    model = circuit_qecc.detector_error_model(decompose_errors=True)
    syndrome_matrix, actual_observables = sampler.sample(shots=n, separate_observables=True)
    syndrome_weights = syndrome_matrix.sum(axis=1)
    threshold = 0
    valid_indices = syndrome_weights <= threshold 
    filtered_syndrome_matrix = syndrome_matrix[valid_indices]
    filtered_actual_observables = actual_observables[valid_indices]
    logical_error_count = filtered_actual_observables.sum()
    filtered_n = len(filtered_actual_observables)
    return  filtered_n / n, logical_error_count / filtered_n if filtered_n > 0 else 1
    
def cal_steane_correction(p_local = 0.001, p_trans = 0.1, shots=10000, protocol_type=1, basis='Z', seed=None):
    # basis: 'Z' (Z_L / Z_L Z_L readout, the original circuits) or 'X' (X_L / X_L X_L readout).
    # seed: stim sampler seed (None = unseeded, the original behaviour).
        # circuit = stim.Circuit.from_file("test.stim")
    protocol_circuits = {
        1: ED_circuit,
        2: EE_circuit,
        3: QECC_circuit_Plus
    }
    Code = codes.SteaneCode()
    
    circuit = protocol_circuits[protocol_type](Code=Code, p_local=p_local, p_trans=p_trans, basis=basis)
    return _ml_correction(circuit, shots, seed=seed)

def cal_2g_steane_correction(p_local = 0.001, p_trans = 0.1, shots=10000, basis='Z', seed=None):
    Code = codes.SteaneCode()
    circuit = EE_circuit_Transversal_CNOT_Teleportation_Aligned(Code=Code, p_local=p_local, p_trans=p_trans, basis=basis)
    return _ml_correction(circuit, shots, seed=seed)

def cal_steane_detection(p_local = 0.001, p_trans = 0.1, shots=10000, protocol_type=1, zx_detector = True,
                         min_failures = 0, max_shots = None, batch = 1_000_000, seed = None, basis = 'Z'):
    protocol_circuits = {
        1: ED_circuit,
        2: EE_circuit,
        4: QECC_circuit_Plus
    }
    Code = codes.SteaneCode()
    # zx_detector=True: compare both X- and Z-type stabilizer outcomes
    circuit = protocol_circuits[protocol_type](Code=Code, p_local=p_local, p_trans=p_trans, zx_detector=zx_detector, basis=basis)
    return _detection_with_decoding(circuit, _ml_lookup_decoder(circuit), shots, min_failures, max_shots, batch, seed)
