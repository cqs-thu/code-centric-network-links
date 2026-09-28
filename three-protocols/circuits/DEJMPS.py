import stim
import numpy as np
from tqdm import tqdm
from itertools import product

def prepare_epr_pairs(circuit, total_qubits, p_local, p_trans=0.001):
    """Prepare 2^n EPR pairs using 2 * 2^n qubits."""
    # Qubit 2*i is in |+>, qubit 2*i + 1 is in |0>
    # Qubit 2*i belongs to Alice, Qubit 2*i + 1 belongs to Bob
    qubits_list = [i for i in range(total_qubits)]
    qubits_even = [i for i in range(0, total_qubits, 2)]
    qubits_odd  = [i for i in range(1, total_qubits, 2)]
    circuit.append("R", qubits_list)
    circuit.append("X_ERROR", qubits_list, p_local)
    circuit.append("H", qubits_even)
    circuit.append("CX", qubits_list)
    circuit.append("DEPOLARIZE1", qubits_odd, p_trans)
        

def distill_group(circuit, groups, p_local, shift = 0):
    """Perform distillation on a group of 4 qubits, returning measurement operations."""
    CNOT_L = []
    CNOT_R = []
    checks = []

    for group in groups:
        q1, q2, q3, q4 = group ## Alice: q1, q3; Bob: q2, q4
        circuit.append("S_DAG", [q1, q3])
        circuit.append("H", [q1, q3])
        circuit.append("S_DAG", [q1, q3])

        circuit.append("S", [q2, q4])
        circuit.append("H", [q2, q4])
        circuit.append("S", [q2, q4])
        circuit.append("DEPOLARIZE1", [q1, q2, q3, q4], p_local)

        CNOT_L += [q1, q3]
        CNOT_R += [q2, q4]
        checks += [q3, q4]

    CNOT_All = CNOT_L + CNOT_R
    circuit.append("CX", CNOT_All)
    circuit.append("DEPOLARIZE2", CNOT_All, p_local)

    circuit.append("X_ERROR", checks, p_local)
    circuit.append("M", checks)
    return checks


def epr_distillation(n, p_local=0.001, p_trans=0.001, fidelity_basis = "ZX", noisy_readout=True):
    """
    Implement EPR distillation protocol for 2^n pairs.
    Args:
        n: Exponent for number of EPR pairs (2^n).
        p_local: Probability of depolarization noise.
        noisy_readout: if False, the ancillas that read out the final ZZ and XX parities are noiseless
            (as the final data readout of the surface code and Steane code circuits).
    Returns:
        stim.Circuit: The complete distillation circuit.
    """
    circuit = stim.Circuit()
    num_pairs = 2 ** n
    total_qubits = 2 * num_pairs

    # Step 1: Prepare 2^n EPR pairs
    prepare_epr_pairs(circuit=circuit, total_qubits=total_qubits, p_local=p_local, p_trans=p_trans)

    # Step 2: Iterative distillation
    current_pairs = num_pairs
    qubit_pairs = [(i * 2, i * 2 + 1) for i in range(num_pairs)]  # List of (q1, q2) pairs
    detectors = []

    shift = 0
    while current_pairs > 1:
        next_pairs = []
        groups = []
        for i in range(0, current_pairs, 2):
            if i + 1 >= current_pairs:
                # Odd number of pairs, carry over the last pair
                next_pairs.append(qubit_pairs[i])
                break
            # Group two EPR pairs: (q1, q2) and (q3, q4)
            q1, q2 = qubit_pairs[i]
            q3, q4 = qubit_pairs[i + 1]
            group = [q1, q2, q3, q4]
            groups.append(group)
            # only keep (q1, q2) as the next EPR pair
            next_pairs.append((q1, q2))
        qubit_pairs = next_pairs
        current_pairs = len(qubit_pairs)
            
        # Perform distillation on this group
        checks = distill_group(circuit=circuit, groups=groups, p_local=p_local, shift=shift)
        shift += len(groups)
        detector_shift = 0
        for i in checks[::-1]:
            detector_shift -= 1
            if abs(detector_shift) % 2 == 0:
                continue
            circuit.append("DETECTOR", [stim.target_rec(detector_shift), stim.target_rec(detector_shift - 1)], [i, 0, 0])
        
        # Detector to check if measurements are equal (q3 == q4)
        circuit.append("SHIFT_COORDS", [], [0, 0, 1])
    
    # Step 3: Final parity measurement
    final_q1, final_q2 = qubit_pairs[0]
    
    if fidelity_basis == "Z":
        circuit.append("M", [final_q1, final_q2])
        # Logical observable: parity +1 if measurement is 0
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1), stim.target_rec(-2)], [0])
    elif fidelity_basis == "ZX":
        # Add ZZ and XX measurement 
        zz_ancilla = total_qubits + 1
        xx_ancilla = total_qubits + 2

        
        circuit.append("R", [zz_ancilla, xx_ancilla])
        if noisy_readout:
            circuit.append("X_ERROR", [zz_ancilla, xx_ancilla], p_local)
        circuit.append("H", xx_ancilla)
        circuit.append("CX", [final_q1, zz_ancilla])
        circuit.append("TICK")
        circuit.append("CX", [final_q2, zz_ancilla])
        circuit.append("TICK")
        circuit.append("CX", [xx_ancilla, final_q1])
        circuit.append("TICK")
        circuit.append("CX", [xx_ancilla, final_q2])
        circuit.append("H", xx_ancilla)
        circuit.append("M", [zz_ancilla, xx_ancilla])
        
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], [0])
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-2)], [1])
    else:
        raise ValueError(f"Invalid basis: {fidelity_basis}. Must be Z or ZX.")
    
    # circuit.to_file("Repetition_Code_ED_{p_local}_{p_trans}.stim")
    return circuit


def simulate_detection_circuit(circuit, shots=10000):
    """Simulate the circuit and determine success rate using vectorized numpy operations."""
    detector_sampler = circuit.compile_detector_sampler()
    detector_results, observable_results = detector_sampler.sample(shots, separate_observables=True)

    # detector_results: (shots, num_detectors)
    # observable_results: (shots, num_observables)
    
    detector_success_mask = np.all(detector_results == 0, axis=1)

    check_successes = np.sum(detector_success_mask)

    # observable_success_mask = observable_results[:, 0] == 0
    observable_success_mask = np.all(observable_results == 0, axis=1)

    successes = np.sum(detector_success_mask & observable_success_mask)

    return check_successes, successes

def simulate_correction_circuit(circuit, shots=10000):
    detector_sampler = circuit.compile_detector_sampler()
    detector_results, observable_results = detector_sampler.sample(shots, separate_observables=True)

    num_of_0 = np.sum(detector_results == 0, axis=1)
    num_of_1 = np.sum(detector_results == 1, axis=1)

    tmp_observable_result = (num_of_0 <= num_of_1).astype(np.uint8)
    observable_target = observable_results[:, 0]
    successes = np.sum(tmp_observable_result == observable_target)

    return successes

