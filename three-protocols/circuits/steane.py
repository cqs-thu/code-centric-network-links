import numpy as np
import qldpc
from qldpc import codes
from qldpc.codes import CSSCode
import networkx as nx
import galois
import stim
from itertools import combinations
from tqdm import tqdm


def get_info(Code, shift = 0):
    ancilla_qubit_list = list(range(shift, shift + Code.num_checks_x + Code.num_checks_z))
    ancilla_x_qubit_list = ancilla_qubit_list[:Code.num_checks_x]
    ancilla_z_qubit_list = ancilla_qubit_list[Code.num_checks_x:]
    data_qubit_list = list(range(shift + Code.num_checks, shift + Code.num_checks + Code.num_qubits))
    qubits_all_list = ancilla_qubit_list + data_qubit_list
    logical_x_operators = Code.get_logical_ops()[0]
    logical_z_operators = Code.get_logical_ops()[1]
    info_dict = {
        "ancilla_qubit_list": ancilla_qubit_list,
        "ancilla_x_qubit_list": ancilla_x_qubit_list,
        "ancilla_z_qubit_list": ancilla_z_qubit_list,
        "data_qubit_list": data_qubit_list,
        "qubits_all_list": qubits_all_list,
        "logical_x_operator": logical_x_operators,
        "logical_z_operator": logical_z_operators
    }
    return info_dict

'''circuit of protocol I: Entanglement Distillation'''
def ED_circuit(Code, p_local, p_trans, zx_detector = False, basis = 'Z'):
    # basis='Z': |0>_L-type readout, observable Z_L(A) Z_L(B) (original circuit).
    # basis='X': same preparation, gates and noise; noiseless MX readout, observable X_L(A) X_L(B).
    assert basis in ('Z', 'X')
    Alice_info = get_info(Code=Code, shift=0)
    Bob_Shift = Code.num_checks + Code.num_qubits
    Bob_info = get_info(Code=Code, shift=Bob_Shift)

    # initial circuits
    Alice_data_qubits = Alice_info["data_qubit_list"]
    Bob_data_qubits   = Bob_info["data_qubit_list"]
    Alice_ancilla_x_qubits = Alice_info["ancilla_x_qubit_list"]
    Alice_ancilla_z_qubits = Alice_info["ancilla_z_qubit_list"]
    Bob_ancilla_x_qubits = Bob_info["ancilla_x_qubit_list"]
    Bob_ancilla_z_qubits = Bob_info["ancilla_z_qubit_list"]
    All_data_qubits = Alice_data_qubits + Bob_data_qubits
    All_qubits = Alice_data_qubits + Alice_ancilla_x_qubits + Alice_ancilla_z_qubits + Bob_data_qubits + Bob_ancilla_x_qubits + Bob_ancilla_z_qubits
    Ancilla_qubits = Alice_ancilla_x_qubits + Alice_ancilla_z_qubits + Bob_ancilla_x_qubits + Bob_ancilla_z_qubits
    Ancilla_x_qubits = Alice_ancilla_x_qubits + Bob_ancilla_x_qubits
    Ancilla_z_qubits = Alice_ancilla_z_qubits + Bob_ancilla_z_qubits

    circuit = stim.Circuit()
    circuit.append("R", All_qubits)
    # add noise on data qubits only
    circuit.append('X_ERROR', All_qubits, p_local)
    circuit.append("H", Alice_data_qubits)
    for i in range(len(Alice_data_qubits)):
        circuit.append("CX", [Alice_data_qubits[i], Bob_data_qubits[i]])
    for i in range(len(Alice_data_qubits)):
        circuit.append("DEPOLARIZE2", [Alice_data_qubits[i], Bob_data_qubits[i]], p_local)

    # Bob qubits transimission
    circuit.append("DEPOLARIZE1", Bob_data_qubits, p_trans)

    # stabilizer measurement
    # stabilizer measurement - Irregular stabilizer measurement count
    matrix_x = Code.matrix_x
    matrix_z = Code.matrix_z

    circuit.append("H", Ancilla_x_qubits)
    circuit.append("TICK")

    # initialize gates scheduling table: X stabilizer
    schedule = []

    # x stabilizer measurement
    for idx, ancilla_qubit in enumerate(Alice_ancilla_x_qubits):
        stabilizer = matrix_x[idx]
        data_qubits = [Alice_data_qubits[j] for j in np.where(stabilizer == 1)[0]]
        cnot_ops = [(ancilla_qubit, data_qubit) for data_qubit in data_qubits]
        schedule.extend(cnot_ops)

    time_steps = []

    while any(schedule):
        time_step_ops = []
        used_qubits = set()
        remaining_ops = []
        for op in schedule:
            if op[0] not in used_qubits and op[1] not in used_qubits:
                time_step_ops.append(op)
                used_qubits.update(op)
            else:
                remaining_ops.append(op)
        time_steps.append(time_step_ops)
        schedule = remaining_ops

    for time_step_ops in time_steps:
        CX_list = []
        single_qubits = set(All_qubits)
        
        for op in time_step_ops:
            # Alice
            CX_list.extend(op)
            single_qubits.difference_update(op)
            # Bob
            new_op = (op[0] + Bob_Shift, op[1] + Bob_Shift)
            CX_list.extend(new_op)
            single_qubits.difference_update(new_op)
        
        if CX_list:
            circuit.append('CX', CX_list)
            circuit.append('DEPOLARIZE2', CX_list, p_local)
            
        if single_qubits:
            circuit.append('DEPOLARIZE1', sorted(single_qubits), p_local)

        circuit.append("TICK")

    circuit.append("H", Ancilla_x_qubits)
    circuit.append("TICK")

    # z stabilizer
    for idx, ancilla_qubit in enumerate(Alice_ancilla_z_qubits):
        stabilizer = matrix_z[idx]
        data_qubits = [Alice_data_qubits[j] for j in np.where(stabilizer == 1)[0]]
        cnot_ops = [(data_qubit, ancilla_qubit) for data_qubit in data_qubits]
        schedule.extend(cnot_ops)

    time_steps = []

    while any(schedule):
        time_step_ops = []
        used_qubits = set()
        remaining_ops = []
        for op in schedule:
            if op[0] not in used_qubits and op[1] not in used_qubits:
                time_step_ops.append(op)
                used_qubits.update(op)
            else:
                remaining_ops.append(op)
        time_steps.append(time_step_ops)
        schedule = remaining_ops

    for time_step_ops in time_steps:
        CX_list = []
        single_qubits = set(All_qubits)
        
        for op in time_step_ops:
            # Alice
            CX_list.extend(op)
            single_qubits.difference_update(op)
            # Bob
            new_op = (op[0] + Bob_Shift, op[1] + Bob_Shift)
            CX_list.extend(new_op)
            single_qubits.difference_update(new_op)
        
        if CX_list:
            circuit.append('CX', CX_list)
            circuit.append('DEPOLARIZE2', CX_list, p_local)
            
        if single_qubits:
            circuit.append('DEPOLARIZE1', sorted(single_qubits), p_local)

        circuit.append("TICK")

    circuit.append('X_ERROR', Ancilla_qubits, p_local)
    circuit.append('MR', Ancilla_qubits)
    circuit.append('X_ERROR', Ancilla_qubits, p_local)

    # add detectors of Alice and Bob
    detector_shift = 0
    for i in Ancilla_qubits[::-1]:
        detector_shift -= 1
        if zx_detector:
            non_detector_ancilla = Alice_ancilla_x_qubits + Alice_ancilla_z_qubits
        else:
            non_detector_ancilla = Alice_ancilla_x_qubits + Alice_ancilla_z_qubits + (Bob_ancilla_x_qubits if basis == 'Z' else Bob_ancilla_z_qubits)
            
        if i in non_detector_ancilla:
            continue
        
        circuit.append("DETECTOR", [stim.target_rec(detector_shift), stim.target_rec(detector_shift - len(Ancilla_qubits) // 2)], [i - Bob_Shift, 0])

    circuit.append("SHIFT_COORDS", [], [0, 1])

    # measure all data qubits
    circuit.append('M' if basis == 'Z' else 'MX', All_data_qubits)

    logical_x_operators = Code.get_logical_ops()[0]
    logical_z_operators = Code.get_logical_ops()[1]
        
    N = Code.num_qubits
    K = (len(logical_x_operators) + len(logical_z_operators)) // 2

    detector_list = []
    detector_list += Ancilla_qubits
    detector_list += All_data_qubits
    detector_shift_list = [(-len(detector_list) + i) for i in range(len(detector_list))]
    
    # perfect z measurement
    for i, detector in enumerate(detector_list):
        if detector in (Alice_ancilla_z_qubits if basis == 'Z' else Alice_ancilla_x_qubits):
            target_rec = []
            target_rec.append(stim.target_rec(detector_shift_list[i]))
            # one stabilizer check - many data qubits
            idx = (Alice_ancilla_z_qubits if basis == 'Z' else Alice_ancilla_x_qubits).index(detector)
            stabilizer = (matrix_z if basis == 'Z' else matrix_x)[idx]
            alice_data_qubits = [Alice_data_qubits[j] for j in np.where(stabilizer == 1)[0]]
            for check_qubit in alice_data_qubits:
                target_rec.append(stim.target_rec(detector_shift_list[detector_list.index(check_qubit)]))
            circuit.append("DETECTOR", target_rec, [detector, 0])
            
    for i, detector in enumerate(detector_list):
        if detector in (Bob_ancilla_z_qubits if basis == 'Z' else Bob_ancilla_x_qubits):
            target_rec = []
            target_rec.append(stim.target_rec(detector_shift_list[i]))
            # one stabilizer check - many data qubits
            idx = (Bob_ancilla_z_qubits if basis == 'Z' else Bob_ancilla_x_qubits).index(detector)
            stabilizer = (matrix_z if basis == 'Z' else matrix_x)[idx]
            bob_data_qubits = [Bob_data_qubits[j] for j in np.where(stabilizer == 1)[0]]
            for check_qubit in bob_data_qubits:
                target_rec.append(stim.target_rec(detector_shift_list[detector_list.index(check_qubit)]))
            circuit.append("DETECTOR", target_rec, [detector, 0])
        
    for k in range(K):
        target_rec = []
        logical_z_operator = logical_z_operators[k]
        support = np.where(logical_z_operator == 1)[0] - N if basis == 'Z' else np.where(np.asarray(logical_x_operators[k])[:N] == 1)[0]
        data_qubits = [Alice_data_qubits[j] for j in support] + [Bob_data_qubits[j] for j in support]
        for i in range(len(detector_list)):
            if detector_list[i] in data_qubits:
                target_rec.append(stim.target_rec(detector_shift_list[i]))
        circuit.append(stim.CircuitInstruction('OBSERVABLE_INCLUDE', target_rec, [k]))
    # circuit.to_file("Steane_Code_ED_{p_local}_{p_trans}.stim")
    return circuit


'''circuit of protocol II: Entanglement Encoding'''
def EE_circuit(Code, p_local, p_trans, zx_detector = False, basis = 'Z'):
    # basis='Z': observable Z_L(A) Z_L(B) (original circuit).
    # basis='X': same preparation, gates and noise; noiseless MX readout, observable X_L(A) X_L(B).
    assert basis in ('Z', 'X')
    Alice_info = get_info(Code=Code, shift=0)
    Bob_Shift = Code.num_checks + Code.num_qubits
    Bob_info = get_info(Code=Code, shift=Bob_Shift)
    # print(Alice_info)
    # print(Bob_info)

    # initial circuits
    Alice_data_qubits = Alice_info["data_qubit_list"]
    Bob_data_qubits   = Bob_info["data_qubit_list"]
    Alice_ancilla_x_qubits = Alice_info["ancilla_x_qubit_list"]
    Alice_ancilla_z_qubits = Alice_info["ancilla_z_qubit_list"]
    Bob_ancilla_x_qubits = Bob_info["ancilla_x_qubit_list"]
    Bob_ancilla_z_qubits = Bob_info["ancilla_z_qubit_list"]
    All_data_qubits = Alice_data_qubits + Bob_data_qubits
    All_qubits = Alice_data_qubits + Alice_ancilla_x_qubits + Alice_ancilla_z_qubits + Bob_data_qubits + Bob_ancilla_x_qubits + Bob_ancilla_z_qubits
    Ancilla_qubits = Alice_ancilla_x_qubits + Alice_ancilla_z_qubits + Bob_ancilla_x_qubits + Bob_ancilla_z_qubits
    Ancilla_x_qubits = Alice_ancilla_x_qubits + Bob_ancilla_x_qubits
    Ancilla_z_qubits = Alice_ancilla_z_qubits + Bob_ancilla_z_qubits

    circuit = stim.Circuit()
    circuit.append("R", All_qubits)
    # add noise on data qubits only
    circuit.append('X_ERROR', All_qubits, p_local)
    circuit.append("H", Alice_data_qubits)

    # encoding stabilizer measurement
    matrix_x = Code.matrix_x
    matrix_z = Code.matrix_z

    def syndrome_extraction(circuit):
        circuit.append("H", Ancilla_x_qubits)
        circuit.append("TICK")

        # initialize gates scheduling table: X stabilizer
        schedule = []

        # x stabilizer measurement
        for idx, ancilla_qubit in enumerate(Alice_ancilla_x_qubits):
            stabilizer = matrix_x[idx]
            data_qubits = [Alice_data_qubits[j] for j in np.where(stabilizer == 1)[0]]
            cnot_ops = [(ancilla_qubit, data_qubit) for data_qubit in data_qubits]
            schedule.extend(cnot_ops)

        time_steps = []

        while any(schedule):
            time_step_ops = []
            used_qubits = set()
            remaining_ops = []
            for op in schedule:
                if op[0] not in used_qubits and op[1] not in used_qubits:
                    time_step_ops.append(op)
                    used_qubits.update(op)
                else:
                    remaining_ops.append(op)
            time_steps.append(time_step_ops)
            schedule = remaining_ops

        for time_step_ops in time_steps:
            CX_list = []
            single_qubits = set(All_qubits)

            for op in time_step_ops:
                # Alice
                CX_list.extend(op)
                single_qubits.difference_update(op)
                # Bob
                new_op = (op[0] + Bob_Shift, op[1] + Bob_Shift)
                CX_list.extend(new_op)
                single_qubits.difference_update(new_op)

            if CX_list:
                circuit.append('CX', CX_list)
                circuit.append('DEPOLARIZE2', CX_list, p_local)

            if single_qubits:
                circuit.append('DEPOLARIZE1', sorted(single_qubits), p_local)

            circuit.append("TICK")

        circuit.append("H", Ancilla_x_qubits)
        circuit.append("TICK")

        # z stabilizer
        for idx, ancilla_qubit in enumerate(Alice_ancilla_z_qubits):
            stabilizer = matrix_z[idx]
            data_qubits = [Alice_data_qubits[j] for j in np.where(stabilizer == 1)[0]]
            cnot_ops = [(data_qubit, ancilla_qubit) for data_qubit in data_qubits]
            schedule.extend(cnot_ops)

        time_steps = []

        while any(schedule):
            time_step_ops = []
            used_qubits = set()
            remaining_ops = []
            for op in schedule:
                if op[0] not in used_qubits and op[1] not in used_qubits:
                    time_step_ops.append(op)
                    used_qubits.update(op)
                else:
                    remaining_ops.append(op)
            time_steps.append(time_step_ops)
            schedule = remaining_ops

        for time_step_ops in time_steps:
            CX_list = []
            single_qubits = set(All_qubits)

            for op in time_step_ops:
                # Alice
                CX_list.extend(op)
                single_qubits.difference_update(op)
                # Bob
                new_op = (op[0] + Bob_Shift, op[1] + Bob_Shift)
                CX_list.extend(new_op)
                single_qubits.difference_update(new_op)

            if CX_list:
                circuit.append('CX', CX_list)
                circuit.append('DEPOLARIZE2', CX_list, p_local)

            if single_qubits:
                circuit.append('DEPOLARIZE1', sorted(single_qubits), p_local)

            circuit.append("TICK")
            
        circuit.append('X_ERROR', Ancilla_qubits, p_local)
        circuit.append('MR', Ancilla_qubits)
        circuit.append('X_ERROR', Ancilla_qubits, p_local)
        
        return circuit

    # stabilizer encoding circuit
    circuit = syndrome_extraction(circuit)
    # add x/z detectors of Alice and Bob
    detector_shift = 0
    for i in Ancilla_qubits[::-1]:
        detector_shift -= 1
        if i in (Alice_ancilla_z_qubits + Bob_ancilla_x_qubits):
            continue
        circuit.append("DETECTOR", [stim.target_rec(detector_shift)], [i, 0])

    circuit.append("SHIFT_COORDS", [], [0, 1])

    # Logical EPR Encoding: Transversal CNOTs
    for i in range(len(Alice_data_qubits)):
        circuit.append("CX", [Alice_data_qubits[i], Bob_data_qubits[i]])
    for i in range(len(Alice_data_qubits)):
        circuit.append("DEPOLARIZE2", [Alice_data_qubits[i], Bob_data_qubits[i]], p_local)

    # Bob qubits transimission
    circuit.append("DEPOLARIZE1", Bob_data_qubits, p_trans)

    # decoding stabilizer measurement
    circuit = syndrome_extraction(circuit)
    # add x/z detectors of Alice and Bob
    detector_shift = 0
    for i in Ancilla_qubits[::-1]:
        detector_shift -= 1
        if zx_detector:
            non_detector_ancilla = Alice_ancilla_x_qubits + Alice_ancilla_z_qubits
        else:
            non_detector_ancilla = Alice_ancilla_x_qubits + Alice_ancilla_z_qubits + (Bob_ancilla_x_qubits if basis == 'Z' else Bob_ancilla_z_qubits)
        if i in non_detector_ancilla:
            continue
        if basis == 'X' and i in Bob_ancilla_x_qubits:
            # after the transversal CNOT A->B: X_A(r2) = X_A(r1) X_B(r1), X_B(r2) = X_B(r1),
            # so X_B(r2) + X_A(r2) + X_A(r1) is deterministic (the Z-type mirror of the detector below)
            circuit.append("DETECTOR", [stim.target_rec(detector_shift), stim.target_rec(detector_shift - len(Ancilla_qubits) // 2), stim.target_rec(detector_shift - len(Ancilla_qubits) // 2 - len(Ancilla_qubits))], [i - Bob_Shift, 0])
            continue
        circuit.append("DETECTOR", [stim.target_rec(detector_shift), stim.target_rec(detector_shift - len(Ancilla_qubits) // 2), stim.target_rec(detector_shift - len(Ancilla_qubits))], [i - Bob_Shift, 0])

    circuit.append("SHIFT_COORDS", [], [0, 1])


    # measure all data qubits
    circuit.append('M' if basis == 'Z' else 'MX', All_data_qubits)

    logical_x_operators = Code.get_logical_ops()[0]
    logical_z_operators = Code.get_logical_ops()[1]
        
    N = Code.num_qubits
    K = (len(logical_x_operators) + len(logical_z_operators)) // 2

    detector_list = []
    detector_list += Ancilla_qubits
    detector_list += All_data_qubits
    detector_shift_list = [(-len(detector_list) + i) for i in range(len(detector_list))]
    
    # perfect z measurement
    for i, detector in enumerate(detector_list):
        if detector in (Alice_ancilla_z_qubits if basis == 'Z' else Alice_ancilla_x_qubits):
            target_rec = []
            target_rec.append(stim.target_rec(detector_shift_list[i]))
            # one stabilizer check - many data qubits
            idx = (Alice_ancilla_z_qubits if basis == 'Z' else Alice_ancilla_x_qubits).index(detector)
            stabilizer = (matrix_z if basis == 'Z' else matrix_x)[idx]
            alice_data_qubits = [Alice_data_qubits[j] for j in np.where(stabilizer == 1)[0]]
            for check_qubit in alice_data_qubits:
                target_rec.append(stim.target_rec(detector_shift_list[detector_list.index(check_qubit)]))
            circuit.append("DETECTOR", target_rec, [detector, 0])
            
    for i, detector in enumerate(detector_list):
        if detector in (Bob_ancilla_z_qubits if basis == 'Z' else Bob_ancilla_x_qubits):
            target_rec = []
            target_rec.append(stim.target_rec(detector_shift_list[i]))
            # one stabilizer check - many data qubits
            idx = (Bob_ancilla_z_qubits if basis == 'Z' else Bob_ancilla_x_qubits).index(detector)
            stabilizer = (matrix_z if basis == 'Z' else matrix_x)[idx]
            bob_data_qubits = [Bob_data_qubits[j] for j in np.where(stabilizer == 1)[0]]
            for check_qubit in bob_data_qubits:
                target_rec.append(stim.target_rec(detector_shift_list[detector_list.index(check_qubit)]))
            circuit.append("DETECTOR", target_rec, [detector, 0])
    
    for k in range(K):
        target_rec = []
        logical_z_operator = logical_z_operators[k]
        support = np.where(logical_z_operator == 1)[0] - N if basis == 'Z' else np.where(np.asarray(logical_x_operators[k])[:N] == 1)[0]
        data_qubits = [Alice_data_qubits[j] for j in support] + [Bob_data_qubits[j] for j in support]
        for i in range(len(detector_list)):
            if detector_list[i] in data_qubits:
                target_rec.append(stim.target_rec(detector_shift_list[i]))
        circuit.append(stim.CircuitInstruction('OBSERVABLE_INCLUDE', target_rec, [k]))
    
    # circuit.to_file(f"Steane_Code_LEE_{p_local}_{p_trans}.stim")
    return circuit


'''circuit of protocol III: QECC Transimission'''
def QECC_circuit_Plus(Code, p_local, p_trans, zx_detector = False, basis = 'Z'):
    # basis='Z': encodes |0>_L and reads Z_L (original circuit; the name notwithstanding, the state is |0>_L).
    # basis='X': X<->Z mirror: encodes |+>_L (RX + Z_ERROR on data, CNOT directions of the encoder reversed),
    #            same syndrome extraction and noise, noiseless MX readout, observable X_L.
    assert basis in ('Z', 'X')
    Alice_info = get_info(Code=Code, shift=0)

    # initial circuits
    Alice_data_qubits = Alice_info["data_qubit_list"]
    Alice_ancilla_x_qubits = Alice_info["ancilla_x_qubit_list"]
    Alice_ancilla_z_qubits = Alice_info["ancilla_z_qubit_list"]
    All_data_qubits = Alice_data_qubits
    All_qubits = Alice_data_qubits + Alice_ancilla_x_qubits + Alice_ancilla_z_qubits
    Ancilla_qubits = Alice_ancilla_x_qubits + Alice_ancilla_z_qubits
    Ancilla_x_qubits = Alice_ancilla_x_qubits
    Ancilla_z_qubits = Alice_ancilla_z_qubits

    circuit = stim.Circuit()
    if basis == 'Z':
        circuit.append("R", All_qubits)
        # add noise on data qubits only
        circuit.append('X_ERROR', All_data_qubits, p_local)
    else:
        # X<->Z mirror of the |0>_L encoder: data prepared in |+> with a Z preparation error
        circuit.append("RX", All_data_qubits)
        circuit.append("R", Ancilla_qubits)
        circuit.append('Z_ERROR', All_data_qubits, p_local)

    # encoding stabilizer measurement
    matrix_x = Code.matrix_x
    matrix_z = Code.matrix_z
    
    # encoding process:
    # 2 <-> 3, 5 <-> 6
    circuit.append("H", [Alice_data_qubits[0], Alice_data_qubits[1], Alice_data_qubits[3]])
    # basis='X': every encoder CNOT is applied with control and target exchanged (H-conjugate of the |0>_L encoder)
    _enc = (lambda a, b: [a, b]) if basis == 'Z' else (lambda a, b: [b, a])
    circuit.append("CX", _enc(Alice_data_qubits[5], Alice_data_qubits[2]))
    circuit.append('DEPOLARIZE2', [Alice_data_qubits[5], Alice_data_qubits[2]], p_local)
    circuit.append("TICK")
    circuit.append("CX", _enc(Alice_data_qubits[5], Alice_data_qubits[4]))
    circuit.append('DEPOLARIZE2', [Alice_data_qubits[5], Alice_data_qubits[4]], p_local)
    circuit.append("TICK")
    circuit.append("CX", _enc(Alice_data_qubits[0], Alice_data_qubits[2]))
    circuit.append('DEPOLARIZE2', [Alice_data_qubits[0], Alice_data_qubits[2]], p_local)
    circuit.append("TICK")
    circuit.append("CX", _enc(Alice_data_qubits[0], Alice_data_qubits[4]))
    circuit.append('DEPOLARIZE2', [Alice_data_qubits[0], Alice_data_qubits[4]], p_local)
    circuit.append("TICK")
    circuit.append("CX", _enc(Alice_data_qubits[0], Alice_data_qubits[6]))
    circuit.append('DEPOLARIZE2', [Alice_data_qubits[0], Alice_data_qubits[6]], p_local)
    circuit.append("TICK")
    circuit.append("CX", _enc(Alice_data_qubits[1], Alice_data_qubits[2]))
    circuit.append('DEPOLARIZE2', [Alice_data_qubits[1], Alice_data_qubits[2]], p_local)
    circuit.append("TICK")
    circuit.append("CX", _enc(Alice_data_qubits[1], Alice_data_qubits[6]))
    circuit.append('DEPOLARIZE2', [Alice_data_qubits[1], Alice_data_qubits[6]], p_local)
    circuit.append("TICK")
    circuit.append("CX", _enc(Alice_data_qubits[1], Alice_data_qubits[5]))
    circuit.append('DEPOLARIZE2', [Alice_data_qubits[1], Alice_data_qubits[5]], p_local)
    circuit.append("TICK")
    circuit.append("CX", _enc(Alice_data_qubits[3], Alice_data_qubits[4]))
    circuit.append('DEPOLARIZE2', [Alice_data_qubits[3], Alice_data_qubits[4]], p_local)
    circuit.append("TICK")
    circuit.append("CX", _enc(Alice_data_qubits[3], Alice_data_qubits[6]))
    circuit.append('DEPOLARIZE2', [Alice_data_qubits[3], Alice_data_qubits[6]], p_local)
    circuit.append("TICK")
    circuit.append("CX", _enc(Alice_data_qubits[3], Alice_data_qubits[5]))
    circuit.append('DEPOLARIZE2', [Alice_data_qubits[3], Alice_data_qubits[5]], p_local)
    circuit.append("TICK")
    
    # fault-tolerant encoding stabilizer check
    
    circuit.append("H", Ancilla_x_qubits)
    circuit.append("TICK")

    # initialize gates scheduling table: X stabilizer
    schedule = []

    # x stabilizer measurement
    for idx, ancilla_qubit in enumerate(Alice_ancilla_x_qubits):
        stabilizer = matrix_x[idx]
        data_qubits = [Alice_data_qubits[j] for j in np.where(stabilizer == 1)[0]]
        cnot_ops = [(ancilla_qubit, data_qubit) for data_qubit in data_qubits]
        schedule.extend(cnot_ops)

    time_steps = []

    while any(schedule):
        time_step_ops = []
        used_qubits = set()
        remaining_ops = []
        for op in schedule:
            if op[0] not in used_qubits and op[1] not in used_qubits:
                time_step_ops.append(op)
                used_qubits.update(op)
            else:
                remaining_ops.append(op)
        time_steps.append(time_step_ops)
        schedule = remaining_ops

    for time_step_ops in time_steps:
        CX_list = []
        single_qubits = set(All_qubits)
        
        for op in time_step_ops:
            # Alice
            CX_list.extend(op)
            single_qubits.difference_update(op)
        
        if CX_list:
            circuit.append('CX', CX_list)
            circuit.append('DEPOLARIZE2', CX_list, p_local)
            
        if single_qubits:
            circuit.append('DEPOLARIZE1', sorted(single_qubits), p_local)

        circuit.append("TICK")

    circuit.append("H", Ancilla_x_qubits)
    circuit.append("TICK")

    # z stabilizer
    for idx, ancilla_qubit in enumerate(Alice_ancilla_z_qubits):
        stabilizer = matrix_z[idx]
        data_qubits = [Alice_data_qubits[j] for j in np.where(stabilizer == 1)[0]]
        cnot_ops = [(data_qubit, ancilla_qubit) for data_qubit in data_qubits]
        schedule.extend(cnot_ops)

    time_steps = []

    while any(schedule):
        time_step_ops = []
        used_qubits = set()
        remaining_ops = []
        for op in schedule:
            if op[0] not in used_qubits and op[1] not in used_qubits:
                time_step_ops.append(op)
                used_qubits.update(op)
            else:
                remaining_ops.append(op)
        time_steps.append(time_step_ops)
        schedule = remaining_ops

    for time_step_ops in time_steps:
        CX_list = []
        single_qubits = set(All_qubits)
        
        for op in time_step_ops:
            # Alice
            CX_list.extend(op)
            single_qubits.difference_update(op)
        
        if CX_list:
            circuit.append('CX', CX_list)
            circuit.append('DEPOLARIZE2', CX_list, p_local)
            
        if single_qubits:
            circuit.append('DEPOLARIZE1', sorted(single_qubits), p_local)

        circuit.append("TICK")

    circuit.append('X_ERROR', Ancilla_qubits, p_local)
    circuit.append('MR', Ancilla_qubits)
    circuit.append('X_ERROR', Ancilla_qubits, p_local)
    


    # add z detectors of Alice
    detector_shift = 0
    for i in Ancilla_qubits[::-1]:
        detector_shift -= 1
        if zx_detector:
            non_detector_ancilla = []
        else:
            non_detector_ancilla = Alice_ancilla_x_qubits if basis == 'Z' else Alice_ancilla_z_qubits
        if i in non_detector_ancilla:
            continue
        circuit.append("DETECTOR", [stim.target_rec(detector_shift)], [i, 0])
        
    circuit.append("SHIFT_COORDS", [], [0, 1])
    
    # Alice qubits transimission
    circuit.append("DEPOLARIZE1", Alice_data_qubits, p_trans)

    # decoding stabilizer measurement
    
    circuit.append("H", Ancilla_x_qubits)
    circuit.append("TICK")

    # initialize gates scheduling table: X stabilizer
    schedule = []

    # x stabilizer measurement
    for idx, ancilla_qubit in enumerate(Alice_ancilla_x_qubits):
        stabilizer = matrix_x[idx]
        data_qubits = [Alice_data_qubits[j] for j in np.where(stabilizer == 1)[0]]
        cnot_ops = [(ancilla_qubit, data_qubit) for data_qubit in data_qubits]
        schedule.extend(cnot_ops)

    time_steps = []

    while any(schedule):
        time_step_ops = []
        used_qubits = set()
        remaining_ops = []
        for op in schedule:
            if op[0] not in used_qubits and op[1] not in used_qubits:
                time_step_ops.append(op)
                used_qubits.update(op)
            else:
                remaining_ops.append(op)
        time_steps.append(time_step_ops)
        schedule = remaining_ops

    for time_step_ops in time_steps:
        CX_list = []
        single_qubits = set(All_qubits)
        
        for op in time_step_ops:
            # Alice
            CX_list.extend(op)
            single_qubits.difference_update(op)
        
        if CX_list:
            circuit.append('CX', CX_list)
            circuit.append('DEPOLARIZE2', CX_list, p_local)
            
        if single_qubits:
            circuit.append('DEPOLARIZE1', sorted(single_qubits), p_local)

        circuit.append("TICK")

    circuit.append("H", Ancilla_x_qubits)
    circuit.append("TICK")

    # z stabilizer
    for idx, ancilla_qubit in enumerate(Alice_ancilla_z_qubits):
        stabilizer = matrix_z[idx]
        data_qubits = [Alice_data_qubits[j] for j in np.where(stabilizer == 1)[0]]
        cnot_ops = [(data_qubit, ancilla_qubit) for data_qubit in data_qubits]
        schedule.extend(cnot_ops)

    time_steps = []

    while any(schedule):
        time_step_ops = []
        used_qubits = set()
        remaining_ops = []
        for op in schedule:
            if op[0] not in used_qubits and op[1] not in used_qubits:
                time_step_ops.append(op)
                used_qubits.update(op)
            else:
                remaining_ops.append(op)
        time_steps.append(time_step_ops)
        schedule = remaining_ops

    for time_step_ops in time_steps:
        CX_list = []
        single_qubits = set(All_qubits)
        
        for op in time_step_ops:
            # Alice
            CX_list.extend(op)
            single_qubits.difference_update(op)
        
        if CX_list:
            circuit.append('CX', CX_list)
            circuit.append('DEPOLARIZE2', CX_list, p_local)
            
        if single_qubits:
            circuit.append('DEPOLARIZE1', sorted(single_qubits), p_local)

        circuit.append("TICK")

    circuit.append('X_ERROR', Ancilla_qubits, p_local)
    circuit.append('MR', Ancilla_qubits)
    circuit.append('X_ERROR', Ancilla_qubits, p_local)


    # add z detectors of Alice
    detector_shift = 0
    for i in Ancilla_qubits[::-1]:
        detector_shift -= 1
        if zx_detector:
            non_detector_ancilla = []
        else:
            non_detector_ancilla = Alice_ancilla_x_qubits if basis == 'Z' else Alice_ancilla_z_qubits
        if i in non_detector_ancilla:
            continue
        circuit.append("DETECTOR", [stim.target_rec(detector_shift), stim.target_rec(detector_shift - len(Ancilla_qubits))], [i, 0])

    # measure all data qubits
    circuit.append('M' if basis == 'Z' else 'MX', All_data_qubits)

    logical_x_operators = Code.get_logical_ops()[0]
    logical_z_operators = Code.get_logical_ops()[1]
        
    N = Code.num_qubits
    K = (len(logical_x_operators) + len(logical_z_operators)) // 2

    detector_list = []
    detector_list += Ancilla_qubits
    detector_list += All_data_qubits
    detector_shift_list = [(-len(detector_list) + i) for i in range(len(detector_list))]
        
    # perfect z measurement
    for i, detector in enumerate(detector_list):
        if detector in (Alice_ancilla_z_qubits if basis == 'Z' else Alice_ancilla_x_qubits):
            target_rec = []
            target_rec.append(stim.target_rec(detector_shift_list[i]))
            # one stabilizer check - many data qubits
            idx = (Alice_ancilla_z_qubits if basis == 'Z' else Alice_ancilla_x_qubits).index(detector)
            stabilizer = (matrix_z if basis == 'Z' else matrix_x)[idx]
            alice_data_qubits = [Alice_data_qubits[j] for j in np.where(stabilizer == 1)[0]]
            for check_qubit in alice_data_qubits:
                target_rec.append(stim.target_rec(detector_shift_list[detector_list.index(check_qubit)]))
            circuit.append("DETECTOR", target_rec, [detector, 0])
    
    
    for k in range(K):
        target_rec = []
        logical_z_operator = logical_z_operators[k]
        support = np.where(logical_z_operator == 1)[0] - N if basis == 'Z' else np.where(np.asarray(logical_x_operators[k])[:N] == 1)[0]
        data_qubits = [Alice_data_qubits[j] for j in support]
        for i in range(len(detector_list)):
            if detector_list[i] in data_qubits:
                target_rec.append(stim.target_rec(detector_shift_list[i]))
        circuit.append(stim.CircuitInstruction('OBSERVABLE_INCLUDE', target_rec, [k]))
    
    # circuit.to_file(f"Steane_Code_SE_Plus_{p_local}_{p_trans}.stim")
    return circuit

'''circuit of protocal II: Transveral CNOT using teleportation_Align'''
def EE_circuit_Transversal_CNOT_Teleportation_Aligned(Code, p_local, p_trans, basis = 'Z'):
    # basis='Z': observable Z_L(A) Z_L(B) with the X-frame from EPR_1 (original circuit).
    # basis='X': same preparation, gates and noise; noiseless MX readout, observable X_L(A) X_L(B)
    #            with the Z-frame from the X-basis outcomes of EPR_2.
    assert basis in ('Z', 'X')
    Alice_info = get_info(Code=Code, shift=0)
    Bob_Shift = Code.num_checks + Code.num_qubits
    Bob_info = get_info(Code=Code, shift=Bob_Shift)

    # initial circuits
    Alice_data_qubits = Alice_info["data_qubit_list"]
    Bob_data_qubits = Bob_info["data_qubit_list"]
    Alice_ancilla_x_qubits = Alice_info["ancilla_x_qubit_list"]
    Alice_ancilla_z_qubits = Alice_info["ancilla_z_qubit_list"]
    Bob_ancilla_x_qubits = Bob_info["ancilla_x_qubit_list"]
    Bob_ancilla_z_qubits = Bob_info["ancilla_z_qubit_list"]
    All_data_qubits = Alice_data_qubits + Bob_data_qubits
    All_qubits = Alice_data_qubits + Alice_ancilla_x_qubits + Alice_ancilla_z_qubits + Bob_data_qubits + Bob_ancilla_x_qubits + Bob_ancilla_z_qubits
    Ancilla_qubits = Alice_ancilla_x_qubits + Alice_ancilla_z_qubits + Bob_ancilla_x_qubits + Bob_ancilla_z_qubits
    Ancilla_x_qubits = Alice_ancilla_x_qubits + Bob_ancilla_x_qubits
    Ancilla_z_qubits = Alice_ancilla_z_qubits + Bob_ancilla_z_qubits

    EPR_Shift = 100
    EPR_1 = [i + EPR_Shift for i in Alice_data_qubits]
    EPR_2 = [i + EPR_Shift for i in Bob_data_qubits]
    All_qubits = Alice_data_qubits + Alice_ancilla_x_qubits + Alice_ancilla_z_qubits + Bob_data_qubits + Bob_ancilla_x_qubits + Bob_ancilla_z_qubits + EPR_1 + EPR_2
    # decoding stabilizer measurement
    matrix_x = Code.matrix_x
    matrix_z = Code.matrix_z

    circuit = stim.Circuit()
    circuit.append("R", All_qubits)
    # add noise on data qubits only
    circuit.append('X_ERROR', All_qubits, p_local)
    circuit.append("H", Alice_data_qubits)

    circuit.append("R", EPR_1 + EPR_2)
    circuit.append("H", EPR_1)

    for i in range(len(EPR_1)):
        circuit.append("CX", [EPR_1[i], EPR_2[i]])

    for i in range(len(EPR_1)):
        circuit.append("DEPOLARIZE2", [EPR_1[i], EPR_2[i]], p_local)

    # transmission
    circuit.append("DEPOLARIZE1", EPR_2, p_trans)
    
    def syndrome_extraction(circuit):
        circuit.append("H", Ancilla_x_qubits)
        circuit.append("TICK")

        # initialize gates scheduling table: X stabilizer
        schedule = []

        # x stabilizer measurement
        for idx, ancilla_qubit in enumerate(Alice_ancilla_x_qubits):
            stabilizer = matrix_x[idx]
            data_qubits = [Alice_data_qubits[j] for j in np.where(stabilizer == 1)[0]]
            cnot_ops = [(ancilla_qubit, data_qubit) for data_qubit in data_qubits]
            schedule.extend(cnot_ops)

        time_steps = []

        while any(schedule):
            time_step_ops = []
            used_qubits = set()
            remaining_ops = []
            for op in schedule:
                if op[0] not in used_qubits and op[1] not in used_qubits:
                    time_step_ops.append(op)
                    used_qubits.update(op)
                else:
                    remaining_ops.append(op)
            time_steps.append(time_step_ops)
            schedule = remaining_ops

        for time_step_ops in time_steps:
            CX_list = []
            single_qubits = set(All_qubits)

            for op in time_step_ops:
                # Alice
                CX_list.extend(op)
                single_qubits.difference_update(op)
                # Bob
                new_op = (op[0] + Bob_Shift, op[1] + Bob_Shift)
                CX_list.extend(new_op)
                single_qubits.difference_update(new_op)

            if CX_list:
                circuit.append('CX', CX_list)
                circuit.append('DEPOLARIZE2', CX_list, p_local)

            if single_qubits:
                circuit.append('DEPOLARIZE1', sorted(single_qubits), p_local)

            circuit.append("TICK")

        circuit.append("H", Ancilla_x_qubits)
        circuit.append("TICK")

        # z stabilizer
        for idx, ancilla_qubit in enumerate(Alice_ancilla_z_qubits):
            stabilizer = matrix_z[idx]
            data_qubits = [Alice_data_qubits[j] for j in np.where(stabilizer == 1)[0]]
            cnot_ops = [(data_qubit, ancilla_qubit) for data_qubit in data_qubits]
            schedule.extend(cnot_ops)

        time_steps = []

        while any(schedule):
            time_step_ops = []
            used_qubits = set()
            remaining_ops = []
            for op in schedule:
                if op[0] not in used_qubits and op[1] not in used_qubits:
                    time_step_ops.append(op)
                    used_qubits.update(op)
                else:
                    remaining_ops.append(op)
            time_steps.append(time_step_ops)
            schedule = remaining_ops

        for time_step_ops in time_steps:
            CX_list = []
            single_qubits = set(All_qubits)

            for op in time_step_ops:
                # Alice
                CX_list.extend(op)
                single_qubits.difference_update(op)
                # Bob
                new_op = (op[0] + Bob_Shift, op[1] + Bob_Shift)
                CX_list.extend(new_op)
                single_qubits.difference_update(new_op)

            if CX_list:
                circuit.append('CX', CX_list)
                circuit.append('DEPOLARIZE2', CX_list, p_local)

            if single_qubits:
                circuit.append('DEPOLARIZE1', sorted(single_qubits), p_local)

            circuit.append("TICK")
            
        circuit.append('X_ERROR', Ancilla_qubits, p_local)
        circuit.append('MR', Ancilla_qubits)
        circuit.append('X_ERROR', Ancilla_qubits, p_local)
        
        return circuit
    
    # stabilizer encoding circuit: actually no need
    
    circuit = syndrome_extraction(circuit)
    detector_list = []
    detector_list += Ancilla_qubits
    detector_shift_list = [(-len(detector_list) + i) for i in range(len(detector_list))]
    # First detector round: Initialize the |+⟩L and |0⟩L.
    detector_shift = 0
    for index in Ancilla_qubits[::-1]:
        detector_shift -= 1
        if index in Alice_ancilla_z_qubits + Bob_ancilla_x_qubits:
            continue
        circuit.append("DETECTOR", [stim.target_rec(detector_shift)], [index, 0])
    
    circuit.append("SHIFT_COORDS", [], [0, 1])
    

    '''Transversal CNOT by gate teleportation'''
    for i in range(len(Alice_data_qubits)):
        circuit.append("CX", [Alice_data_qubits[i], EPR_1[i]])
    for i in range(len(Alice_data_qubits)):
        circuit.append("DEPOLARIZE2", [Alice_data_qubits[i], EPR_1[i]], p_local)
    
    for i in range(len(Bob_data_qubits)):
        circuit.append("CX", [EPR_2[i], Bob_data_qubits[i]])
    for i in range(len(Bob_data_qubits)):
        circuit.append("DEPOLARIZE2", [EPR_2[i], Bob_data_qubits[i]], p_local)
    
    for i in range(len(EPR_2)):
        circuit.append_operation("H", [EPR_2[i]])
        
    
    circuit.append("X_ERROR", EPR_1 + EPR_2, p_local)
    circuit.append("M", EPR_1 + EPR_2)
    

    circuit = syndrome_extraction(circuit)
    
    # add x/z detectors of Alice and Bob
    detector_list = []
    detector_list += (EPR_1 + EPR_2)
    detector_list += Ancilla_qubits
    detector_shift_list = [(-len(detector_list) + i) for i in range(len(detector_list))]
    
    if basis == 'Z':
      for idx, ancilla_qubit in enumerate(Bob_ancilla_z_qubits):
        stabilizer = matrix_z[idx]
        data_qubits = [Bob_data_qubits[j] for j in np.where(stabilizer == 1)[0]]
        target_rec = []
        alice_ancilla_qubit = ancilla_qubit - Bob_Shift
        target_rec.append(stim.target_rec(detector_shift_list[detector_list.index(ancilla_qubit)]))
        target_rec.append(stim.target_rec(detector_shift_list[detector_list.index(alice_ancilla_qubit)]))
        
        ''' Pauli Frame '''
        for bob_data_qubit in data_qubits:
            alice_data_qubit = bob_data_qubit - Bob_Shift
            epr1_qubit = alice_data_qubit + EPR_Shift
            target_rec.append(stim.target_rec(detector_shift_list[detector_list.index(epr1_qubit)]))
            
        circuit.append("DETECTOR", target_rec, [ancilla_qubit, 0])
    else:
      # X-type mirror: X_A(r2) X_B(r2) times the Z-frame on Alice (X-basis outcomes of EPR_2 on the check support)
      for idx, ancilla_qubit in enumerate(Alice_ancilla_x_qubits):
        stabilizer = matrix_x[idx]
        data_qubits = [Alice_data_qubits[j] for j in np.where(stabilizer == 1)[0]]
        target_rec = []
        bob_ancilla_qubit = ancilla_qubit + Bob_Shift
        target_rec.append(stim.target_rec(detector_shift_list[detector_list.index(ancilla_qubit)]))
        target_rec.append(stim.target_rec(detector_shift_list[detector_list.index(bob_ancilla_qubit)]))

        ''' Pauli Frame '''
        for alice_data_qubit in data_qubits:
            epr2_qubit = alice_data_qubit + Bob_Shift + EPR_Shift
            target_rec.append(stim.target_rec(detector_shift_list[detector_list.index(epr2_qubit)]))

        circuit.append("DETECTOR", target_rec, [ancilla_qubit, 0])

    circuit.append("SHIFT_COORDS", [], [0, 1])

    # measure all data qubits
    # circuit.append("H", Alice_data_qubits)
    circuit.append('M' if basis == 'Z' else 'MX', All_data_qubits)

    logical_x_operators = Code.get_logical_ops()[0]
    logical_z_operators = Code.get_logical_ops()[1]

    N = Code.num_qubits
    K = (len(logical_x_operators) + len(logical_z_operators)) // 2

    detector_list = []
    detector_list += (EPR_1 + EPR_2)
    detector_list += Ancilla_qubits
    detector_list += All_data_qubits
    detector_shift_list = [(-len(detector_list) + i) for i in range(len(detector_list))]

    # perfect z measurement
    for i, detector in enumerate(detector_list):
        if detector in (Alice_ancilla_z_qubits if basis == 'Z' else Alice_ancilla_x_qubits):
            target_rec = []
            target_rec.append(stim.target_rec(detector_shift_list[i]))
            # one stabilizer check - many data qubits
            idx = (Alice_ancilla_z_qubits if basis == 'Z' else Alice_ancilla_x_qubits).index(detector)
            stabilizer = (matrix_z if basis == 'Z' else matrix_x)[idx]
            alice_data_qubits = [Alice_data_qubits[j] for j in np.where(stabilizer == 1)[0]]
            for check_qubit in alice_data_qubits:
                target_rec.append(stim.target_rec(detector_shift_list[detector_list.index(check_qubit)]))
            circuit.append("DETECTOR", target_rec, [detector, 0])
            
    for i, detector in enumerate(detector_list):
        if detector in (Bob_ancilla_z_qubits if basis == 'Z' else Bob_ancilla_x_qubits):
            target_rec = []
            target_rec.append(stim.target_rec(detector_shift_list[i]))
            # one stabilizer check - many data qubits
            idx = (Bob_ancilla_z_qubits if basis == 'Z' else Bob_ancilla_x_qubits).index(detector)
            stabilizer = (matrix_z if basis == 'Z' else matrix_x)[idx]
            bob_data_qubits = [Bob_data_qubits[j] for j in np.where(stabilizer == 1)[0]]
            for check_qubit in bob_data_qubits:
                target_rec.append(stim.target_rec(detector_shift_list[detector_list.index(check_qubit)]))
            circuit.append("DETECTOR", target_rec, [detector, 0])
    
    for k in range(K):
        target_rec = []
        logical_z_operator = logical_z_operators[k]
        support = np.where(logical_z_operator == 1)[0] - N if basis == 'Z' else np.where(np.asarray(logical_x_operators[k])[:N] == 1)[0]
        data_qubits = [Alice_data_qubits[j] for j in support] + [Bob_data_qubits[j] for j in support]
        for i in range(len(detector_list)):
            if detector_list[i] in data_qubits:
                target_rec.append(stim.target_rec(detector_shift_list[i]))
                ''' Pauli Frame '''
                if basis == 'Z' and detector_list[i] in Bob_data_qubits:
                    alice_data_qubit = detector_list[i] - Bob_Shift
                    epr1_qubit = alice_data_qubit + EPR_Shift
                    target_rec.append(stim.target_rec(detector_shift_list[detector_list.index(epr1_qubit)]))
                if basis == 'X' and detector_list[i] in Alice_data_qubits:
                    # Z-frame on Alice: X-basis outcome of EPR_2 at the same code position
                    epr2_qubit = detector_list[i] + Bob_Shift + EPR_Shift
                    target_rec.append(stim.target_rec(detector_shift_list[detector_list.index(epr2_qubit)]))
                
        circuit.append(stim.CircuitInstruction('OBSERVABLE_INCLUDE', target_rec, [k]))

    # circuit.to_file(f"Steane_Code_Transversal_2nd_{p_local}_{p_trans}_Aligned.stim")
    return circuit