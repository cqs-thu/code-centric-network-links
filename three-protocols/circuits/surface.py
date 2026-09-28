import numpy as np
import stim
import pymatching
from tqdm import tqdm
import time

def EPR_QECC_Protocal_Circuit(d, num_layer, p_local=0.001, p_trans=0.001, basis="Z"):
    """Fault Tolerant EPR State Preparation and Distribution Circuit of Rotated Surface Code [QECC]
    
    (1) Initialization: Alice: |+⟩, Bob: |+⟩.
    
    (2) Logical ZZ measurement by Alice.
        If M_{ZZ} = +1: EPR of Alice-Bob = √2/2 * (|00⟩ + |11⟩)
        If M_{ZZ} = -1: EPR of Alice-Bob = √2/2 * (|01⟩ + |10⟩)
        
    (3) Send Bob's patch to Bob.
    
    (4) Quantum memory: Alice and Bob protect their entangled EPR states by quantum memory until they need the final states.
    
    (5) Final measurement: Finally, Alice and Bob measure their data qubits and perform decoding to get the logical results.
        Actually, the decoding should be performed together. So one classical communication process is still needed.
    
    (6) Check fidelity: The decoding results should be consistent with the true EPR states.
        Actually, the decoder should correct all the errors during the memory and ZZ measurement process.

    Args:
        d (INT): Code distance 
        num_layer (int): Measurement rounds
        p_local (float, optional): Physical error rate of local operations. Defaults to 0.001.
        p_trans (float, optional): Physical error rate of transmission. Defaults to 0.001.
        basis (str, optional): "Z" (default) reads the Z_L Z_L parity of the Bell pair with a noiseless M readout;
            "X" reads the X_L X_L parity with a noiseless MX readout. Everything before the final readout is identical.

    Returns:
        Stim.Circuit(): An object of Stim.Circuit()
    """
    
    circuit = stim.Circuit()

    data_qubit_per_row = d
    data_qubit_per_col = d

    data_qubits_list = []

    Bob_data_qubits = []
    Alice_data_qubits = []

    Alice_Z_Logical_Operator = []
    Bob_Z_Logical_Operator = []
    # [basis] X_L of each patch: row 0 (x = 1)
    Alice_X_Logical_Operator = []
    Bob_X_Logical_Operator = []
    assert basis in ("Z", "X")

    stabilizer_x_list = []
    stabilizer_z_list = []

    alice_stabilizer_x_list = []
    bob_stabilizer_x_list = []
    alice_stabilizer_z_list = []
    bob_stabilizer_z_list = []

    qubit_index_position = {}

    # append data qubits of Alice
    for col in range(data_qubit_per_row):
        index_shift = col
        for row in range(data_qubit_per_col):
            X = 2 * row + 1
            Y = 2 * col + 1
            index = 2 * (row + col * d) + 1 + index_shift
            circuit.append("QUBIT_COORDS", [index], [X, Y])
            data_qubits_list.append(index)
            Alice_data_qubits.append(index)
            if col == d - 1:
                Alice_Z_Logical_Operator.append(index)
            if row == 0:
                Alice_X_Logical_Operator.append(index)
            qubit_index_position[index] = (X, Y)

    # all the indices of Bob's qubits should be Alice_Qubit[i] + Bob_Shift
    Bob_Shift = 4 * d**2
    Bob_Y_Shift = 2 * d + 1
    # append data qubits of Bob
    for col in range(data_qubit_per_row):
        index_shift = col
        for row in range(data_qubit_per_col):
            X = 2 * row + 1
            Y = 2 * col + 1 + Bob_Y_Shift
            index = 2 * (row + col * d) + 1 + index_shift + Bob_Shift
            circuit.append("QUBIT_COORDS", [index], [X, Y])
            data_qubits_list.append(index)
            Bob_data_qubits.append(index)
            if col == 0:
                Bob_Z_Logical_Operator.append(index)
            if row == 0:
                Bob_X_Logical_Operator.append(index)
            qubit_index_position[index] = (X, Y)

    # append boundary z stabilizer ancilla qubit (Alice)
    for col in range(data_qubit_per_row):
        index_shift = col
        # up z boundary - data qubit - right up
        for row in [0]:
            if col % 2 == 1:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1
            x_z = X - 1
            y_z = Y - 1
            if y_z <= 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_z = index - 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            alice_stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)
        # down z boundary - data qubit - right down
        for row in [data_qubit_per_col - 1]:
            if col % 2 == 0:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1
            x_z = X + 1
            y_z = Y - 1
            if y_z <= 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_z = index + 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            alice_stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)

    # append boundary x stabilizer ancilla qubit (Alice)
    for row in range(data_qubit_per_col):
        # right x boundary - data qubit - right up
        for col in [0]:
            index_shift = col
            if row % 2 == 1:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1
            x_x = X + 1
            y_x = Y - 1
            if x_x >= data_qubit_per_col * 2:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_x = index + 1
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            alice_stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)
        # left x boundary - data qubit - right down
        for col in [data_qubit_per_row - 1]:
            index_shift = col
            if row % 2 == 0:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1
            x_x = X + 1
            y_x = Y + 1
            index = 2 * (row + col * d) + 1 + index_shift
            index_x = index + 1 + (2 * d + 1)
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            alice_stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)

    # append boundary z stabilizer ancilla qubit (Bob)
    for col in range(data_qubit_per_row):
        index_shift = col
        # up z boundary - data qubit - right up
        for row in [0]:
            if col % 2 == 0:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1 + Bob_Y_Shift
            x_z = X - 1
            y_z = Y - 1
            if y_z - Bob_Y_Shift <= 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift + Bob_Shift
            index_z = index - 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            bob_stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)
        # down z boundary - data qubit - right down
        for row in [data_qubit_per_col - 1]:
            if col % 2 == 1:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1 + Bob_Y_Shift
            x_z = X + 1
            y_z = Y - 1
            if y_z - Bob_Y_Shift <= 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift + Bob_Shift
            index_z = index + 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            bob_stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)

    # append boundary x stabilizer ancilla qubit (Bob)
    for row in range(data_qubit_per_col):
        # right x boundary - data qubit - right up
        for col in [0]:
            index_shift = col
            if row % 2 == 0:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1 + Bob_Y_Shift
            x_x = X + 1
            y_x = Y - 1
            if x_x >= data_qubit_per_col * 2:
                continue
            index = 2 * (row + col * d) + 1 + index_shift + Bob_Shift
            index_x = index + 1
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            bob_stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)
        # left x boundary - data qubit - right down
        for col in [data_qubit_per_row - 1]:
            index_shift = col
            if row % 2 == 1:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1 + Bob_Y_Shift
            x_x = X + 1
            y_x = Y + 1
            if x_x >= data_qubit_per_col * 2:
                continue
            index = 2 * (row + col * d) + 1 + index_shift + Bob_Shift
            index_x = index + 1 + (2 * d + 1)
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            bob_stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)


    # append z stabilizer (general case of Alice)
    for row in range(data_qubit_per_col - 1):
        for col in range(1, data_qubit_per_row):
            index_shift = col
            X = 2 * row + 1
            Y = 2 * col + 1
            x_z = X + 1
            y_z = Y - 1
            if ((x_z + y_z) // 2) % 2 == 1:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_z = index + 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            alice_stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)

    # append x stabilizer (general case of Alice)
    for row in range(data_qubit_per_col - 1):
        for col in range(1, data_qubit_per_row):
            index_shift = col
            X = 2 * row + 1
            Y = 2 * col + 1
            x_x = X + 1
            y_x = Y - 1
            if ((x_x + y_x) // 2) % 2 == 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_x = index + 1
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            alice_stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)


    # append z stabilizer (general case of Bob)
    for row in range(data_qubit_per_col - 1):
        for col in range(1, data_qubit_per_row):
            index_shift = col
            X = 2 * row + 1
            Y = 2 * col + 1 + Bob_Y_Shift
            x_z = X + 1
            y_z = Y - 1
            if ((x_z + y_z) // 2) % 2 == 1:
                continue
            index = 2 * (row + col * d) + 1 + index_shift + Bob_Shift
            index_z = index + 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            bob_stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)

    # append x stabilizer (general case of Bob)
    for row in range(data_qubit_per_col - 1):
        for col in range(1, data_qubit_per_row):
            index_shift = col
            X = 2 * row + 1
            Y = 2 * col + 1 + Bob_Y_Shift
            x_x = X + 1
            y_x = Y - 1
            if ((x_x + y_x) // 2) % 2 == 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift + Bob_Shift
            index_x = index + 1
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            bob_stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)


    # (1) State preparation of Alice and Bob |+⟩_L ⊗ |+⟩_L memory
    qubits_all_list = data_qubits_list + stabilizer_x_list + stabilizer_z_list
    stabilizer_list = stabilizer_x_list + stabilizer_z_list

    circuit.append("R", qubits_all_list)
    circuit.append('X_ERROR', qubits_all_list, p_local)
    circuit.append("H", data_qubits_list)
    circuit.append("TICK")


    def syndrome_extraction_initialization(circuit):
        """Simulate the initialization circuit of syndrome extraction in the Surface Code

        Args:
            circuit (Stim.Circuit): Origin circuit

        Returns:
            circuit (Stim.Circuit): Circuit with syndrome extraction appended
        """
        circuit.append("H", stabilizer_x_list)
        circuit.append("TICK")
        # 1 - Syndrome Extraction
        CX_list1 = []
        for i in stabilizer_x_list:
            if i + 1 in data_qubits_list:
                CX_list1.extend([i, i + 1])
        for i in data_qubits_list:
            if i - 1 in stabilizer_z_list:
                CX_list1.extend([i, i - 1])

        idle1 = list(set(qubits_all_list) - set(CX_list1))

        circuit.append('CX', CX_list1)
        circuit.append('DEPOLARIZE2', CX_list1, p_local)
        circuit.append('DEPOLARIZE1', idle1, p_local)
        circuit.append("TICK")

        # 2 - Syndrome Extraction
        CX_list2 = []
        for i in stabilizer_x_list:
            if i - 1 in data_qubits_list:
                CX_list2.extend([i, i - 1])
        for i in data_qubits_list:
            if i + 2 * d in stabilizer_z_list:
                CX_list2.extend([i, i + 2 * d])

        idle2 = list(set(qubits_all_list) - set(CX_list2))

        circuit.append('CX', CX_list2)
        circuit.append('DEPOLARIZE2', CX_list2, p_local)
        circuit.append('DEPOLARIZE1', idle2, p_local)
        circuit.append("TICK")

        # 3 - Syndrome Extraction
        CX_list3 = []
        for i in stabilizer_x_list:
            if i - 2 * d in data_qubits_list:
                CX_list3.extend([i, i - 2 * d])
        for i in data_qubits_list:
            if i + 1 in stabilizer_z_list:
                CX_list3.extend([i, i + 1])

        idle3 = list(set(qubits_all_list) - set(CX_list3))

        circuit.append('CX', CX_list3)
        circuit.append('DEPOLARIZE2', CX_list3, p_local)
        circuit.append('DEPOLARIZE1', idle3, p_local)
        circuit.append("TICK")

        # 4 - Syndrome Extraction
        CX_list4 = []
        for i in stabilizer_x_list:
            if i - (2 * d + 2) in data_qubits_list:
                CX_list4.extend([i, i - (2 * d + 2)])
        for i in data_qubits_list:
            if i + (2 * d + 2) in stabilizer_z_list:
                CX_list4.extend([i, i + (2 * d + 2)])

        idle4 = list(set(qubits_all_list) - set(CX_list4))

        circuit.append('CX', CX_list4)
        circuit.append('DEPOLARIZE2', CX_list4, p_local)
        circuit.append('DEPOLARIZE1', idle4, p_local)
        circuit.append("TICK")

        circuit.append("H", stabilizer_x_list)
        circuit.append("TICK")

        circuit.append('X_ERROR', stabilizer_list, p_local)
        circuit.append('MR', stabilizer_list)
        circuit.append('X_ERROR', stabilizer_list, p_local)

        return circuit


    circuit = syndrome_extraction_initialization(circuit)
    # print(stabilizer_x_list)

    # first layer : only x detectors, no z detectors
    detector_shift = 0
    for i in stabilizer_list[::-1]:
        detector_shift -= 1
        if i in stabilizer_x_list:
            X, Y = qubit_index_position[i]
            circuit.append("DETECTOR", [stim.target_rec(detector_shift)], [X, Y, 0])

    # Add repeat block for initialization of |+⟩_L ⊗ |+⟩_L memory
    repeated_block = stim.Circuit()
    repeated_block.append('DEPOLARIZE1', data_qubits_list, p_local)
    repeated_block.append("TICK")
    repeated_block = syndrome_extraction_initialization(repeated_block)
    repeated_block.append("SHIFT_COORDS", [], [0, 0, 1])

    # mid layer : all x detectors and z detectors
    detector_shift = 0
    for i in stabilizer_list[::-1]:
        detector_shift -= 1
        X, Y = qubit_index_position[i]
        repeated_block.append("DETECTOR", [stim.target_rec(detector_shift), stim.target_rec(detector_shift - len(stabilizer_list))], [X, Y, 0])

    # circuit += repeated_block
 
    # set up the middle ancilla qubits
    mid_shift = 2 * d
    delta_bob = 2 * d**2 - d
    mid_stabilizer_x_list = []
    mid_stabilizer_z_list = []
    mid_stabilizer_pair = {}
    inter_drop_set_x = []
    pair_single_mid_stabilizer = {}
    mid_stabilizer_single_list = []

    for i, qubit in enumerate(Alice_Z_Logical_Operator):
        X, Y = qubit_index_position[qubit]
        if ((X + 1)//2) % 2 == 1:
            # add middle z stabilizer
            index_z = qubit + mid_shift + 2 * d + 2
            mid_stabilizer_z_list.append(index_z)
            x_z = X + 1
            y_z = Y + 1.5
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            qubit_index_position[index_z] = (x_z, y_z)
        else:
            # add middle x stabilizer
            index_x = qubit + mid_shift + 2 * d + 2
            mid_stabilizer_x_list.append(qubit + mid_shift + 2 * d + 2)
            x_x = X + 1
            y_x = Y + 1.5
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            qubit_index_position[index_x] = (x_x, y_x)
            alice_stabilizer = Alice_Z_Logical_Operator[i] + 2 * d + 2
            bob_stabilizer = Bob_Z_Logical_Operator[i] + 1
            inter_drop_set_x.extend([alice_stabilizer, bob_stabilizer])
            mid_stabilizer_pair[index_x] = (alice_stabilizer, bob_stabilizer)
            pair_single_mid_stabilizer[alice_stabilizer] = index_x
            pair_single_mid_stabilizer[bob_stabilizer] = index_x
            mid_stabilizer_single_list.extend([alice_stabilizer, bob_stabilizer])

    mid_stabilizer_list = mid_stabilizer_x_list + mid_stabilizer_z_list
    temp_x_stabilizer_list = list(set(bob_stabilizer_x_list + mid_stabilizer_x_list + alice_stabilizer_x_list) - set(inter_drop_set_x))
    temp_z_stabilizer_list = bob_stabilizer_z_list + mid_stabilizer_z_list + alice_stabilizer_z_list
    temp_stabilizer_list = temp_x_stabilizer_list + temp_z_stabilizer_list
    temp_data_qubits = Bob_data_qubits + Alice_data_qubits


    def syndrome_extraction_zz_measurement(circuit):
        """Simulate the zz measurement circuit of syndrome extraction in the Surface Code (On Bob's patch only)

        Args:
            circuit (Stim.Circuit): Origin circuit

        Returns:
            circuit (Stim.Circuit): Circuit with syndrome extraction appended
        """
        circuit.append("H", temp_x_stabilizer_list)
        circuit.append("TICK")
        MAGIC_NUM = delta_bob - mid_shift
        # 1 - Syndrome Extraction
        CX_list1 = []
        # left-down data qubits X measurement
        for i in temp_x_stabilizer_list:
            if i in (bob_stabilizer_x_list + alice_stabilizer_x_list) and i + 1 in temp_data_qubits:
                CX_list1.extend([i, i + 1])
            if i in mid_stabilizer_x_list and i + MAGIC_NUM + 1 in temp_data_qubits:
                CX_list1.extend([i, i + MAGIC_NUM + 1])
        # left-down data qubits Z measurement
        for i in data_qubits_list:
            if i - 1 in (bob_stabilizer_z_list + alice_stabilizer_z_list):
                CX_list1.extend([i, i - 1])
            if i in Bob_Z_Logical_Operator and i - MAGIC_NUM - 1 in mid_stabilizer_z_list:
                CX_list1.extend([i, i - MAGIC_NUM - 1])

        idle1 = list(set(qubits_all_list) - set(CX_list1))
        circuit.append('CX', CX_list1)
        circuit.append('DEPOLARIZE2', CX_list1, p_local)
        circuit.append('DEPOLARIZE1', idle1, p_local)
        circuit.append("TICK")

        # 2 - Syndrome Extraction
        CX_list2 = []
        # left-up data qubits X measurement
        for i in temp_x_stabilizer_list:
            if i in (bob_stabilizer_x_list + alice_stabilizer_x_list) and i - 1 in temp_data_qubits:
                CX_list2.extend([i, i - 1])
            if i in mid_stabilizer_x_list and i + MAGIC_NUM - 1 in temp_data_qubits:
                CX_list2.extend([i, i + MAGIC_NUM - 1])
        # right-down data qubit Z measurement
        for i in temp_data_qubits:
            if i not in Alice_Z_Logical_Operator and i + 2 * d in (bob_stabilizer_z_list + alice_stabilizer_z_list):
                CX_list2.extend([i, i + 2 * d])
            if i in Alice_Z_Logical_Operator and i + mid_shift + 2 * d in mid_stabilizer_z_list:
                CX_list2.extend([i, i + mid_shift + 2 * d])

        idle2 = list(set(qubits_all_list) - set(CX_list2))
        circuit.append('CX', CX_list2)
        circuit.append('DEPOLARIZE2', CX_list2, p_local)
        circuit.append('DEPOLARIZE1', idle2, p_local)
        circuit.append("TICK")

        # 3 - Syndrome Extraction
        CX_list3 = []
        # right-down data qubits X measurement
        for i in temp_x_stabilizer_list:
            if i in (bob_stabilizer_x_list + alice_stabilizer_x_list) and i - 2 * d in data_qubits_list:
                CX_list3.extend([i, i - 2 * d])
            if i in mid_stabilizer_x_list and i - mid_shift - 2 * d in Alice_Z_Logical_Operator:
                CX_list3.extend([i, i - mid_shift - 2 * d])
        # left-up data qubit Z measurement
        for i in temp_data_qubits:
            if i + 1 in (bob_stabilizer_z_list + alice_stabilizer_z_list):
                CX_list3.extend([i, i + 1])
            if i - MAGIC_NUM + 1 in mid_stabilizer_z_list:
                CX_list3.extend([i, i - MAGIC_NUM + 1])

        idle3 = list(set(qubits_all_list) - set(CX_list3))
        circuit.append('CX', CX_list3)
        circuit.append('DEPOLARIZE2', CX_list3, p_local)
        circuit.append('DEPOLARIZE1', idle3, p_local)
        circuit.append("TICK")

        # 4 - Syndrome Extraction
        CX_list4 = []
        # right-up data qubits X measurement
        for i in temp_x_stabilizer_list:
            if i in (bob_stabilizer_x_list + alice_stabilizer_x_list) and i - (2 * d + 2) in data_qubits_list:
                CX_list4.extend([i, i - (2 * d + 2)])
            if i in mid_stabilizer_x_list and i - mid_shift - (2 * d + 2) in Alice_Z_Logical_Operator:
                CX_list4.extend([i, i - mid_shift - (2 * d + 2)])
        # right-up data qubit Z measurement
        for i in temp_data_qubits:
            if i in data_qubits_list and i + (2 * d + 2) in (bob_stabilizer_z_list + alice_stabilizer_z_list):
                CX_list4.extend([i, i + (2 * d + 2)])
            if i in Alice_Z_Logical_Operator and i + mid_shift + (2 * d + 2) in mid_stabilizer_z_list:
                CX_list4.extend([i, i + mid_shift + (2 * d + 2)])

        idle4 = list(set(qubits_all_list) - set(CX_list4))
        circuit.append('CX', CX_list4)
        circuit.append('DEPOLARIZE2', CX_list4, p_local)
        circuit.append('DEPOLARIZE1', idle4, p_local)
        circuit.append("TICK")

        circuit.append("H", temp_x_stabilizer_list)
        circuit.append("TICK")

        circuit.append('X_ERROR', temp_stabilizer_list, p_local)
        circuit.append('MR', temp_stabilizer_list)
        circuit.append('X_ERROR', temp_stabilizer_list, p_local)

        return circuit


    # (2) Logical ZZ measurement
    # there should be a different detector round before the repeated block: for drop out of the ancilla stabilizers
    circuit = syndrome_extraction_zz_measurement(circuit)

    detector_list = []
    detector_list += stabilizer_list
    detector_list += temp_stabilizer_list

    interim_detector_list = []
    interim_detector_list += stabilizer_list
    interim_detector_list += temp_stabilizer_list
    interim_detector_shift_list = [(-len(interim_detector_list) + i) for i in range(len(interim_detector_list))]
    interim_detector_list_set = set(interim_detector_list)

    circuit.append("SHIFT_COORDS", [], [0, 0, 1])

    detector_shift = 0
    for index in temp_stabilizer_list[::-1]:
        detector_shift -= 1
        X, Y = qubit_index_position[index]
        target_rec = []
        if index in mid_stabilizer_z_list:
            continue
        if index in mid_stabilizer_x_list:
            pair_a, pair_b = mid_stabilizer_pair[index]
            target_rec.append(stim.target_rec(detector_shift))
            target_rec.append(stim.target_rec(interim_detector_shift_list[interim_detector_list.index(pair_a)]))
            target_rec.append(stim.target_rec(interim_detector_shift_list[interim_detector_list.index(pair_b)]))
        if index in stabilizer_list:
            target_rec.extend([stim.target_rec(detector_shift), stim.target_rec(interim_detector_shift_list[interim_detector_list.index(index)])])
        circuit.append("DETECTOR", target_rec, [X, Y, 0])

    repeated_block = stim.Circuit()
    repeated_block.append('DEPOLARIZE1', data_qubits_list, p_local)
    repeated_block.append("TICK")
    repeated_block = syndrome_extraction_zz_measurement(repeated_block)
    repeated_block.append("SHIFT_COORDS", [], [0, 0, 1])

    detector_shift = 0
    for index in temp_stabilizer_list[::-1]:
        detector_shift -= 1
        X, Y = qubit_index_position[index]
        repeated_block.append("DETECTOR", [stim.target_rec(detector_shift), stim.target_rec(detector_shift - len(temp_stabilizer_list))], [X, Y, 0])

    
    # perform d rounds of syndrome measurement on Bob - ZZ measurement
    circuit.append(stim.CircuitRepeatBlock(d - 1, repeated_block))
    for i in range(d - 1):
        detector_list += temp_stabilizer_list

    final_detector_list = []
    final_detector_list += temp_stabilizer_list

    # (3) Split, send Bob's data qubits to Bob
    circuit.append('DEPOLARIZE1', Bob_data_qubits, p_trans)

    circuit = syndrome_extraction_initialization(circuit)
    detector_list += stabilizer_list
    detector_shift_list = [(-len(detector_list) + i) for i in range(len(detector_list))]

    # split operatation, there should be a different detector round before the repeated block: for drop out of the ancilla stabilizers
    interim_detector_list = []
    interim_detector_list += temp_stabilizer_list
    interim_detector_list += stabilizer_list
    interim_detector_shift_list = [(-len(interim_detector_list) + i) for i in range(len(interim_detector_list))]
    interim_detector_list_set = set(interim_detector_list)

    final_detector_list += stabilizer_list

    circuit.append("SHIFT_COORDS", [], [0, 0, 1])
    detector_shift = 0
    for index in stabilizer_list[::-1]:
        detector_shift -= 1
        X, Y = qubit_index_position[index]
        target_rec = []
        if index in mid_stabilizer_single_list:
            if Y <= 2 * d:
                continue
            previous_detector = pair_single_mid_stabilizer[index]
            pair_a, pair_b = mid_stabilizer_pair[previous_detector]
            target_rec.append(stim.target_rec(interim_detector_shift_list[interim_detector_list.index(pair_a)]))
            target_rec.append(stim.target_rec(interim_detector_shift_list[interim_detector_list.index(pair_b)]))
            target_rec.append(stim.target_rec(interim_detector_shift_list[interim_detector_list.index(previous_detector)]))
        elif index in stabilizer_list:
            target_rec.extend([stim.target_rec(detector_shift), stim.target_rec(interim_detector_shift_list[interim_detector_list.index(index)])])
        circuit.append("DETECTOR", target_rec, [X, Y, 0])

    # (4) Quantum memory of the EPR state
    repeated_block = stim.Circuit()
    repeated_block.append('DEPOLARIZE1', data_qubits_list, p_local)
    repeated_block.append("TICK")
    repeated_block = syndrome_extraction_initialization(repeated_block)
    repeated_block.append("SHIFT_COORDS", [], [0, 0, 1])

    # mid layer : all x detectors and z detectors
    detector_shift = 0
    for i in stabilizer_list[::-1]:
        detector_shift -= 1
        X, Y = qubit_index_position[i]
        repeated_block.append("DETECTOR", [stim.target_rec(detector_shift), stim.target_rec(detector_shift - len(stabilizer_list))], [X, Y, 0])
    if num_layer > 1:
        circuit.append(stim.CircuitRepeatBlock(num_layer - 1, repeated_block))
        for i in range(num_layer - 1):
            final_detector_list += stabilizer_list

    # (5) Final measurement
    # Measure all data qubits at Z basis
    # circuit.append('X_ERROR', data_qubits_list, p_local)
    # [basis] noiseless readout of all data qubits in the Z (M) or X (MX) basis
    circuit.append('M' if basis == "Z" else 'MX', data_qubits_list)
    final_detector_list += data_qubits_list
    final_detector_shift_list = [(-len(final_detector_list) + i) for i in range(len(final_detector_list))]

    final_detector_set = set(final_detector_list)
    final_data_qubits_set = set(data_qubits_list)
    # [basis] readout detectors from the Z-type checks (M) or the X-type checks (MX); the support offsets
    # (-1, +1, -2d-2, -2d) are the CX partners of both ancilla types in syndrome_extraction_initialization
    final_stabilizer_z_set = set(stabilizer_z_list) if basis == "Z" else set(stabilizer_x_list)
    # perform "perfect" measurement of z stabilizers on Alice's and Bob's patch
    for i, detector in enumerate(final_detector_list):
        if detector in final_stabilizer_z_set:
            # only the last rounds of stabilizer z measurement results are used for perfect measurement
            if (len(final_detector_list) - i) > len(qubits_all_list):
                continue
            X, Y = qubit_index_position[detector]
            target_rec = []
            target_rec.append(stim.target_rec(final_detector_shift_list[i]))
            # 1 stabilizer - "4" data qubits
            possible_neighbors = [
                detector - 1,
                detector + 1,
                detector - 2 * d - 2,
                detector - 2 * d
            ]
            for neighbor in possible_neighbors:
                if neighbor in final_detector_set and neighbor in final_data_qubits_set:
                    target_rec.append(stim.target_rec(final_detector_shift_list[final_detector_list.index(neighbor)]))
            circuit.append("DETECTOR", target_rec, [X, Y, 1])

    # (6) Check fidelity, logical observables
    # obserable = $M_{Z_{L1}Z_{L2}} ^ Z^{\prime}_{L1} ^ Z^{\prime}_{L2}$
    # [basis] X: observable = X_L(A) X_L(B). X_L(A) X_L(B) commutes with the merged checks and is +1 on |+>_L|+>_L,
    # so no lattice-surgery outcome enters it.
    target_rec = []
    if basis == "Z":
        # append M_{Z_{L1}Z_{L2}} result
        for i in range(len(temp_stabilizer_list)):
            if final_detector_list[i] in mid_stabilizer_z_list:
                target_rec.append(stim.target_rec(final_detector_shift_list[i]))
    Alice_Logical_Operator = Alice_Z_Logical_Operator if basis == "Z" else Alice_X_Logical_Operator
    Bob_Logical_Operator = Bob_Z_Logical_Operator if basis == "Z" else Bob_X_Logical_Operator
    # append Alice's Z logical operators
    for i in range(len(final_detector_list) - len(data_qubits_list), len(final_detector_list)):
        if final_detector_list[i] in Alice_Logical_Operator:
            target_rec.append(stim.target_rec(final_detector_shift_list[i]))
    # append Bob's Z logical operators
    for i in range(len(final_detector_list) - len(data_qubits_list), len(final_detector_list)):
        if final_detector_list[i] in Bob_Logical_Operator:
            target_rec.append(stim.target_rec(final_detector_shift_list[i]))

    circuit.append(stim.CircuitInstruction('OBSERVABLE_INCLUDE', target_rec, [0]))
    return circuit

def EPR_LOCC_Protocal_Circuit(d, num_layer, p_local=0.001, p_trans=0.001, basis="Z"):
    """Fault Tolerant EPR State Preparation and Distribution Circuit of Rotated Surface Code [LOCC]

    (1) Initialization: Alice-Bob: √2/2 * (|00⟩ + |11⟩)^{⊗n}.

    (2) Send Bob's data qubits to Bob.

    (3) Repeat block for √2/2 * (|00⟩_L + |11⟩_L) memory separately.

    (4) Final measurement: Finally, Alice and Bob measure their data qubits and perform decoding to get the logical results.
        Actually, the decoding should be performed together by XOR Alice and Bob. One classical communication process is still needed.

    (5) Check fidelity: The decoding results should be consistent with the true EPR states.

    Args:
        d (INT): Code distance 
        num_layer (int): Measurement rounds during the memory process in repeat block
        p_local (float, optional): Physical error rate of local operations. Defaults to 0.001.
        p_trans (float, optional): Physical error rate of transmission. Defaults to 0.001.
        basis (str, optional): "Z" (default) reads the Z_L Z_L parity of the Bell pair with a noiseless M readout;
            "X" reads the X_L X_L parity with a noiseless MX readout. Everything before the final readout is identical.

    Returns:
        Stim.Circuit(): An object of Stim.Circuit()
    """

    circuit = stim.Circuit()

    data_qubit_per_row = d
    data_qubit_per_col = d

    data_qubits_list = []

    Bob_data_qubits = []
    Alice_data_qubits = []

    Alice_Z_Logical_Operator = []
    Bob_Z_Logical_Operator = []
    # [basis] X_L of each patch: row 0 (x = 1)
    Alice_X_Logical_Operator = []
    Bob_X_Logical_Operator = []
    assert basis in ("Z", "X")

    stabilizer_x_list = []
    stabilizer_z_list = []

    alice_stabilizer_x_list = []
    bob_stabilizer_x_list = []
    alice_stabilizer_z_list = []
    bob_stabilizer_z_list = []

    qubit_index_position = {}

    # append data qubits of Alice
    for col in range(data_qubit_per_row):
        index_shift = col
        for row in range(data_qubit_per_col):
            X = 2 * row + 1
            Y = 2 * col + 1
            index = 2 * (row + col * d) + 1 + index_shift
            circuit.append("QUBIT_COORDS", [index], [X, Y])
            data_qubits_list.append(index)
            Alice_data_qubits.append(index)
            if col == 0:
                Alice_Z_Logical_Operator.append(index)
            if row == 0:
                Alice_X_Logical_Operator.append(index)
            qubit_index_position[index] = (X, Y)

    # all the indices of Bob's qubits should be Alice_Qubit[i] + Bob_Shift
    Bob_Shift = 4 * d**2
    Bob_Y_Shift = 2 * d + 1
    # append data qubits of Bob
    for col in range(data_qubit_per_row):
        index_shift = col
        for row in range(data_qubit_per_col):
            X = 2 * row + 1
            Y = 2 * col + 1 + Bob_Y_Shift
            index = 2 * (row + col * d) + 1 + index_shift + Bob_Shift
            circuit.append("QUBIT_COORDS", [index], [X, Y])
            data_qubits_list.append(index)
            Bob_data_qubits.append(index)
            if col == 0:
                Bob_Z_Logical_Operator.append(index)
            if row == 0:
                Bob_X_Logical_Operator.append(index)
            qubit_index_position[index] = (X, Y)

    # append boundary z stabilizer ancilla qubit (Alice)
    for col in range(data_qubit_per_row):
        index_shift = col
        # up z boundary - data qubit - right up
        for row in [0]:
            if col % 2 == 1:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1
            x_z = X - 1
            y_z = Y - 1
            if y_z <= 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_z = index - 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            alice_stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)
        # down z boundary - data qubit - right down
        for row in [data_qubit_per_col - 1]:
            if col % 2 == 0:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1
            x_z = X + 1
            y_z = Y - 1
            if y_z <= 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_z = index + 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            alice_stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)

    # append boundary x stabilizer ancilla qubit (Alice)
    for row in range(data_qubit_per_col):
        # right x boundary - data qubit - right up
        for col in [0]:
            index_shift = col
            if row % 2 == 1:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1
            x_x = X + 1
            y_x = Y - 1
            if x_x >= data_qubit_per_col * 2:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_x = index + 1
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            alice_stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)
        # left x boundary - data qubit - right down
        for col in [data_qubit_per_row - 1]:
            index_shift = col
            if row % 2 == 0:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1
            x_x = X + 1
            y_x = Y + 1
            index = 2 * (row + col * d) + 1 + index_shift
            index_x = index + 1 + (2 * d + 1)
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            alice_stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)

    # append boundary z stabilizer ancilla qubit (Bob)
    for col in range(data_qubit_per_row):
        index_shift = col
        # up z boundary - data qubit - right up
        for row in [0]:
            if col % 2 == 1:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1 + Bob_Y_Shift
            x_z = X - 1
            y_z = Y - 1
            if y_z - Bob_Y_Shift <= 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift + Bob_Shift
            index_z = index - 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            bob_stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)
        # down z boundary - data qubit - right down
        for row in [data_qubit_per_col - 1]:
            if col % 2 == 0:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1 + Bob_Y_Shift
            x_z = X + 1
            y_z = Y - 1
            if y_z - Bob_Y_Shift <= 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift + Bob_Shift
            index_z = index + 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            bob_stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)

    # append boundary x stabilizer ancilla qubit (Bob)
    for row in range(data_qubit_per_col):
        # right x boundary - data qubit - right up
        for col in [0]:
            index_shift = col
            if row % 2 == 1:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1 + Bob_Y_Shift
            x_x = X + 1
            y_x = Y - 1
            if x_x >= data_qubit_per_col * 2:
                continue
            index = 2 * (row + col * d) + 1 + index_shift + Bob_Shift
            index_x = index + 1
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            bob_stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)
        # left x boundary - data qubit - right down
        for col in [data_qubit_per_row - 1]:
            index_shift = col
            if row % 2 == 0:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1 + Bob_Y_Shift
            x_x = X + 1
            y_x = Y + 1
            if x_x >= data_qubit_per_col * 2:
                continue
            index = 2 * (row + col * d) + 1 + index_shift + Bob_Shift
            index_x = index + 1 + (2 * d + 1)
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            bob_stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)


    # append z stabilizer (general case of Alice)
    for row in range(data_qubit_per_col - 1):
        for col in range(1, data_qubit_per_row):
            index_shift = col
            X = 2 * row + 1
            Y = 2 * col + 1
            x_z = X + 1
            y_z = Y - 1
            if ((x_z + y_z) // 2) % 2 == 1:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_z = index + 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            alice_stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)

    # append x stabilizer (general case of Alice)
    for row in range(data_qubit_per_col - 1):
        for col in range(1, data_qubit_per_row):
            index_shift = col
            X = 2 * row + 1
            Y = 2 * col + 1
            x_x = X + 1
            y_x = Y - 1
            if ((x_x + y_x) // 2) % 2 == 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_x = index + 1
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            alice_stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)


    # append z stabilizer (general case of Bob)
    for row in range(data_qubit_per_col - 1):
        for col in range(1, data_qubit_per_row):
            index_shift = col
            X = 2 * row + 1
            Y = 2 * col + 1 + Bob_Y_Shift
            x_z = X + 1
            y_z = Y - 1
            if ((x_z + y_z) // 2) % 2 == 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift + Bob_Shift
            index_z = index + 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            bob_stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)

    # append x stabilizer (general case of Bob)
    for row in range(data_qubit_per_col - 1):
        for col in range(1, data_qubit_per_row):
            index_shift = col
            X = 2 * row + 1
            Y = 2 * col + 1 + Bob_Y_Shift
            x_x = X + 1
            y_x = Y - 1
            if ((x_x + y_x) // 2) % 2 == 1:
                continue
            index = 2 * (row + col * d) + 1 + index_shift + Bob_Shift
            index_x = index + 1
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            bob_stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)

    # print("Bob stabilizer x list:", bob_stabilizer_x_list)
    # print("Bob stabilizer z list:", bob_stabilizer_z_list)

    # (1) State preparation of Alice and Bob √2/2 * (|00⟩ + |11⟩)^{⊗n}
    qubits_all_list = data_qubits_list + stabilizer_x_list + stabilizer_z_list
    stabilizer_list = stabilizer_x_list + stabilizer_z_list

    circuit.append("R", qubits_all_list)
    circuit.append('X_ERROR', qubits_all_list, p_local)
    circuit.append("H", Alice_data_qubits)
    circuit.append("TICK")

    EPR_CX = []
    for qubit_a in Alice_data_qubits:
        qubit_b = qubit_a + Bob_Shift
        EPR_CX.extend([qubit_a, qubit_b])
    circuit.append('CX', EPR_CX)
    circuit.append('DEPOLARIZE2', EPR_CX, p_local)


    # (2) Send Bob's data qubits to Bob
    circuit.append('DEPOLARIZE1', Bob_data_qubits, p_trans)
    # Alice's data qubits: idling
    circuit.append('DEPOLARIZE1', Alice_data_qubits, p_local)

    def syndrome_extraction_initialization(circuit):
        """Simulate the initialization circuit of syndrome extraction in the Surface Code

        Args:
            circuit (Stim.Circuit): Origin circuit

        Returns:
            circuit (Stim.Circuit): Circuit with syndrome extraction appended
        """
        circuit.append("H", stabilizer_x_list)
        circuit.append("TICK")
        # 1 - Syndrome Extraction
        CX_list1 = []
        for i in stabilizer_x_list:
            if i + 1 in data_qubits_list:
                CX_list1.extend([i, i + 1])
        for i in data_qubits_list:
            if i - 1 in stabilizer_z_list:
                CX_list1.extend([i, i - 1])

        idle1 = list(set(qubits_all_list) - set(CX_list1))

        circuit.append('CX', CX_list1)
        circuit.append('DEPOLARIZE2', CX_list1, p_local)
        circuit.append('DEPOLARIZE1', idle1, p_local)
        circuit.append("TICK")

        # 2 - Syndrome Extraction
        CX_list2 = []
        for i in stabilizer_x_list:
            if i - 1 in data_qubits_list:
                CX_list2.extend([i, i - 1])
        for i in data_qubits_list:
            if i + 2 * d in stabilizer_z_list:
                CX_list2.extend([i, i + 2 * d])

        idle2 = list(set(qubits_all_list) - set(CX_list2))

        circuit.append('CX', CX_list2)
        circuit.append('DEPOLARIZE2', CX_list2, p_local)
        circuit.append('DEPOLARIZE1', idle2, p_local)
        circuit.append("TICK")

        # 3 - Syndrome Extraction
        CX_list3 = []
        for i in stabilizer_x_list:
            if i - 2 * d in data_qubits_list:
                CX_list3.extend([i, i - 2 * d])
        for i in data_qubits_list:
            if i + 1 in stabilizer_z_list:
                CX_list3.extend([i, i + 1])

        idle3 = list(set(qubits_all_list) - set(CX_list3))

        circuit.append('CX', CX_list3)
        circuit.append('DEPOLARIZE2', CX_list3, p_local)
        circuit.append('DEPOLARIZE1', idle3, p_local)
        circuit.append("TICK")

        # 4 - Syndrome Extraction
        CX_list4 = []
        for i in stabilizer_x_list:
            if i - (2 * d + 2) in data_qubits_list:
                CX_list4.extend([i, i - (2 * d + 2)])
        for i in data_qubits_list:
            if i + (2 * d + 2) in stabilizer_z_list:
                CX_list4.extend([i, i + (2 * d + 2)])

        idle4 = list(set(qubits_all_list) - set(CX_list4))

        circuit.append('CX', CX_list4)
        circuit.append('DEPOLARIZE2', CX_list4, p_local)
        circuit.append('DEPOLARIZE1', idle4, p_local)
        circuit.append("TICK")

        circuit.append("H", stabilizer_x_list)
        circuit.append("TICK")

        circuit.append('X_ERROR', stabilizer_list, p_local)
        circuit.append('MR', stabilizer_list)
        circuit.append('X_ERROR', stabilizer_list, p_local)

        return circuit

    circuit = syndrome_extraction_initialization(circuit)
    detector_list = []
    detector_list += stabilizer_list
    detector_shift_list = [(-len(detector_list) + i) for i in range(len(detector_list))]

    # First detector round
    detector_shift = 0
    for index in stabilizer_list[::-1]:
        detector_shift -= 1
        if index in (bob_stabilizer_x_list + bob_stabilizer_z_list):
            continue
        qubit_a = index
        qubit_b = index + Bob_Shift
        X, Y = qubit_index_position[index]
        circuit.append("DETECTOR", [stim.target_rec(detector_shift), stim.target_rec(detector_shift_list[detector_list.index(qubit_b)])], [X, Y, 0])

    # (3) Add repeat block for √2/2 * (|00⟩_L + |11⟩_L) memory
    repeated_block = stim.Circuit()
    repeated_block.append('DEPOLARIZE1', data_qubits_list, p_local)
    repeated_block.append("TICK")
    repeated_block = syndrome_extraction_initialization(repeated_block)
    repeated_block.append("SHIFT_COORDS", [], [0, 0, 1])

    # All x detectors and z detectors with detector[t] xor detector[t-1]
    detector_shift = 0
    for index in stabilizer_list[::-1]:
        detector_shift -= 1
        if index in (bob_stabilizer_x_list + bob_stabilizer_z_list):
            continue
        qubit_a = index
        qubit_b = index + Bob_Shift
        X, Y = qubit_index_position[index]
        target_rec = []
        target_rec.append(stim.target_rec(detector_shift))
        target_rec.append(stim.target_rec(detector_shift_list[detector_list.index(qubit_b)]))
        target_rec.append(stim.target_rec(detector_shift - len(stabilizer_list)))
        target_rec.append(stim.target_rec(detector_shift_list[detector_list.index(qubit_b)] - len(stabilizer_list)))
        repeated_block.append("DETECTOR", target_rec, [X, Y, 0])

    if num_layer > 0:
        circuit.append(stim.CircuitRepeatBlock(num_layer, repeated_block))


    # (4) Final measurement
    # circuit.append('X_ERROR', data_qubits_list, p_local)
    # [basis] noiseless readout of all data qubits in the Z (M) or X (MX) basis
    circuit.append('M' if basis == "Z" else 'MX', data_qubits_list)

    final_detector_list = []
    final_detector_list += stabilizer_list
    final_detector_list += data_qubits_list
    final_detector_shift_list = [(-len(final_detector_list) + i) for i in range(len(final_detector_list))]

    final_detector_set = set(final_detector_list)
    final_data_qubits_set = set(data_qubits_list)
    # [basis] readout detectors from the Z-type checks (M) or the X-type checks (MX); the support offsets
    # (-1, +1, -2d-2, -2d) are the CX partners of both ancilla types in syndrome_extraction_initialization
    final_stabilizer_z_set = set(stabilizer_z_list) if basis == "Z" else set(stabilizer_x_list)

    # perform "perfect" measurement of z stabilizers on Alice's and Bob's patch
    for i, detector in enumerate(final_detector_list):
        if detector in final_stabilizer_z_set:
            # only the last rounds of stabilizer z measurement results are used for perfect measurement
            if (len(final_detector_list) - i) > len(qubits_all_list):
                continue
            X, Y = qubit_index_position[detector]
            target_rec = []
            target_rec.append(stim.target_rec(final_detector_shift_list[i]))
            # 1 stabilizer - "4" data qubits
            possible_neighbors = [
                detector - 1,
                detector + 1,
                detector - 2 * d - 2,
                detector - 2 * d
            ]
            for neighbor in possible_neighbors:
                if neighbor in final_detector_set and neighbor in final_data_qubits_set:
                    target_rec.append(stim.target_rec(final_detector_shift_list[final_detector_list.index(neighbor)]))
            circuit.append("DETECTOR", target_rec, [X, Y, 1])

    # (5) Check fidelity, logical observables

    # [basis] observable: Z_L(A) Z_L(B) (column 0 of each patch) or X_L(A) X_L(B) (row 0 of each patch)
    Alice_Logical_Operator = Alice_Z_Logical_Operator if basis == "Z" else Alice_X_Logical_Operator
    Bob_Logical_Operator = Bob_Z_Logical_Operator if basis == "Z" else Bob_X_Logical_Operator
    target_rec = []
    # append Alice's Z logical operators
    for i in range(len(final_detector_list) - len(data_qubits_list), len(final_detector_list)):
        if final_detector_list[i] in Alice_Logical_Operator:
            target_rec.append(stim.target_rec(final_detector_shift_list[i]))
    # append Bob's Z logical operators
    for i in range(len(final_detector_list) - len(data_qubits_list), len(final_detector_list)):
        if final_detector_list[i] in Bob_Logical_Operator:
            target_rec.append(stim.target_rec(final_detector_shift_list[i]))
    circuit.append(stim.CircuitInstruction('OBSERVABLE_INCLUDE', target_rec, [0]))
    return circuit

def Direct_QECC_Circuit(d, num_layer, p_local = 0.001, p_trans = 0.001, basis = "Z"):
    """Direct transmission of one encoded patch.
    basis="Z" (default): |0>_L, noiseless M readout, observable Z_L (column d-1).
    basis="X": |+>_L (data RX + Z_ERROR instead of R + X_ERROR; ancillas unchanged), X-type first-layer
    detectors, noiseless MX readout with X-type readout detectors, observable X_L (row 0).
    Layout, CX schedule and all other noise are identical."""
    assert basis in ("Z", "X")
    circuit = stim.Circuit()
    
    data_qubit_per_row = d
    data_qubit_per_col = d
    

    data_qubits_list = []
    stabilizer_x_list = []
    stabilizer_z_list = []
    qubit_index_position = {}

    # append data qubit
    for col in range(data_qubit_per_row):
        index_shift = col
        for row in range(data_qubit_per_col):
            X = 2 * row + 1
            Y = 2 * col + 1
            index = 2 * (row + col * d) + 1 + index_shift
            circuit.append("QUBIT_COORDS", [index], [X, Y])
            data_qubits_list.append(index)
            qubit_index_position[index] = (X, Y)

    # append boundary z stabilizer ancilla qubit
    for col in range(data_qubit_per_row):
        index_shift = col
        # up z boundary - data qubit - right up 
        for row in [0]:
            if col % 2 == 1:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1
            x_z = X - 1
            y_z = Y - 1
            if y_z <= 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_z = index - 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)
        # down z boundary - data qubit - right down
        for row in [data_qubit_per_col - 1]:
            if col % 2 == 0:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1
            x_z = X + 1
            y_z = Y - 1
            if y_z <= 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_z = index + 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)

    # append boundary x stabilizer ancilla qubit
    for row in range(data_qubit_per_col):
        # right x boundary - data qubit - right up 
        for col in [0]:
            index_shift = col
            if row % 2 == 1:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1
            x_x = X + 1
            y_x = Y - 1
            if x_x >= data_qubit_per_col * 2:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_x = index + 1
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)
        # left x boundary - data qubit - right down
        for col in [data_qubit_per_row - 1]:
            index_shift = col
            if row % 2 == 0:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1
            x_x = X + 1
            y_x = Y + 1
            index = 2 * (row + col * d) + 1 + index_shift
            index_x = index + 1 + (2 * d + 1)
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)

    # append z stabilizer (general case)
    for row in range(data_qubit_per_col - 1):
        for col in range(1, data_qubit_per_row):
            index_shift = col
            X = 2 * row + 1
            Y = 2 * col + 1
            x_z = X + 1
            y_z = Y - 1
            if ((x_z + y_z) // 2) % 2 == 1:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_z = index + 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)

    # append x stabilizer (general case)
    for row in range(data_qubit_per_col - 1):
        for col in range(1, data_qubit_per_row):
            index_shift = col
            X = 2 * row + 1
            Y = 2 * col + 1
            x_x = X + 1
            y_x = Y - 1
            if ((x_x + y_x) // 2) % 2 == 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_x = index + 1
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)

    # circuit.append("R", [index_x], [x_x, y_x])
    # print the circuit


    qubits_all_list = data_qubits_list + stabilizer_x_list + stabilizer_z_list
    stabilizer_list = stabilizer_x_list + stabilizer_z_list
    
    if basis == "Z":
        circuit.append("R", qubits_all_list)
        circuit.append('X_ERROR', qubits_all_list, p_local)
    else:
        # [basis] data in |+> (RX followed by Z_ERROR(p) equals R, X_ERROR(p), noiseless H); ancillas unchanged
        circuit.append("RX", data_qubits_list)
        circuit.append("R", stabilizer_list)
        circuit.append('Z_ERROR', data_qubits_list, p_local)
        circuit.append('X_ERROR', stabilizer_list, p_local)
    circuit.append("TICK")

    def syndrome_extraction(circuit):
        circuit.append("H", stabilizer_x_list)
        circuit.append("TICK")
        # 1 - Syndrome Extraction
        CX_list1 = [] 
        for i in stabilizer_x_list:
            if i + 1 in data_qubits_list:
                CX_list1.extend([i, i + 1])
        for i in data_qubits_list:
            if i - 1 in stabilizer_z_list:
                CX_list1.extend([i, i - 1])
                
        single1 = list(set(qubits_all_list) - set(CX_list1))

        circuit.append('CX', CX_list1)
        circuit.append('DEPOLARIZE2', CX_list1, p_local)
        circuit.append('DEPOLARIZE1', single1, p_local)
        circuit.append("TICK")

        # 2 - Syndrome Extraction
        CX_list2 = [] 
        for i in stabilizer_x_list:
            if i - 1 in data_qubits_list:
                CX_list2.extend([i, i - 1])
        for i in data_qubits_list:
            if i + 2 * d in stabilizer_z_list:
                CX_list2.extend([i, i + 2 * d])
                
        single2 = list(set(qubits_all_list) - set(CX_list2))

        circuit.append('CX', CX_list2)
        circuit.append('DEPOLARIZE2', CX_list2, p_local)
        circuit.append('DEPOLARIZE1', single2, p_local)
        circuit.append("TICK")

        # 3 - Syndrome Extraction
        CX_list3 = [] 
        for i in stabilizer_x_list:
            if i - 2 * d in data_qubits_list:
                CX_list3.extend([i, i - 2 * d])
        for i in data_qubits_list:
            if i + 1 in stabilizer_z_list:
                CX_list3.extend([i, i + 1])
                
        single3 = list(set(qubits_all_list) - set(CX_list3))

        circuit.append('CX', CX_list3)
        circuit.append('DEPOLARIZE2', CX_list3, p_local)
        circuit.append('DEPOLARIZE1', single3, p_local)
        circuit.append("TICK")

        # 4 - Syndrome Extraction
        CX_list4 = [] 
        for i in stabilizer_x_list:
            if i - (2 * d + 2) in data_qubits_list:
                CX_list4.extend([i, i - (2 * d + 2)])
        for i in data_qubits_list:
            if i + (2 * d + 2) in stabilizer_z_list:
                CX_list4.extend([i, i + (2 * d + 2)])
                
        single4 = list(set(qubits_all_list) - set(CX_list4))

        circuit.append('CX', CX_list4)
        circuit.append('DEPOLARIZE2', CX_list4, p_local)
        circuit.append('DEPOLARIZE1', single4, p_local)
        circuit.append("TICK")

        circuit.append("H", stabilizer_x_list)
        circuit.append("TICK")
        
        circuit.append('X_ERROR', stabilizer_list, p_local)
        circuit.append('MR', stabilizer_list)
        circuit.append('X_ERROR', stabilizer_list, p_local)
        
        return circuit

    circuit = syndrome_extraction(circuit)


    # first layer : only z detectors, no x detectors ([basis] X: only x detectors)
    first_layer_list = stabilizer_z_list if basis == "Z" else stabilizer_x_list
    detector_shift = 0
    for i in stabilizer_list[::-1]:
        detector_shift -= 1
        if i in first_layer_list:
            X, Y = qubit_index_position[i]
            circuit.append("DETECTOR",[stim.target_rec(detector_shift)], [X, Y, 0])

    # Add transmission error once
    circuit.append('DEPOLARIZE1', data_qubits_list, p_trans)
    
    # Add repeat block
    repeated_block = stim.Circuit()
    # repeated_block.append('DEPOLARIZE1', data_qubits_list + data_qubits_list, p_local)
    repeated_block.append("TICK")
    repeated_block = syndrome_extraction(repeated_block)
    repeated_block.append("SHIFT_COORDS", [], [0, 0, 1])

    # mid layer : all x detectors and z detectors
    detector_shift = 0
    for i in stabilizer_list[::-1]:
        detector_shift -= 1
        X, Y = qubit_index_position[i]
        repeated_block.append("DETECTOR",[stim.target_rec(detector_shift), stim.target_rec(detector_shift - len(stabilizer_list))], [X, Y, 0])

    circuit.append(stim.CircuitRepeatBlock(num_layer, repeated_block))

    # measure all data qubits
    # circuit.append('X_ERROR', data_qubits_list, p_local)

    # [basis] noiseless readout of all data qubits in the Z (M) or X (MX) basis
    circuit.append('M' if basis == "Z" else 'MX', data_qubits_list)


    detector_list = []
    detector_list += stabilizer_list
    detector_list += data_qubits_list
    detector_shift_list = [(-len(detector_list) + i) for i in range(len(detector_list))]

    # for memory circuit, perform "perfect" measurement of z stabilizers
    # [basis] X: X-type checks; their data support has the same offsets (-1, +1, -2d-2, -2d)
    readout_check_list = stabilizer_z_list if basis == "Z" else stabilizer_x_list
    for i in range(len(detector_list)):
        if detector_list[i] in readout_check_list:   
            target_rec = []
            target_rec.append(stim.target_rec(detector_shift_list[i]))
            # 1 stabilizer - "4" data qubits
            for j in range(len(detector_list)):
                if detector_list[j] not in data_qubits_list:
                    continue
                if detector_list[j] == detector_list[i] - 1:
                    target_rec.append(stim.target_rec(detector_shift_list[j]))
                if detector_list[j] == detector_list[i] + 1:
                    target_rec.append(stim.target_rec(detector_shift_list[j]))
                if detector_list[j] == detector_list[i] - 2 * d - 2:
                    target_rec.append(stim.target_rec(detector_shift_list[j]))
                if detector_list[j] == detector_list[i] - 2 * d:
                    target_rec.append(stim.target_rec(detector_shift_list[j]))
            
            X, Y = qubit_index_position[detector_list[i]]
            circuit.append("DETECTOR",target_rec, [X, Y, 1])
                
        
    if basis == "Z":
        circuit.append(stim.CircuitInstruction('OBSERVABLE_INCLUDE', [stim.target_rec(-j - 1) for j in range(d)], [0]))
    else:
        # [basis] X_L on row 0: data (row 0, col c) is at position c * d of data_qubits_list (column-major)
        n_data = len(data_qubits_list)
        circuit.append(stim.CircuitInstruction('OBSERVABLE_INCLUDE', [stim.target_rec(-(n_data - c * d)) for c in range(d)], [0]))
    # print(circuit)
    # circuit.to_file(f"Surface_SE_{num_layer}_{p_local}.stim")
    return circuit

def EE_Surface_Transversal_CNOT_Teleportation(d, p_local=0.001, p_trans=0.001, basis="Z"):
    """Fault Tolerant EPR State Preparation and Distribution Circuit of Rotated Surface Code

    (1) Initialization: Alice, Bob: |+⟩L, |0⟩L.

    (2) Transversal CNOT between Alice and Bob through teleportation

    (3) Measure all stabilizers for error correction.

    (4) Final measurement: Finally, Alice and Bob measure their data qubits and perform decoding to get the logical results.
        Actually, the decoding should be performed together by XOR Alice and Bob. One classical communication process is still needed.

    (5) Check fidelity: The decoding results should be consistent with the true EPR states.

    Args:
        d (INT): Code distance 
        num_layer (int): Measurement rounds during the intermediate memory process
        p_local (float, optional): Physical error rate of local operations. Defaults to 0.001.
        p_trans (float, optional): Physical error rate of transmission. Defaults to 0.001.
        basis (str, optional): "Z" (default) reads the Z_L Z_L parity of the Bell pair with a noiseless M readout;
            "X" reads the X_L X_L parity with a noiseless MX readout. In the X basis the round after the
            teleported CNOT has X-type detectors X(A) X(B) with the Pauli frame of the EPR_2 (X-basis) outcomes,
            the mirror of the Z-type detectors Z(A) Z(B) with the EPR_1 frame used in the Z basis.

    Returns:
        Stim.Circuit(): An object of Stim.Circuit()
    """

    circuit = stim.Circuit()

    data_qubit_per_row = d
    data_qubit_per_col = d

    data_qubits_list = []

    Bob_data_qubits = []
    Alice_data_qubits = []

    Alice_Z_Logical_Operator = []
    Bob_Z_Logical_Operator = []
    # [basis] X_L of each patch: row 0 (x = 1)
    Alice_X_Logical_Operator = []
    Bob_X_Logical_Operator = []
    assert basis in ("Z", "X")

    stabilizer_x_list = []
    stabilizer_z_list = []

    alice_stabilizer_x_list = []
    bob_stabilizer_x_list = []
    alice_stabilizer_z_list = []
    bob_stabilizer_z_list = []

    qubit_index_position = {}

    # append data qubits of Alice
    for col in range(data_qubit_per_row):
        index_shift = col
        for row in range(data_qubit_per_col):
            X = 2 * row + 1
            Y = 2 * col + 1
            index = 2 * (row + col * d) + 1 + index_shift
            circuit.append("QUBIT_COORDS", [index], [X, Y])
            data_qubits_list.append(index)
            Alice_data_qubits.append(index)
            if col == 0:
                Alice_Z_Logical_Operator.append(index)
            if row == 0:
                Alice_X_Logical_Operator.append(index)
            qubit_index_position[index] = (X, Y)

    # all the indices of Bob's qubits should be Alice_Qubit[i] + Bob_Shift
    Bob_Shift = 4 * d**2
    Bob_Y_Shift = 2 * d + 1
    # append data qubits of Bob
    for col in range(data_qubit_per_row):
        index_shift = col
        for row in range(data_qubit_per_col):
            X = 2 * row + 1
            Y = 2 * col + 1 + Bob_Y_Shift
            index = 2 * (row + col * d) + 1 + index_shift + Bob_Shift
            circuit.append("QUBIT_COORDS", [index], [X, Y])
            data_qubits_list.append(index)
            Bob_data_qubits.append(index)
            if col == 0:
                Bob_Z_Logical_Operator.append(index)
            if row == 0:
                Bob_X_Logical_Operator.append(index)
            qubit_index_position[index] = (X, Y)

    # append boundary z stabilizer ancilla qubit (Alice)
    for col in range(data_qubit_per_row):
        index_shift = col
        # up z boundary - data qubit - right up
        for row in [0]:
            if col % 2 == 1:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1
            x_z = X - 1
            y_z = Y - 1
            if y_z <= 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_z = index - 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            alice_stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)
        # down z boundary - data qubit - right down
        for row in [data_qubit_per_col - 1]:
            if col % 2 == 0:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1
            x_z = X + 1
            y_z = Y - 1
            if y_z <= 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_z = index + 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            alice_stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)

    # append boundary x stabilizer ancilla qubit (Alice)
    for row in range(data_qubit_per_col):
        # right x boundary - data qubit - right up
        for col in [0]:
            index_shift = col
            if row % 2 == 1:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1
            x_x = X + 1
            y_x = Y - 1
            if x_x >= data_qubit_per_col * 2:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_x = index + 1
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            alice_stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)
        # left x boundary - data qubit - right down
        for col in [data_qubit_per_row - 1]:
            index_shift = col
            if row % 2 == 0:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1
            x_x = X + 1
            y_x = Y + 1
            index = 2 * (row + col * d) + 1 + index_shift
            index_x = index + 1 + (2 * d + 1)
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            alice_stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)

    # append boundary z stabilizer ancilla qubit (Bob)
    for col in range(data_qubit_per_row):
        index_shift = col
        # up z boundary - data qubit - right up
        for row in [0]:
            if col % 2 == 1:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1 + Bob_Y_Shift
            x_z = X - 1
            y_z = Y - 1
            if y_z - Bob_Y_Shift <= 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift + Bob_Shift
            index_z = index - 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            bob_stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)
        # down z boundary - data qubit - right down
        for row in [data_qubit_per_col - 1]:
            if col % 2 == 0:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1 + Bob_Y_Shift
            x_z = X + 1
            y_z = Y - 1
            if y_z - Bob_Y_Shift <= 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift + Bob_Shift
            index_z = index + 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            bob_stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)

    # append boundary x stabilizer ancilla qubit (Bob)
    for row in range(data_qubit_per_col):
        # right x boundary - data qubit - right up
        for col in [0]:
            index_shift = col
            if row % 2 == 1:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1 + Bob_Y_Shift
            x_x = X + 1
            y_x = Y - 1
            if x_x >= data_qubit_per_col * 2:
                continue
            index = 2 * (row + col * d) + 1 + index_shift + Bob_Shift
            index_x = index + 1
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            bob_stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)
        # left x boundary - data qubit - right down
        for col in [data_qubit_per_row - 1]:
            index_shift = col
            if row % 2 == 0:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1 + Bob_Y_Shift
            x_x = X + 1
            y_x = Y + 1
            if x_x >= data_qubit_per_col * 2:
                continue
            index = 2 * (row + col * d) + 1 + index_shift + Bob_Shift
            index_x = index + 1 + (2 * d + 1)
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            bob_stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)

    # append z stabilizer (general case of Alice)
    for row in range(data_qubit_per_col - 1):
        for col in range(1, data_qubit_per_row):
            index_shift = col
            X = 2 * row + 1
            Y = 2 * col + 1
            x_z = X + 1
            y_z = Y - 1
            if ((x_z + y_z) // 2) % 2 == 1:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_z = index + 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            alice_stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)

    # append x stabilizer (general case of Alice)
    for row in range(data_qubit_per_col - 1):
        for col in range(1, data_qubit_per_row):
            index_shift = col
            X = 2 * row + 1
            Y = 2 * col + 1
            x_x = X + 1
            y_x = Y - 1
            if ((x_x + y_x) // 2) % 2 == 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_x = index + 1
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            alice_stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)


    # append z stabilizer (general case of Bob)
    for row in range(data_qubit_per_col - 1):
        for col in range(1, data_qubit_per_row):
            index_shift = col
            X = 2 * row + 1
            Y = 2 * col + 1 + Bob_Y_Shift
            x_z = X + 1
            y_z = Y - 1
            if ((x_z + y_z) // 2) % 2 == 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift + Bob_Shift
            index_z = index + 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            bob_stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)

    # append x stabilizer (general case of Bob)
    for row in range(data_qubit_per_col - 1):
        for col in range(1, data_qubit_per_row):
            index_shift = col
            X = 2 * row + 1
            Y = 2 * col + 1 + Bob_Y_Shift
            x_x = X + 1
            y_x = Y - 1
            if ((x_x + y_x) // 2) % 2 == 1:
                continue
            index = 2 * (row + col * d) + 1 + index_shift + Bob_Shift
            index_x = index + 1
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            bob_stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)

    # print("Bob stabilizer x list:", bob_stabilizer_x_list)
    # print("Bob stabilizer z list:", bob_stabilizer_z_list)

    # (1) State preparation of Alice and Bob: |+⟩L and |0⟩L
    qubits_all_list = data_qubits_list + stabilizer_x_list + stabilizer_z_list
    stabilizer_list = stabilizer_x_list + stabilizer_z_list
    
    # EPR pairs for gate teleportation
    EPR_Shift = 1000
    EPR_1 = [i + EPR_Shift for i in Alice_data_qubits]
    EPR_2 = [i + EPR_Shift for i in Bob_data_qubits]

    circuit.append("R", qubits_all_list)
    circuit.append('X_ERROR', qubits_all_list, p_local)
    circuit.append("H", Alice_data_qubits)
    circuit.append("TICK")
    
    # Prepare EPR pairs for gate teleportation
    circuit.append("R", EPR_1 + EPR_2)
    circuit.append("H", EPR_1)
    
    for i in range(len(EPR_1)):
        circuit.append("CX", [EPR_1[i], EPR_2[i]])

    for i in range(len(EPR_1)):
        circuit.append("DEPOLARIZE2", [EPR_1[i], EPR_2[i]], p_local)
    
    # transmission
    circuit.append("DEPOLARIZE1", EPR_2, p_trans)
    
    def syndrome_extraction_initialization(circuit):
        """Simulate the initialization circuit of syndrome extraction in the Surface Code

        Args:
            circuit (Stim.Circuit): Origin circuit

        Returns:
            circuit (Stim.Circuit): Circuit with syndrome extraction appended
        """
        circuit.append("H", stabilizer_x_list)
        circuit.append("TICK")
        # 1 - Syndrome Extraction
        CX_list1 = []
        for i in stabilizer_x_list:
            if i + 1 in data_qubits_list:
                CX_list1.extend([i, i + 1])
        for i in data_qubits_list:
            if i - 1 in stabilizer_z_list:
                CX_list1.extend([i, i - 1])

        idle1 = list(set(qubits_all_list) - set(CX_list1))

        circuit.append('CX', CX_list1)
        circuit.append('DEPOLARIZE2', CX_list1, p_local)
        circuit.append('DEPOLARIZE1', idle1, p_local)
        circuit.append("TICK")

        # 2 - Syndrome Extraction
        CX_list2 = []
        for i in stabilizer_x_list:
            if i - 1 in data_qubits_list:
                CX_list2.extend([i, i - 1])
        for i in data_qubits_list:
            if i + 2 * d in stabilizer_z_list:
                CX_list2.extend([i, i + 2 * d])

        idle2 = list(set(qubits_all_list) - set(CX_list2))

        circuit.append('CX', CX_list2)
        circuit.append('DEPOLARIZE2', CX_list2, p_local)
        circuit.append('DEPOLARIZE1', idle2, p_local)
        circuit.append("TICK")

        # 3 - Syndrome Extraction
        CX_list3 = []
        for i in stabilizer_x_list:
            if i - 2 * d in data_qubits_list:
                CX_list3.extend([i, i - 2 * d])
        for i in data_qubits_list:
            if i + 1 in stabilizer_z_list:
                CX_list3.extend([i, i + 1])

        idle3 = list(set(qubits_all_list) - set(CX_list3))

        circuit.append('CX', CX_list3)
        circuit.append('DEPOLARIZE2', CX_list3, p_local)
        circuit.append('DEPOLARIZE1', idle3, p_local)
        circuit.append("TICK")

        # 4 - Syndrome Extraction
        CX_list4 = []
        for i in stabilizer_x_list:
            if i - (2 * d + 2) in data_qubits_list:
                CX_list4.extend([i, i - (2 * d + 2)])
        for i in data_qubits_list:
            if i + (2 * d + 2) in stabilizer_z_list:
                CX_list4.extend([i, i + (2 * d + 2)])

        idle4 = list(set(qubits_all_list) - set(CX_list4))

        circuit.append('CX', CX_list4)
        circuit.append('DEPOLARIZE2', CX_list4, p_local)
        circuit.append('DEPOLARIZE1', idle4, p_local)
        circuit.append("TICK")

        circuit.append("H", stabilizer_x_list)
        circuit.append("TICK")

        circuit.append('X_ERROR', stabilizer_list, p_local)
        circuit.append('MR', stabilizer_list)
        circuit.append('X_ERROR', stabilizer_list, p_local)

        return circuit

    # stabilizer encoding circuit: actually no need
    circuit = syndrome_extraction_initialization(circuit)
    detector_list = []
    detector_list += stabilizer_list
    detector_shift_list = [(-len(detector_list) + i) for i in range(len(detector_list))]
    
    # First detector round: Initialize the |+⟩L and |0⟩L.
    detector_shift = 0
    for index in stabilizer_list[::-1]:
        detector_shift -= 1
        if index in alice_stabilizer_z_list + bob_stabilizer_x_list:
            continue
        X, Y = qubit_index_position[index]
        circuit.append("DETECTOR", [stim.target_rec(detector_shift)], [X, Y, 0])
    circuit.append("SHIFT_COORDS", [], [0, 0, 1])
    
    # (2) Transversal CNOT Gate Teleportation
    '''Transversal CNOT Gates Between Alice and Bob by Gate Teleportation'''
    # Transversal_CX = []
    # for qubit_a in Alice_data_qubits:
    #     qubit_b = qubit_a + Bob_Shift
    #     Transversal_CX.extend([qubit_a, qubit_b])
    # circuit.append('CX', Transversal_CX)
    # circuit.append('DEPOLARIZE2', Transversal_CX, p_local)
    
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
    

    # New stabilizer measurement
    circuit.append('DEPOLARIZE1', data_qubits_list, p_local)
    circuit.append("TICK")
    circuit = syndrome_extraction_initialization(circuit)
    
    # circuit.append('DEPOLARIZE1', data_qubits_list, p_local)
    # circuit.append("TICK")
    # circuit = syndrome_extraction_initialization(circuit)
    # circuit.append("SHIFT_COORDS", [], [0, 0, 1])
    
    detector_list = []
    detector_list += (EPR_1 + EPR_2)
    detector_list += stabilizer_list
    detector_shift_list = [(-len(detector_list) + i) for i in range(len(detector_list))]
    
    
    if basis == "Z":
        for idx, alice_ancilla in enumerate(alice_stabilizer_z_list):
            target_rec = []
            X, Y = qubit_index_position[alice_ancilla]
            qubit_a = alice_ancilla
            qubit_b = alice_ancilla + Bob_Shift
            possible_neighbors = [
                qubit_b - 1,
                qubit_b + 1,
                qubit_b - 2 * d - 2,
                qubit_b - 2 * d
            ]
            target_rec.append(stim.target_rec(detector_shift_list[detector_list.index(qubit_a)]))
            target_rec.append(stim.target_rec(detector_shift_list[detector_list.index(qubit_b)]))
            for neighbor in possible_neighbors:
                if neighbor in Bob_data_qubits:
                    bob_data_qubit = neighbor
                    alice_data_qubit = bob_data_qubit - Bob_Shift
                    epr1_qubit = alice_data_qubit + EPR_Shift
                    target_rec.append(stim.target_rec(detector_shift_list[detector_list.index(epr1_qubit)]))

            circuit.append("DETECTOR", target_rec, [X, Y, 0])
    else:
        # [basis] mirror: after the CNOT, X(A) X(B) of each X-type check equals the X(A) check of |+>_L (= +1),
        # up to the Z frame Z^{m2} on Alice's data (m2 = EPR_2 X-basis outcomes).
        for idx, alice_ancilla in enumerate(alice_stabilizer_x_list):
            target_rec = []
            X, Y = qubit_index_position[alice_ancilla]
            qubit_a = alice_ancilla
            qubit_b = alice_ancilla + Bob_Shift
            possible_neighbors = [
                qubit_a - 1,
                qubit_a + 1,
                qubit_a - 2 * d - 2,
                qubit_a - 2 * d
            ]
            target_rec.append(stim.target_rec(detector_shift_list[detector_list.index(qubit_a)]))
            target_rec.append(stim.target_rec(detector_shift_list[detector_list.index(qubit_b)]))
            for neighbor in possible_neighbors:
                if neighbor in Alice_data_qubits:
                    bob_data_qubit = neighbor + Bob_Shift
                    epr2_qubit = bob_data_qubit + EPR_Shift
                    target_rec.append(stim.target_rec(detector_shift_list[detector_list.index(epr2_qubit)]))

            circuit.append("DETECTOR", target_rec, [X, Y, 0])

    # (4) Final measurement
    # circuit.append('X_ERROR', data_qubits_list, p_local)
    # [basis] noiseless readout of all data qubits in the Z (M) or X (MX) basis
    circuit.append('M' if basis == "Z" else 'MX', data_qubits_list)

    final_detector_list = []
    final_detector_list += (EPR_1 + EPR_2)
    final_detector_list += stabilizer_list
    final_detector_list += data_qubits_list
    final_detector_shift_list = [(-len(final_detector_list) + i) for i in range(len(final_detector_list))]

    final_detector_set = set(final_detector_list)
    final_data_qubits_set = set(data_qubits_list)
    # [basis] readout detectors from the Z-type checks (M) or the X-type checks (MX)
    final_stabilizer_z_set = set(stabilizer_z_list) if basis == "Z" else set(stabilizer_x_list)

    # perform "perfect" measurement of z stabilizers on Alice's and Bob's patch
    for i, detector in enumerate(final_detector_list):
        if detector in final_stabilizer_z_set:
            # only the last rounds of stabilizer z measurement results are used for perfect measurement
            if (len(final_detector_list) - i) > len(qubits_all_list):
                continue
            X, Y = qubit_index_position[detector]
            target_rec = []
            target_rec.append(stim.target_rec(final_detector_shift_list[i]))
            # 1 stabilizer - "4" data qubits
            possible_neighbors = [
                detector - 1,
                detector + 1,
                detector - 2 * d - 2,
                detector - 2 * d
            ]
            for neighbor in possible_neighbors:
                if neighbor in final_detector_set and neighbor in final_data_qubits_set:
                    target_rec.append(stim.target_rec(final_detector_shift_list[final_detector_list.index(neighbor)]))
            circuit.append("DETECTOR", target_rec, [X, Y, 1])

    # (5) Check fidelity, logical observables
    target_rec = []
    if basis == "Z":
        # append Alice's Z logical operators
        for i in range(len(final_detector_list) - len(data_qubits_list), len(final_detector_list)):
            if final_detector_list[i] in Alice_Z_Logical_Operator:
                target_rec.append(stim.target_rec(final_detector_shift_list[i]))
        # append Bob's Z logical operators
        for i in range(len(final_detector_list) - len(data_qubits_list), len(final_detector_list)):
            if final_detector_list[i] in Bob_Z_Logical_Operator:
                target_rec.append(stim.target_rec(final_detector_shift_list[i]))
                '''Pauli Frame'''
                alice_data_qubit = final_detector_list[i] - Bob_Shift
                epr1_qubit = alice_data_qubit + EPR_Shift
                target_rec.append(stim.target_rec(final_detector_shift_list[final_detector_list.index(epr1_qubit)]))

    else:
        # [basis] X_L(A) X_L(B) (row 0 of each patch); Pauli frame: Z^{m2} on Alice's data, m2 = EPR_2 outcomes
        for i in range(len(final_detector_list) - len(data_qubits_list), len(final_detector_list)):
            if final_detector_list[i] in Alice_X_Logical_Operator:
                target_rec.append(stim.target_rec(final_detector_shift_list[i]))
                epr2_qubit = final_detector_list[i] + Bob_Shift + EPR_Shift
                target_rec.append(stim.target_rec(final_detector_shift_list[final_detector_list.index(epr2_qubit)]))
        for i in range(len(final_detector_list) - len(data_qubits_list), len(final_detector_list)):
            if final_detector_list[i] in Bob_X_Logical_Operator:
                target_rec.append(stim.target_rec(final_detector_shift_list[i]))

    circuit.append(stim.CircuitInstruction('OBSERVABLE_INCLUDE', target_rec, [0]))
    # circuit.to_file(f"Surface_2nd_{p_local}_{p_trans}.stim")
    return circuit
