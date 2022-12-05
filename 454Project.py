# Import Packages
import pandas as pd
import numpy as np
from numpy.linalg import inv

def main():
    # Read input Data
    # Get the whole input data
    args = pd.read_excel('system_basecase_contingency2.xlsx', sheet_name = None, header = 0)

    # Get each sheet of the input data (Bus Data & Line Data)
    BusData = args.get('BusData')
    LineData = args.get('LineData')

    # Get number of buses
    N = BusData['Bus #'].max(axis=0)

    # Get index of slack bus
    Type = np.array(BusData['Type'])
    slack_index = int(np.where(Type == 'S')[0])

    # Get index of everything else except slack bus
    anti_slack_index = np.array(np.where(Type != 'S')[0])

    # Get index of PV buses
    PV_index = np.array(np.where(Type == 'G')[0])

    # Get index of PQ buses
    PQ_index = np.array(np.where(Type == 'D')[0])

    # Get the power injected at each bus
    P_gen = np.array(BusData['P Gen'])
    P_load = np.array(BusData['P MW'])
    Q_load = np.array(BusData['Q MVAr'])

    # Get data for bus connections
    bus_from = LineData['From']
    bus_to = LineData['To']

    # Get Rtotal, Xtotal, Btotal for the bus connections
    Rtotal = LineData['Rtotal, p.u.']
    Xtotal = LineData['Xtotal, p.u.']
    Btotal = 0.5j * LineData['Btotal, p.u.'] # Multiplied by 1/2 to get B for each bus

    # Get voltage at each bus
    Vset = np.array(BusData['V Set'])

    # Get the flow limit from input file
    F_max = LineData["Fmax, MVA"]

    # Initialize
    # Define base power
    Sb = 100 # 100 MVA in this case, not included in input file

    # Get the power injected at each bus
    P_total = (P_gen - P_load)/Sb
    Q_total = -(Q_load)/Sb

    # Get the power injected at the 'implicit' buses
    P_total_implicit = np.delete(P_total, slack_index, 0)
    Q_total_implicit = np.take(Q_total, PQ_index, axis=0)

    # Because we don't know the angle of any of the buses, we'll start with flat start
    # Starting point = slack bus (angle of slack bus = 0)
    angleSet = np.zeros(N)

    # Initialize iteration number
    iteration_num = 1

    # Build the admittance matrix of the system
    Y_matrix = admittance_matrix()
    G_matrix = Y_matrix.real
    B_matrix = Y_matrix.imag

    # Make a DataFrame for the output file
    # 1. Y matrix / Network Admittance Matrix
    admittance_mat = pd.DataFrame(Y_matrix, columns = None)
    # 2. Convergence history of bus results
    converge_history = pd.DataFrame({'Iteration #': [],
                                    'Largest P Mismatch (p.u.)': [],
                                    'Lasgest P Mismatch location': [],
                                    'Largest Q Mismatch (p.u.)': [],
                                    'Largest Q Mismatch location': []})

    # 3. Final bus results (P, Q, V, theta) & voltage limit check
    bus_results = pd.DataFrame({'Bus #': [],
                                'Voltage (p.u.)': [],
                                'Angle (degrees)': [],
                                'P (MW)': [],
                                'Q (MVAr)': [],
                                'Voltage Limit Check': []})

    # 4. Line flows & MVA limit check
    line_flow_results = pd.DataFrame({'Line From': [],
                                    'Line To': [],
                                    'P (MW)': [],
                                    'Q (MVAr)': [],
                                    'S (MVA)': [],
                                    'Flow Limit Check': []})
    
    # Solve the implicit equations
    while True:
        # Calculate the mismatches
        mismatch_matrix = mismatch_calculator(angleSet, Vset)

        # Build the Jacobian matrix
        jacobian_matrix = jacobian_generator(angleSet, Vset)

        # Invert the Jacobian matrix
        jacobian_invert_matrix = jacobian_inv_generator(jacobian_matrix)

        # Calculate the corrections
        corrections = corrections_calculator(jacobian_invert_matrix, mismatch_matrix)

        # Update the voltage magnitudes and angles
        angleSet, Vset = update_volt_angle(corrections, angleSet, Vset)
        
        # Update DataFrame convergence history
        converge_history = record_convergence(iteration_num, mismatch_matrix, converge_history)
        
        # Add the iteration number
        iteration_num += 1
        
        # If the mismatch of all the powers are under 0.1 p.u.
        if np.all(mismatch_matrix <= 0.1):
            # Update DataFrame for bus results & voltage limit check
            system_results(angleSet, Vset)
            
            # Update DataFrame for line flows & line flow limit check
            line_flow_check(Vset)
            
            # Combine all of the output file into one .csv file
            # create a excel writer object
            with pd.ExcelWriter("final_results.xlsx") as writer:

                # use to_excel function and specify the sheet_name and index
                # to store the dataframe in specified sheet
                converge_history.to_excel(writer, sheet_name="Admittance Matrix", index=False)
                bus_results.to_excel(writer, sheet_name="Bus Results", index=False)
                line_flow_results.to_excel(writer, sheet_name="Line Flow Results", index=False)
            break

if __name__ == "__main__":
    main()

def admittance_matrix():
    """Returns the admittance matrix of the system
    
    :return: Admittance matrix of the system
    :rtype: numpy array (data type = complex)
    """
    # Create matrix Y for the admittance matrix
    Y_matrix = np.zeros((N, N), dtype = complex)

    # Create loop to fill in the admittance matrix Y
    for i in range(0, bus_from.size):
        # The impedance of the line between two buses
        Z_total = complex(Rtotal[i], Xtotal[i])
        
        # If there is a connection between the buses
        if (np.abs(Z_total) != 0):
            # Fill in the Y matrix
            Y_matrix[(bus_from[i] - 1), (bus_to[i] - 1)] = -1 / Z_total
            Y_matrix[(bus_to[i] - 1), (bus_from[i] - 1)] = -1 / Z_total
            Y_matrix[(bus_from[i] - 1), (bus_from[i] - 1)] += (1/Z_total) + Btotal[i]
            Y_matrix[(bus_to[i] - 1), (bus_to[i] - 1)] += (1/Z_total) + Btotal[i]
    
    return Y_matrix 

def P_k (V_k, V_i, G_ki, B_ki, theta_ki):
    """Returns the active power at ONE bus with index k,
    given angles and voltages of the whole system (the total admittance
    at bus k (G and B) is fixed regardless if the voltage and angle change)

    :param V_k: Voltage at bus with index k
    :type V_k: float
    :param V_i: Voltage at bus with index i
    :type V_i: float
    :param G_ki: The real part of the admittance matrix at row k
        and column i
    :type G_ki: float
    :param B_ki: Imaginary part of the admittance matrix @ row k
        and column i
    :type B_ki: float
    :theta_ki: Angle difference between buses with index k and i
    :type theta_ki: float

    :return: The active power at one bus
    :rtype: float
    """
    return V_k * V_i * (G_ki * np.cos(theta_ki) + B_ki * np.sin(theta_ki))

def Q_k (V_k, V_i, G_ki, B_ki, theta_ki):
    """Returns the reactive power at ONE bus with index k,
    given angles and voltages of the whole system (the total admittance
    at bus k (G + jB) is fixed regardless if the voltage and angle change)

    :param V_k: Voltage at bus with index k
    :type V_k: float
    :param V_i: Voltage at bus with index i
    :type V_i: float
    :param G_ki: The real part of the admittance matrix at row k
        and column i
    :type G_ki: float
    :param B_ki: Imaginary part of the admittance matrix @ row k
        and column i
    :type B_ki: float
    :theta_ki: Angle difference between buses with index k and i
    :type theta_ki: float

    :return: The reactive power at bus with index k
    :rtype: float
    """
    return V_k * V_i * (G_ki * np.sin(theta_ki) - B_ki * np.cos(theta_ki))

def power_calculator(angleSet, Vset):
    """Returns the active and reactive power of ALL buses in the
    system given the current set of angles and voltages 
    
    :param angleSet: All the angles of each bus in the system
    :type angleSet: numpy array
    :param Vset: All the voltages of each bus in the system
    :type Vset: numpy array
    
    :return: 2 arrays for each the active and reactive power
        of each bus in the system
    :rtype: numpy arrays (data type: float)
    """
    # Create an array to put in the active and 
    # reactive power 
    P_eq = np.zeros(N)
    Q_eq = np.zeros(N)
    
    for k in range(0, N):
        # Known values
        # Voltage at a bus with index k
        V_k = Vset[k]

        for i in range(0, N):
            # Known values
            # Voltage at a bus with index i
            V_i = Vset[i]
            # Element of G matrix at row k, column i
            G_ki = G_matrix[k,i]
            # Element of B matrix at row k, column i
            B_ki = B_matrix[k,i]
            # Angle difference between bus k and bus i
            theta_ki = angleSet[k] - angleSet[i]

            # Calculate the active and reactive power at bus k
            P_eq[k] += P_k(V_k, V_i, G_ki, B_ki, theta_ki)
            Q_eq[k] += Q_k(V_k, V_i, G_ki, B_ki, theta_ki)
    
    return P_eq, Q_eq

def mismatch_calculator(angleSet, Vset):
    """Calculate the mismatch equations of the system 
    
    :param angleSet: All the angles of each bus in the system
    :type angleSet: numpy array
    :param Vset: All the voltages of each bus in the system
    :type Vset: numpy array
    
    :return: an array containing mismatches of the system 
    :rtype: numpy array (data type: float)
    """
    # Calculate power at each bus
    P_eq, Q_eq = power_calculator(angleSet, Vset)
    
    # Get the total power for the implicit equations first
    P_eq_implicit = np.delete(P_eq, slack_index, 0)
    Q_eq_implicit = np.take(Q_eq, PQ_index, axis=0)
    
    # Calculate mismatches of the implicit equations
    # Mismatch  = power at each bus - power injected to each bus
    P_mismatch = P_eq_implicit - P_total_implicit
    Q_mismatch = Q_eq_implicit - Q_total_implicit
    
    # Combine into one single mismatch matrix
    mismatch_matrix = np.append(P_mismatch, Q_mismatch, axis = 0)
    
    return mismatch_matrix

def jacobian_generator(angleSet, Vset):
    """Makes a Jacobian matrix of the current given set of 
    angles and voltages in each bus. The Jacobian matrix is made
    of 4 smaller matrices called: H, N, M, L. Each of them has 
    its own function so we only need to call them in this function
    
    :param angleSet: All the angles of each bus in the system
    :type angleSet: numpy array
    :param Vset: All the voltages of each bus in the system
    :type Vset: numpy array
    
    :return: The Jacobian matrix of the current angles and voltages 
    :rtype: numpy array (data type: float)
    """
    # Get H, N, M, L matrices
    H_matrix = H_mat(angleSet, Vset)
    N_matrix = N_mat(angleSet, Vset)
    M_matrix = M_mat(angleSet, Vset)
    L_matrix = L_mat(angleSet, Vset)
    
    # Combine the matrices together to get our jacobian matrix
    jacobian_column_1 = np.append(H_matrix, N_matrix, axis = 0)
    jacobian_column_2 = np.append(M_matrix, L_matrix, axis = 0)
    jacobian = np.append(jacobian_column_1, jacobian_column_2, axis = 1)
    
    return jacobian

def H_mat(angleSet, Vset):
    """Returns the H matrix of our Jacobian matrix with the given
    current set of voltages and angles. 
    
    :param angleSet: All the angles of each bus in the system
    :type angleSet: numpy array
    :param Vset: All the voltages of each bus in the system
    :type Vset: numpy array
    
    :return: The H matrix of the current angles and voltages 
    :rtype: numpy array (data type: float)
    """
    # Initialize an H (faux) matrix with dimension N by N
    H_faux_matrix = np.zeros((N, N))

    # Loop through k
    for k in range(0, N):
        # Known values
        # Voltage at a bus with index k
        V_k = Vset[k]

        for i in range(0, N):
            # Known values
            # Voltage at a bus with index i
            V_i = Vset[i]
            # Element of G matrix at row k, column i
            G_ki = G_matrix[k,i]
            # Element of B matrix at row k, column i
            B_ki = B_matrix[k,i]
            # Angle difference between bus k and bus i
            theta_ki = angleSet[k] - angleSet[i]
        
            # If i != k
            if (i != k):
                # Fill in the H (faux) matrix with corresponding formula for i != k
                H_faux_matrix[k, i] = H_matrix_value_input_1(V_k, V_i, theta_ki, G_ki, B_ki)
            
            # If i = k
            else:
                # Create loop to go through all buses except bus k
                for j in range(0, N):
                    # Known values
                    # Voltage at a bus with index j
                    V_j = Vset[j]
                    # Angle difference between bus k and bus j
                    theta_kj = angleSet[k] - angleSet[j]
                    # Element of G matrix at row k, column j
                    G_kj = G_matrix[k,j]
                    # Element of B matrix at row k, column j
                    B_kj = B_matrix[k,j]
                    # Fill in the H (faux) matrix with corresponding formula for i = k
                    H_faux_matrix[k, k] += H_matrix_value_input_2(V_k, V_j, theta_kj, G_kj, B_kj)

                # Subtract when j = k so it doesn't include bus k
                H_faux_matrix[k,k] -= H_matrix_value_input_2(V_k, V_i, theta_ki, G_ki, B_ki)

    # Select the correct range for H matrix from our faux H matrix
    # Correct range: does not include slack bus
    H_matrix = np.delete(H_faux_matrix, slack_index, 0)
    H_matrix = np.delete(H_matrix, slack_index, 1)

    return H_matrix

def H_matrix_value_input_1 (V_k, V_i, theta_ki, G_ki, B_ki):
    """Returns the element that goes in H matrix at row k, column i
    when i != k
    
    :param V_k: Voltage at bus with index k
    :type V_k: float
    :param V_i: Voltage at bus with index i
    :type V_i: float
    :theta_ki: Angle difference between buses with index k and i
    :type theta_ki: float
    :param G_ki: The real part of the admittance matrix at row k
        and column i
    :type G_ki: float
    :param B_ki: Imaginary part of the admittance matrix @ row k
        and column i
    :type B_ki: float

    :return: Element for H matrix at row k, column i when i != k
    :rtype: float
    """
    # H_ki = element to go at H matrix row k column i
    H_ki = V_k * V_i * (G_ki * np.sin(theta_ki) - B_ki * np.cos(theta_ki))
    return H_ki

def H_matrix_value_input_2 (V_k, V_j, theta_kj, G_kj, B_kj):
    """Returns part of the element that will go in H matrix at 
    row k, column k (when i = k). It is part of the element because
    it is part of a summation going through all the buses in the 
    system (the element that is in the loop when index = j)
    
    :param V_k: Voltage at bus with index k
    :type V_k: float
    :param V_j: Voltage at bus with index j
    :type V_j: float
    :theta_kj: Angle difference between buses with index k and j
    :type theta_kj: float
    :param G_kj: The real part of the admittance matrix at row k
        and column j
    :type G_kj: float
    :param B_kj: Imaginary part of the admittance matrix @ row k
        and column j
    :type B_kj: float

    :return: a number for part of the summation element when 
        the index of the loop is j. For H matrix at row k, 
        column k
    :rtype: float
    """
    # H_kk = element to go at H matrix row k column k when index of
    # the loop is j
    H_kk = V_k * V_j * (- G_kj * np.sin(theta_kj) + B_kj * np.cos(theta_kj))
    return H_kk

def N_mat(angleSet, Vset):
    """Returns the N matrix of our Jacobian matrix with the given
    current set of voltages and angles. 
    
    :param angleSet: All the angles of each bus in the system
    :type angleSet: numpy array
    :param Vset: All the voltages of each bus in the system
    :type Vset: numpy array
    
    :return: The N matrix of the current angles and voltages 
    :rtype: numpy array (data type: float)
    """
    # Initialize an N (faux) matrix with dimension N by N
    N_faux_matrix = np.zeros((N, N))

    # Loop through k
    for k in range(0, N):
        # Known values
        # Voltage at a bus with index k
        V_k = Vset[k]

        for i in range(0, N):
            # Known values
            # Voltage at a bus with index i
            V_i = Vset[i]
            # Element of G matrix at row k, column i
            G_ki = G_matrix[k,i]
            # Element of B matrix at row k, column i
            B_ki = B_matrix[k,i]
            # Angle difference between bus k and bus i
            theta_ki = angleSet[k] - angleSet[i]

            # If i != k
            if (i != k):
                # Fill in the N (faux) matrix with corresponding formula for i != k
                N_faux_matrix[k, i] = N_matrix_value_input_1(V_k, V_i, theta_ki, G_ki, B_ki)
            # If i = k
            else:
                # Create loop to go through all buses except bus k
                for j in range(0, N):
                    # Known values
                    # Voltage at a bus with index j
                    V_j = Vset[j]
                    # Angle difference between bus k and bus j
                    theta_kj = angleSet[k] - angleSet[j]
                    # Element of G matrix at row k, column j
                    G_kj = G_matrix[k,j]
                    # Element of B matrix at row k, column j
                    B_kj = B_matrix[k,j]
                    # Fill in the N (faux) matrix with corresponding formula for i = k
                    N_faux_matrix[k, k] += N_matrix_value_input_2(V_k, V_j, theta_kj, G_kj, B_kj)

                # Subtract when j = k so it doesn't include bus k
                N_faux_matrix[k,k] -= N_matrix_value_input_2(V_k, V_i, theta_ki, G_ki, B_ki)

    # Select the correct range for N matrix from our faux N matrix
    # Delete rows with PV buses
    N_matrix = np.take(N_faux_matrix, np.array(PQ_index), axis=0)
    # Delete column of slack bus
    N_matrix = np.delete(N_matrix, slack_index, 1)

    return N_matrix

def N_matrix_value_input_1 (V_k, V_i, theta_ki, G_ki, B_ki):
    """Returns the element that goes in N matrix at row k, column i
    when i != k
    
    :param V_k: Voltage at bus with index k
    :type V_k: float
    :param V_i: Voltage at bus with index i
    :type V_i: float
    :theta_ki: Angle difference between buses with index k and i
    :type theta_ki: float
    :param G_ki: The real part of the admittance matrix at row k
        and column i
    :type G_ki: float
    :param B_ki: Imaginary part of the admittance matrix @ row k
        and column i
    :type B_ki: float

    :return: Element for N matrix at row k, column i when i != k
    :rtype: float
    """
    # N_ki = element to go at N matrix row k column i
    N_ki = V_k * V_i * (- G_ki * np.cos(theta_ki) - B_ki * np.sin(theta_ki))
    return N_ki

def N_matrix_value_input_2 (V_k, V_j, theta_kj, G_kj, B_kj):
    """Returns part of the element that will go in N matrix at 
    row k, column k (when i = k). It is part of the element because
    it is part of a summation going through all the buses in the 
    system (the element that is in the loop when index = j)
    
    :param V_k: Voltage at bus with index k
    :type V_k: float
    :param V_j: Voltage at bus with index j
    :type V_j: float
    :theta_kj: Angle difference between buses with index k and j
    :type theta_kj: float
    :param G_kj: The real part of the admittance matrix at row k
        and column j
    :type G_kj: float
    :param B_kj: Imaginary part of the admittance matrix @ row k
        and column j
    :type B_kj: float

    :return: a number for part of the summation element when 
        the index of the loop is j. For N matrix at row k, 
        column k
    :rtype: float
    """
    # N_kk = element to go at N matrix row k column k when index of
    # the loop is j
    N_kk = V_k * V_j * (G_kj * np.cos(theta_kj) + B_kj * np.sin(theta_kj))
    return N_kk

def M_mat(angleSet, Vset):
    """Returns the M matrix of our Jacobian matrix with the given
    current set of voltages and angles. 
    
    :param angleSet: All the angles of each bus in the system
    :type angleSet: numpy array
    :param Vset: All the voltages of each bus in the system
    :type Vset: numpy array
    
    :return: The M matrix of the current angles and voltages 
    :rtype: numpy array (data type: float)
    """
    # Initialize an M (faux) matrix with dimension N by N
    M_faux_matrix = np.zeros((N, N))

    # Loop through k
    for k in range(0, N):
        # Known values
        # Voltage at a bus with index k
        V_k = Vset[k]

        for i in range(0, N):
            # Known values
            # Voltage at a bus with index i
            V_i = Vset[i]
            # Element of G matrix at row k, column i
            G_ki = G_matrix[k,i]
            # Element of B matrix at row k, column i
            B_ki = B_matrix[k,i]
            # Angle difference between bus k and bus i
            theta_ki = angleSet[k] - angleSet[i]

            # If i != k
            if (i != k):
                # Fill in the M (faux) matrix with corresponding formula for i != k
                M_faux_matrix[k, i] = M_matrix_value_input_1(V_k, theta_ki, G_ki, B_ki)
            # If i = k
            else:
                # Create loop to go through all buses except bus k
                for j in range(0, N):
                    # Known values
                    # Voltage at a bus with index j
                    V_j = Vset[j]
                    # Angle difference between bus k and bus j
                    theta_kj = angleSet[k] - angleSet[j]
                    # Element of G matrix at row k, column j
                    G_kj = G_matrix[k,j]
                    # Element of B matrix at row k, column j
                    B_kj = B_matrix[k,j]
                    # Fill in the M (faux) matrix with corresponding formula for i = k
                    M_faux_matrix[k, k] += M_matrix_value_input_2(V_j, theta_kj, G_kj, B_kj)

                # Subtract when j = k so it doesn't include bus k
                M_faux_matrix[k,k] -= M_matrix_value_input_2(V_i, theta_ki, G_ki, B_ki)
                
                # Element of G matrix at row k, column k
                G_kk = G_matrix[k,k]
                # Finish the formula for M_kk                
                M_faux_matrix[k,k] =  M_faux_matrix[k,k] + 2 * G_kk * V_k

    # Select the correct range for M matrix from our faux M matrix
    # Delete columns with PV buses
    M_matrix = np.take(M_faux_matrix, np.array(PQ_index), axis=1)

    # Delete row w/ slack bus
    M_matrix = np.delete(M_matrix, slack_index, 0)
    
    return M_matrix

def M_matrix_value_input_1 (V_k, theta_ki, G_ki, B_ki):
    """Returns the element that goes in M matrix at row k, column i
    when i != k
    
    :param V_k: Voltage at bus with index k
    :type V_k: float
    :theta_ki: Angle difference between buses with index k and i
    :type theta_ki: float
    :param G_ki: The real part of the admittance matrix at row k
        and column i
    :type G_ki: float
    :param B_ki: Imaginary part of the admittance matrix @ row k
        and column i
    :type B_ki: float

    :return: Element for M matrix at row k, column i when i != k
    :rtype: float
    """
    # M_ki = element to go at M matrix row k column i
    M_ki = V_k * (G_ki * np.cos(theta_ki) + B_ki * np.sin(theta_ki))
    return M_ki

def M_matrix_value_input_2 (V_j, theta_kj, G_kj, B_kj):
    """Returns part of the element that will go in M matrix at 
    row k, column k (when i = k). It is part of the element because
    it is part of a summation going through all the buses in the 
    system (the element that is in the loop when index = j)
    
    :param V_j: Voltage at bus with index j
    :type V_j: float
    :theta_kj: Angle difference between buses with index k and j
    :type theta_kj: float
    :param G_kj: The real part of the admittance matrix at row k
        and column j
    :type G_kj: float
    :param B_kj: Imaginary part of the admittance matrix @ row k
        and column j
    :type B_kj: float

    :return: a number for part of the summation element when 
        the index of the loop is j. For M matrix at row k, 
        column k
    :rtype: float
    """
    # M_kk = element to go at M matrix row k column k when index of
    # the loop is j
    M_kk = V_j * (G_kj * np.cos(theta_kj) + B_kj * np.sin(theta_kj))
    return M_kk

def L_mat(angleSet, Vset):
    """Returns the L matrix of our Jacobian matrix with the given
    current set of voltages and angles. 
    
    :param angleSet: All the angles of each bus in the system
    :type angleSet: numpy array
    :param Vset: All the voltages of each bus in the system
    :type Vset: numpy array
    
    :return: The L matrix of the current angles and voltages 
    :rtype: numpy array (data type: float)
    """
    # Initialize an L (faux) matrix with dimension N by N
    L_faux_matrix = np.zeros((N, N))

    # Loop through k
    for k in range(0, N):
        # Known values
        # Voltage at a bus with index k
        V_k = Vset[k]

        for i in range(0, N):
            # Known values
            # Voltage at a bus with index i
            V_i = Vset[i]
            # Element of G matrix at row k, column i
            G_ki = G_matrix[k,i]
            # Element of B matrix at row k, column i
            B_ki = B_matrix[k,i]
            # Angle difference between bus k and bus i
            theta_ki = angleSet[k] - angleSet[i]

            # If i != k
            if (i != k):
                # Fill in the L (faux) matrix with corresponding formula for i != k
                L_faux_matrix[k, i] = L_matrix_value_input_1(V_k, theta_ki, G_ki, B_ki)
            # If i = k
            else:
                # Create loop to go through all buses except bus k
                for j in range(0, N):
                    # Known values
                    # Voltage at a bus with index j
                    V_j = Vset[j]
                    # Angle difference between bus k and bus j
                    theta_kj = angleSet[k] - angleSet[j]
                    # Element of G matrix at row k, column j
                    G_kj = G_matrix[k,j]
                    # Element of B matrix at row k, column j
                    B_kj = B_matrix[k,j]
                    # Fill in the L (faux) matrix with corresponding formula for i = k
                    L_faux_matrix[k, k] += L_matrix_value_input_2(V_j, theta_kj, G_kj, B_kj)

                # Subtract when j = k so it doesn't include bus k
                L_faux_matrix[k,k] -= L_matrix_value_input_2(V_i, theta_ki, G_ki, B_ki)
                
                # Element of B matrix at row k, column k
                B_kk = B_matrix[k,k]
                # Finish the formula for L_kk
                L_faux_matrix[k,k] =  L_faux_matrix[k,k] - 2 * B_kk * V_k

    # Select the correct range for L matrix from our faux L matrix
    # Delete columns with slack + PV buses
    L_matrix = np.take(L_faux_matrix, np.array(PQ_index), axis=1)

    # Delete rows with slack + PV buses
    L_matrix = np.take(L_matrix, np.array(PQ_index), axis=0)
    
    return L_matrix

def L_matrix_value_input_1 (V_k, theta_ki, G_ki, B_ki):
    """Returns the element that goes in L matrix at row k, column i
    when i != k
    
    :param V_k: Voltage at bus with index k
    :type V_k: float
    :theta_ki: Angle difference between buses with index k and i
    :type theta_ki: float
    :param G_ki: The real part of the admittance matrix at row k
        and column i
    :type G_ki: float
    :param B_ki: Imaginary part of the admittance matrix @ row k
        and column i
    :type B_ki: float

    :return: Element for L matrix at row k, column i when i != k
    :rtype: float
    """
    # L_ki = element to go at L matrix row k column i
    L_ki = V_k * (G_ki * np.sin(theta_ki) - B_ki * np.cos(theta_ki))
    return L_ki

# Function to fill in L matrix when i = k
def L_matrix_value_input_2 (V_j, theta_kj, G_kj, B_kj):
    """Returns part of the element that will go in L matrix at 
    row k, column k (when i = k). It is part of the element because
    it is part of a summation going through all the buses in the 
    system (the element that is in the loop when index = j)
    
    :param V_j: Voltage at bus with index j
    :type V_j: float
    :theta_kj: Angle difference between buses with index k and j
    :type theta_kj: float
    :param G_kj: The real part of the admittance matrix at row k
        and column j
    :type G_kj: float
    :param B_kj: Imaginary part of the admittance matrix @ row k
        and column j
    :type B_kj: float

    :return: a number for part of the summation element when 
        the index of the loop is j. For L matrix at row k, 
        column k
    :rtype: float
    """
    # L_kk = element to go at L matrix row k column k when index of
    # the loop is j
    L_kk = V_j * (G_kj * np.sin(theta_kj) - B_kj * np.cos(theta_kj))
    return L_kk

def jacobian_inv_generator(jacobian):
    """Returns the inverse of the given Jacobian matrix
    
    :param jacobian: The given Jacobian matrix
    :type jacobian: numpy array (data type: float)
    
    :return: The inverse of the Jacobian matrix
    :rtype: numpy array (data type: float)
    """
    return inv(jacobian)

def corrections_calculator(jacobian_inv, mismatch_eq):
    """Returns the corrections for the angles and voltages
    of the implicit equations for the next iteration
    
    :param jacobian_inv: The given inverse of the jacobian matrix
    :type jacobian_inv: numpy array (data type: float)
    :param mismatch_eq: The mismatches calculated in function
        mismatch_calculator
    :type mismatch_eq: numpy array (data type: float)
    
    :return corrections: Corrections for the angles and
        voltages of the implicit equations
    :rtype corrections: numpy array (data type: float)
    """
    corrections = -np.matmul(jacobian_inv, mismatch_eq)
    return corrections

def update_volt_angle(corrections, angleSet, Vset):
    """Returns the updated voltages and angles of all buses in the
    system after having the corrections from the implicit equations
    
    :param corrections: The given inverse of the jacobian matrix
    :type corrections: numpy array (data type: float)
    :param angleSet: All the angles of each bus in the system
    :type angleSet: numpy array (data type: float)
    :param Vset: All the voltages of each bus in the system
    :type Vset: numpy array (data type: float)
    
    :return angleSet: The updated angles for each bus in the system
        after the correction
    :rtype angleSet: numpy array (data type: float)
    :return Vset: The updated voltages for each bus in the system
        after the correction
    :rtype Vset: numpy array (data type: float)
    """
    # Reduce the angles to not include slack bus
    angle_reduced = np.delete(angleSet, slack_index, 0)
    
    # Reduce the voltages to only include PQ buses
    Vreduced = np.take(Vset, PQ_index, axis=0)
    
    # Combine the reduced angles and voltages
    angle_V_reduced = np.append(angle_reduced, Vreduced, axis = 0)
    
    # New reduced angles & voltages = old reduced angles & voltages + corrections
    angle_V_reduced += corrections
    
    # Separate the reduced angles & voltages
    angleReduced = angle_V_reduced[0:len(angle_reduced)]
    Vreduced = angle_V_reduced[len(angle_reduced):]
    
    # Combine the new reduced angles & voltages into the full set of angles & voltages
    np.put(angleSet, anti_slack_index, angleReduced)
    np.put(Vset, PQ_index, Vreduced)
    
    return angleSet, Vset

def record_convergence(iteration_num, mismatch_matrix, converge_history):
    """Updates the convergence record to our output DataFrame 
    (converge_history) for ONE iteration of the Newton-Rhapson 
    method. It records the largest active and reactive mismatch 
    out of all the mismatches, along with which buses those 
    two mismatches happen at.
    
    :param iteration_num: Given iteration number
    :type iteration_num: int
    :param mismatch_matrix: The matrix containing all of 
        the mismatches from the implicit equations
    :type mismatch_matrix: numpy array (float)
    :param converge_history: Latest convergence history DataFrame
    :type converge_history: DataFrame
    
    :return: returns the updated convergence history
    :rtype: DataFrame
    """
    # Split the mismatch matrix to real and reactive power
    P_mismatch = mismatch_matrix[0:N-1]
    Q_mismatch = mismatch_matrix[N-1:]
    
    # Find the largest P and Q mismatch
    P_max_mismatch = np.amax(P_mismatch)
    Q_max_mismatch = np.amax(Q_mismatch)
    
    # Find location of the bus
    P_max_mismatch_index = int(anti_slack_index[np.argmax(P_mismatch)])
    Q_max_mismatch_index = int(PQ_index[np.argmax(Q_mismatch)])
    
    # Create the new row to be added
    new_row = pd.DataFrame([[iteration_num, P_max_mismatch, 
                             P_max_mismatch_index, Q_max_mismatch,
                            Q_max_mismatch_index]], 
                           columns = converge_history.columns)
    
    # Put into converge_history DataFrame
    converge_history = pd.concat([converge_history, new_row], ignore_index = True)
    
    return converge_history

def system_results(angleSet, Vset):
    """Reports the voltage, angle, active power,
    reactive power and voltage limit check of each bus
    
    :param angleSet: All the angles of each bus in the system
    :type angleSet: numpy array (data type: float)
    :param Vset: All the voltages of each bus in the system
    :type Vset: numpy array (data type: float)
    
    :return: Not really a return, but it fills in the DataFrame
        bus_results
    """
    # Find the active and reactive power at each bus with given 
    # angles and voltages
    P, Q = power_calculator(angleSet, Vset)
    
    # Convert to SI units
    P = P * Sb
    Q = Q * Sb
    
    # Loop through all the buses
    for i in range(0, N):
        # Do voltage limit check
        if (Vset[i] > 1.05 or Vset[i] < 0.95 ):
            Vcheck = 'Fail'
        else:
            Vcheck = 'Pass'
        
        # Update the DataFrame for bus_results
        bus_results.loc[i] = [i + 1, Vset[i], ((angleSet[i]*180)/np.pi), P[i], Q[i], Vcheck]

def line_flow_check(Vset):
    """Updates the DataFrame to report the line flow 
    between all the buses (line flow from bus 1 to 2 !=
    line from from 2 to 1) and also do a check if it exceeds
    the line flow limits
    
    :param Vset: All the voltages of each bus in the system
    :type Vset: numpy array (data type: float)
    
    :return: Not really a return, but it fills in the DataFrame
        line_flow_results 
    """
    # Make a empty matrices for:
    # Line flow limits
    F_matrix = np.zeros((N, N))
    # Line impedance
    Z_lineflow = np.zeros((N, N), dtype = complex)
    # Shunt capacitance
    B_lineflow = np.zeros((N, N), dtype = complex)
    # Also make one for the actual line flow 
    S_LineFlow = np.zeros((N, N),dtype = complex)
    
    # Fill in the matrices we just made (except the line flow)
    for j in range(0, bus_from.size):
        # Calculate the line impedance
        Z = complex(Rtotal[j], Xtotal[j])
        
        # Fill in Z_lineflow matrix
        Z_lineflow[(bus_from[j] - 1), (bus_to[j] - 1)] = Z
        Z_lineflow[(bus_to[j] - 1), (bus_from[j] - 1)] = Z
        
        # Fill in F_matrix
        F_matrix[(bus_from[j] - 1), (bus_to[j] - 1)] = F_max[j]
        F_matrix[(bus_to[j] - 1), (bus_from[j] - 1)] = F_max[j]
        
        # Fill in B_lineflow matrix
        B_lineflow[(bus_from[j] - 1), (bus_from[j] - 1)] = Btotal[j]
        B_lineflow[(bus_to[j] - 1), (bus_to[j] - 1)] = Btotal[j]
    
    # We can now fill in the line flow matrix
    for i in range (0, N):
        for k in range (0, N):
            # Voltage at bus i
            V_i = Vset[i]
            # Voltage at bus k
            V_k = Vset[k]
            # Get the line impedance and shunt capacitance for bus i
            Z_ik = Z_lineflow[i,k]
            B_ik = B_lineflow[i,k]
            
            # Find the line flow of bus i to bus k
            S_LineFlow[i,k] = calculate_LineFlow(V_i, V_k,Z_ik,B_ik)
            
            # Line flow limit check
            if (np.abs(S_LineFlow[i,k]) > F_matrix[i,k]):
                lineCheck = 'Fail'
            else:
                lineCheck = 'Pass'
                
            # Update the DataFrame for bus_results
            line_flow_results.loc[i] = [i + 1, k + 1, S_LineFlow[i,k].real, 
                                        S_LineFlow[i,k].imag, S_LineFlow[i,k], lineCheck]

def calculate_LineFlow (V_i, V_k,Z_ik,B_ik):
    """Returns the line flow (in complex power) from 
    bus i to bus k
    
    :param V_i: Voltage at bus i
    :type V_i: float
    :param V_k: Voltage at bus k
    :type V_k: float
    :param Z_ik: Line impedance between bus i and k
    :type Z_ik: complex
    :param B_ik: 1/2 of shunt capacitance between bus i and k
    
    :return S: Line flow from bus i to bus k (complex power)
    :rtype S: complex
    """
    # Find the current from bus i to bus k
    # If impedance = 0, there's no connection between the two buses
    if (Z_ik != 0):
        I_ik = (V_i-V_k)/Z_ik + V_i*B_ik
    else:
        I_ik = 0
    
    # Find the line flow from bus i to k (in MVA)
    S = V_i * np.conjugate(I_ik) * Sb
    return S