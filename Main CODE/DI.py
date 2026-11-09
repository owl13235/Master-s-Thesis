import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.linalg import block_diag

# ==============================================================================
# SEZIONE 1: GEOMETRIA E TOPOLOGIA DEL SISTEMA
# ==============================================================================
# In questa sezione si definiscono le proprietà spaziali del bridge chirale.
# Il calcolo vettoriale è necessario per determinEnergy_2_eV = Energy_2_eV[-len(Energy_1):]are il termine di Spin-Orbita
# dipendente dalla geometria (termini r x p).

def calcola_ui(n_siti):
    """
    Calcola i vettori di chiralità υi locali per una struttura elicoidale discreta.
    
    Output:
        - ui = velocità
    """
    ui = np.tile([0.0, 0.0, 1.0], (n_siti, 1))

    return np.array(ui)


# ==============================================================================
# SEZIONE 2: GENERATORI DI HAMILTONIANA E OSSERVABILI
# ==============================================================================
# Qui si costruiscono le matrici operatoriali. 
# Si utilizzano prodotti di Kronecker per comporre gli spazi di Hilbert:
# H_tot = H_site (Posizione) ⊗ H_spin (Spin) ⊗ H_phon (Fononi)

def genera_h_spin_orbita(n_siti, s_arr, ui, sigma, so_vicini=1):
    """
    Costruisce l'Hamiltoniana di Spin-Orbita (H_SO).
    
    Modello Fisico:
        H_SO = i * sum_{k} lambda_k * (u_k * sigma) * (|k><k+1| - h.c.)
        Rappresenta il termine di hopping dipendente dallo spin indotto dalla 
        geometria chirale.
    
    Input:
        - sigma: Lista matrici di Pauli [Sx, Sy, Sz].
        - so_vicini: Range di interazione (1 = Nearest Neighbors).
    
    Output:
        - Matrice densa complessa nello spazio Posizione ⊗ Spin.
    """
    # Dimensione sottospazio Bridge (esclusi Lead A/B)
    dim_bridge_spin = n_siti * 2
    h_so_bridge = np.zeros((dim_bridge_spin, dim_bridge_spin), dtype=complex)

    # Iterazione sulle componenti spaziali dello spin (x, y, z)
    for j in range(3):
        # Matrice di hopping spaziale pura
        m = np.zeros((n_siti, n_siti), dtype=complex)
        
        # Riempimento elementi di matrice off-diagonal
        for i in range(0, n_siti-so_vicini):
            # Coupling: lambda * componente geometrica u_j
            coupling = s_arr[i] * ui[i, j]
            m[i, i + so_vicini] = coupling
        
        # Tensor product: Spazio(m) ⊗ Spin(sigma_j) + Hermitian Conjugate
        term = np.kron(m, sigma[j])
        h_so_bridge += term + term.conj().T

    # Padding per includere i siti A e B (senza SOC)
    # H_total = [0_A] ⊕ [H_bridge] ⊕ [0_B]
    zeros_site = np.zeros((2, 2), dtype=complex)
    h_so_totale = block_diag(zeros_site, h_so_bridge, zeros_site)
    
    return h_so_totale


# --- Funzioni Ausiliarie per Osservabili Locali ---

def rotate(evecs, operator):
    """Trasformazione unitaria: O_rot = U^dag O U."""
    rotated = np.conj(evecs).T @ operator @ evecs
    return rotated

def f_pop_n(n, n_siti, id_fon):
    """
    Costruisce l'operatore Popolazione per il sito n:
    P_n = |n><n| ⊗ I_spin ⊗ I_phon
    """
    # Proiettore spaziale |n><n| nella base {A, 1..N, B}
    pop_n = np.diag([0] * (n - 1) + [1] + [0] * (n_siti + 2 - n))
    # Estensione allo spazio totale
    pop_n = np.kron(pop_n, np.eye(2)) # Spazio Spin
    pop_n = np.kron(pop_n, id_fon)    # Spazio Fononico
    return pop_n

def f_pos_n(n, n_siti, fon):
    """
    Costruisce l'operatore Coordinate Fononica locale Q_n:
    Q_n = P_n ⊗ I_spin ⊗ (b + b^dag)/sqrt(2)
    """
    # Proiettore spaziale
    pop_n = np.diag([0] * (n - 1) + [1] + [0] * (n_siti + 2 - n))
    pop_n = np.kron(pop_n, np.eye(2))
    
    # Operatore di Posizione Fononica (adimensionale)
    b = np.zeros((fon, fon))
    for k in range(1, fon):
        b[k-1, k] = np.sqrt(k)
    Q = (b + b.T) * (1/np.sqrt(2))
    
    pos_n = np.kron(pop_n, Q)
    return pos_n

def f_num_n(n, n_siti, fon):
    """
    Costruisce l'operatore Numero Fononico locale N_ph_n:
    N_ph_n = P_n ⊗ I_spin ⊗ (b^dag b)
    """
    pop_n = np.diag([0] * (n - 1) + [1] + [0] * (n_siti + 2 - n))
    pop_n = np.kron(pop_n, np.eye(2))
    
    # Operatore numero
    b = np.zeros((fon, fon))
    for k in range(1, fon):
        b[k-1, k] = np.sqrt(k)
    num = b.T @ b
    
    num_n = np.kron(pop_n, num)
    return num_n

def hop(n_siti, id_spin, id_fon):
    """
    Definisce l'operatore di tunneling tra Lead e Bridge.
    T = sum (|i><j| + h.c.)
    """
    hop_A = np.diag([1] + [0]*(n_siti), k=1) + np.diag([1] + [0]*(n_siti), k=-1)
    hop_B = np.diag([0]*(n_siti) + [1], k=1) + np.diag([0]*(n_siti) + [1], k=-1)
    
    hop_A = np.kron(np.kron(hop_A, id_spin), id_fon)
    hop_B = np.kron(np.kron(hop_B, id_spin), id_fon)
    return hop_A, hop_B

def f_mag_n(n, n_siti, id_fon):
    """
    Costruisce gli operatori di Magnetizzazione Locale (Spin).
    M_i = P_n ⊗ sigma_i ⊗ I_phon  per i in {x,y,z}
    """
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    
    # Base spaziale
    pop_n = np.diag([0] * (n - 1) + [1] + [0] * (n_siti + 2 - n))
    
    # Tensor products
    pop_n_z = np.kron(np.kron(pop_n, sigma_z), id_fon)
    pop_n_x = np.kron(np.kron(pop_n, sigma_x), id_fon)
    pop_n_y = np.kron(np.kron(pop_n, sigma_y), id_fon)
    
    return pop_n_x, pop_n_y, pop_n_z

# --- Orchestrazione Operatori ---

def genera_mappa_operatori(n_siti, fon, func_generatrice):
    """
    Genera un dizionario strutturato di operatori per tutti i siti (D, 1..N, A).
    Utilizzata per istanziare massivamente osservabili dello stesso tipo.
    """
    # Etichette siti: D (Donatore), 1..N (Bridge), A (Accettore)
    etichette_siti = ['D'] + list(range(1, n_siti + 1)) + ['A']
    mappa_operatori = {}
    
    for i, label in enumerate(etichette_siti):
        # Indice 'i+1' perché le funzioni generatrici assumono indicizzazione 1-based per i siti
        ops = func_generatrice(i + 1, n_siti, fon)
        mappa_operatori[label] = ops
        
    return mappa_operatori

def genera_suite_osservabili(n_siti, diz_pop, diz_pol, diz_pos, diz_num, id_tot):
    """
    Aggrega tutte le mappe di operatori in liste piatte (ops, labels)
    pronte per essere passate al solver di QuTiP.
    """
    siti = ['D'] + list(range(1, n_siti + 1)) + ['A']
    e_ops = []
    e_labels = []

    # 1. Popolazioni elettroniche
    for s in siti:
        if s in diz_pop:
            e_ops.append(diz_pop[s])
            e_labels.append(f'pop_{s}')

    # 2. Polarizzazioni di spin (x, y, z)
    for s in siti:
        if s in diz_pol:
            componenti = ['x', 'y', 'z']
            for i, comp in enumerate(componenti):
                e_ops.append(diz_pol[s][i])
                e_labels.append(f'pol_{s}_{comp}')

    # 3. Osservabili Fononiche (Posizione Q e Numero n)
    for s in siti:
        if s in diz_pos:
            e_ops.append(diz_pos[s])
            e_labels.append(f'pos_{s}')
    for s in siti:
        if s in diz_num:
            e_ops.append(diz_num[s])
            e_labels.append(f'num_{s}')

    # 4. Controllo Normalizzazione
    e_ops.append(id_tot)
    e_labels.append('id_tot')

    return e_ops, e_labels


def project_first(fon, n_siti, M):
    """Proietta l'operatore M nel sottospazio efficace."""
    # Definizione delle dimensioni originali intermedie
    dim =  M.shape[0]
    
    # Definizione dei blocchi di indici da mantenere
    a_block = [0 , fon]
    b_block = np.arange(2*fon, dim)
    
    # Costruzione array indici
    indices_to_keep = np.concatenate((a_block, b_block)) # Indici nello spazio "grande"
    
    #Proiezione su sottospazio efficace
    M_final = M[np.ix_(indices_to_keep, indices_to_keep)]
    
    return M_final
    
def project_M(fon, n_siti, M, evecs):
    """Applica proiezione geometrica, rotazione e proiezione energetica."""
    
    M = project_first(fon, n_siti, M)
    M = rotate(evecs, M)
    
    return M

# ==============================================================================
# SEZIONE 3: CORE DELLA SIMULAZIONE (TIGHT BINDING + REDFIELD)
# ==============================================================================

def simulazione_tight_binding(dep,tra,pL,n_siti):
    """
    Esegue la dinamica di sistema aperto (Master Equation).
    
    Modello:
        H = H_el + H_vib + H_int + H_bath
        - H_el: Tight binding con Spin-Orbita.
        - H_vib: Oscillatori armonici locali.
        - H_int: Accoppiamento Elettrone-Fonone (Peierls + Holstein).
        - Dissipazione: Teoria di Redfield (accoppiamento debole col bagno).

    Parametri:
        dep : Rilassamento di Spin
        tra : Trasferimento ponte A-D
        pL : Phonon Loss Operator 
        n_siti: Numero siti del ponte
    """
    
    # --------------------------------------------------------------------------
    # 3.1 Costanti Fisiche e Parametri del Modello
    # --------------------------------------------------------------------------
    hbar = 6.58212e-4 # [eV ps] Costante di Planck ridotta

    # Parametri Strutturali
    fon = 15
    n_siti = n_siti        # Lunghezza bridge
    w_o = 0.001        # [eV] Energia del modo fononico

    # Parametri Hamiltoniana Elettronica
    t = 0.001          # [eV] Hopping  (NN)
    t1 = 0.001         # [eV] Hopping  Peierls
    eT = 0            # [eV] On-site energy (Bridge)
    so = 0.0001j       # [eV] SOC
    so_vicini = 1    # Range SOC
    chi = -0.0001j   # [eV] SOC Peierls
    
    # Parametri Lead (Donatore/Accettore)
    tA = tra; tB = tra # [eV] Accoppiamento ai lead
    eA = 10; eB = -10     # [eV] Potenziale chimico 
    
    # Parametri Dephasing
    spin_dep_z = 0.0  # [eV] dephasing di spin
    spin_dep_xy = dep  # [eV] rilassamento di spin
    p_loss = pL # [eV] Phonon Loss Operator

    # Parametri per convergenza
    a_Q = 1e-5  # [eV] Termine quartico
    zee = 1e-12       # [eV] Campo magnetico esterno (Zeeman splitting)
    
    # Griglia temporale integrazione
    t_max = n_siti*40      # [ps]
    t_int = t_max*5
    tlist = np.linspace(0, t_max, t_int)

    # --------------------------------------------------------------------------
    # 3.2 Costruzione degli Spazi Vettoriali e Operatori Base
    # --------------------------------------------------------------------------
    # Matrici Identità
    id_fon = np.eye(fon)
    id_el = np.eye(n_siti)
    id_spin = np.eye(2)
    id_el_spin = np.eye((n_siti + 2) * 2) # Include Donatore/Accettore
    id_tot = np.kron(id_el_spin, id_fon) 
    
    # Matrici di Spin
    sigma_z = np.array([[1, 0], [0, -1]])
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_p = np.array([[0, 1], [0, 0]])
    sigma_m = np.array([[0, 0], [1, 0]])
    sigma = np.array([sigma_x, sigma_y, sigma_z])  
    
    # Geometria
    ui = calcola_ui(n_siti)

    # Setup Array Parametri (Topologia 1D: A -> Bridge -> B)
    t_arr = np.array([0] + [t]*(n_siti-1) + [0])
    s_arr = np.array([so]*(n_siti-1))
    #chi_arr = np.array([chi]*(n_siti-1))
    e_arr = np.array([eA] + [eT]*n_siti + [eB])
    #pei_arr = np.array([0] + [t1]*(n_siti-1) + [0])
    zee_arr = np.array([zee]*(n_siti+2))
    
    n_legami = n_siti - 1
    bond_indx = int((n_legami - 1) / 2)
    # Localizzazione del coupling elettrone-fonone al centro del ponte 
    chi_arr = np.zeros(n_legami, dtype=complex)
    pei_arr = np.zeros(n_legami)
    if n_legami > 0:
        chi_arr[bond_indx] = chi
        pei_arr[bond_indx] = t1
    # --------------------------------------------------------------------------
    # 3.3 Assemblaggio Hamiltoniana Totale
    # --------------------------------------------------------------------------
    
    # Operatori di annichilazione fononica
    b = np.zeros((fon, fon))
    for n in range(1, fon):
        b[n-1, n] = np.sqrt(n)
    Q = (b + b.T) * (1/np.sqrt(2))
    
    # 1. H_vib: Hamiltoniana vibrazionale (Oscillatori Armonici Indipendenti)
    h_fon_base = w_o * (b.T @ b) + w_o/2 * id_fon
    h_fon = np.kron(id_el_spin, h_fon_base)

    # 2. H_el: Parte cinetica elettronica (Hopping) + Potenziale (On-site)
    h_t = np.diag(t_arr, k=1) + np.diag(t_arr, k=-1)
    h_t = np.kron(np.kron(h_t, id_spin), id_fon)
    
    h_e = np.diag(e_arr) 
    h_e = np.kron(np.kron(h_e, id_spin), id_fon)

    # 3. H_so: Spin-Orbita Statico (Geometria fissa)
    h_so_base = genera_h_spin_orbita(n_siti, s_arr, ui, sigma, so_vicini)
    h_so = np.kron(h_so_base, id_fon)
    h_so = h_so + h_so.conj().T

    # 4. H_pei: Accoppiamento Elettrone-Fonone (Termine di Peierls)
    # Modula l'hopping in funzione della distorsione del reticolo (b + b_dag)
    h_pei_base = np.diag(pei_arr, k=1) + np.diag(pei_arr, k=-1)
    h_pei = np.kron(np.kron(h_pei_base, id_spin), Q)

    # 5. h_chi: Accoppiamento Spin-Orbita-Fonone
    # Modulazione dinamica del SOC (ordine superiore)
    h_chi_base = genera_h_spin_orbita(n_siti, chi_arr, ui, sigma, so_vicini)
    h_chi = np.kron(h_chi_base, Q)
    h_chi = h_chi + h_chi.conj().T

    # 6. H_zee: Termine Zeeman (Campo magnetico)
    h_zee_base = np.diag(zee_arr, k=0)
    h_zee = np.kron(np.kron(h_zee_base, sigma_z), id_fon)
    
    # 7. Termine Anarmonico
    h_Q = a_Q * np.linalg.matrix_power((b.T + b), 4) 
    h_Q = np.kron(id_el_spin, h_Q)
    


    # SOMMA HAMILTONIANA TOTALE (in unità di frequenza rad/ps)
    H = h_fon - h_t - h_pei + h_e + h_so + h_chi + h_zee + h_Q
    H = H / hbar 
    H  = project_first(fon, n_siti, H)
    evals, evecs = np.linalg.eigh(H)

    # --------------------------------------------------------------------------
    # 3.5 Stato Iniziale e Definizione del Bagno
    # --------------------------------------------------------------------------
    
    # Stato Iniziale: Elettrone localizzato su A (up/down misto)
    rho_el = np.kron(np.diag([0.5] + [0] + [0]*(n_siti)), id_spin) # Matrice densità elettronica
    rho_vib = np.diag([1]+[0]*(fon-1)); rho = np.kron(rho_el, rho_vib) # Matrice densità totale iniziale
    rho = project_M(fon, n_siti, rho, evecs)
    
    # Canale 1: Iniezione (Donatore -> Bridge)
    Transfer_array_A = np.array([1] + [0]*(n_siti-1) + [0])
    Trans_A = np.diag(Transfer_array_A, k=-1) 
    Trans_A = np.kron(np.kron(Trans_A, id_spin), id_fon)
    Trans_A = project_M(fon, n_siti, Trans_A, evecs)
    
    # Canale 2: Estrazione (Bridge -> Accettore)
    Transfer_array_B = np.array([0] + [0]*(n_siti-1) + [1])
    Trans_B = np.diag(Transfer_array_B, k=-1)
    Trans_B = np.kron(np.kron(Trans_B, id_spin), id_fon)
    Trans_B = project_M(fon, n_siti, Trans_B, evecs)

    # Canale 3: Dephasing di Spin
    Spin_dep_z = np.kron(np.kron(np.diag([0] + [1]*n_siti + [0]), sigma_z), id_fon)
    Spin_dep_x = np.kron(np.kron(np.diag([0] + [1]*n_siti + [0]), sigma_x), id_fon)
    Spin_dep_y = np.kron(np.kron(np.diag([0] + [1]*n_siti + [0]), sigma_y), id_fon)
    Spin_dep_p = np.kron(np.kron(np.diag([0] + [1]*n_siti + [0]), sigma_p), id_fon)
    Spin_dep_m = np.kron(np.kron(np.diag([0] + [1]*n_siti + [0]), sigma_m), id_fon)
    Spin_dep_z = project_M(fon, n_siti, Spin_dep_z, evecs)
    Spin_dep_x = project_M(fon, n_siti, Spin_dep_x, evecs)
    Spin_dep_y = project_M(fon, n_siti, Spin_dep_y, evecs)
    Spin_dep_p = project_M(fon, n_siti, Spin_dep_p, evecs)
    Spin_dep_m = project_M(fon, n_siti, Spin_dep_m, evecs)

    # Canale 4: Phonon loss operator
    Phonon_loss = np.kron(np.kron(np.diag( [1]*(n_siti+2) ), id_spin), b)
    Phonon_loss = project_M(fon, n_siti, Phonon_loss, evecs)
    # Triu + diagonale = somma colonna
    
    # Lista operatori di interazione col bagno (corretto metodo adjoint per array NumPy)
    sb_operators = [
        Trans_A,
        Trans_B,
        Spin_dep_z,
        Spin_dep_x,
        Spin_dep_y,
        Phonon_loss,
        # Coniugati (Hermitian Conjugates / Dagger) nello stesso ordine per array Numpy
        Trans_A.conj().T,
        Trans_B.conj().T,
        Spin_dep_z.conj().T,
        Spin_dep_x.conj().T,
        Spin_dep_y.conj().T,
        Phonon_loss.conj().T
    ]
    
    
    # Generazione osservabili da tracciare durante l'evoluzione
    diz_pop = genera_mappa_operatori(n_siti, id_fon, f_pop_n)
    diz_pol = genera_mappa_operatori(n_siti, id_fon, f_mag_n)
    diz_pos = genera_mappa_operatori(n_siti, fon, f_pos_n)
    diz_num = genera_mappa_operatori(n_siti, fon, f_num_n)
    
    e_ops_list, e_labels = genera_suite_osservabili(n_siti, diz_pop, diz_pol, diz_pos, diz_num, id_tot)
    for i, e in enumerate(e_ops_list):
        e_ops_list[i] = project_M(fon, n_siti, e, evecs)
        
    observables = [op for op in e_ops_list]
    rates = [tA, tB, spin_dep_z, spin_dep_xy, spin_dep_xy, p_loss]
    
    # --------------------------------------------------------------------------
    # 3.6 STAMPA INPUT E FILE INFO
    # --------------------------------------------------------------------------
    
    # Definizione directory di output (fittizia per l'esempio)
    output_python = "output_python"
    if not os.path.exists(output_python):
        os.makedirs(output_python)

    with open(f'{output_python}/rates.dat', 'w') as file:
        for iii,rrr in enumerate(rates):
            print(f'{rrr/hbar}', file=file)

    ham = np.diag(evals)
    with open(f'{output_python}/ham.dat', 'w') as file:
        for i in range(len(evals)):
            for j in range(len(evals)):
                print(f'{ham[i,j].real} {ham[i,j].imag}', file=file)

    with open(f'{output_python}/rho.dat', 'w') as file:
        for i in range(len(evals)):
            for j in range(len(evals)):
                print(f'{rho[i,j].real} {rho[i,j].imag}', file=file)

    with open(f'{output_python}/sb.dat', 'w') as file:
        for op in sb_operators:
            for i in range(len(evals)):
                for j in range(len(evals)):
                    print(f'{op[i,j].real} {op[i,j].imag}', file=file)

    with open(f'{output_python}/obs.dat', 'w') as file:
        for op in observables:
            for i in range(len(evals)):
                for j in range(len(evals)):
                    print(f'{op[i,j].real} {op[i,j].imag}', file=file)

    times = tlist # Assegno tlist a times per il print nel file di input
    with open(f'input_dynamics.inp', 'w') as file:
        print(f'{len(evals)} rho.bin ham.bin {int(len(sb_operators)/2)} sb.bin rates.bin {len(observables)} obs.bin {int(times[-1])} {len(times)} {int(len(times)/10)} output.dat', file=file)    
    
    counter = 1
    while True:
        # L'aggiunta della 'f' prima delle virgolette abilita l'interpolazione della variabile
        info_filename = f"DATI/info_{counter}.txt"
        data_filename = f"DATI/data_{counter}.dat"
        if not os.path.exists(info_filename):
            break
        counter += 1
        
    # Salvataggio Serie Temporali
    header_str = "Time " + " ".join(e_labels)

    # Questo assicura che savetxt non trovi dati da scrivere dopo l'header
    empty_matrix = np.array([]).reshape(0, len(e_labels) + 1)

    np.savetxt(
        data_filename, 
        empty_matrix, 
        header=header_str, 
        fmt='%.8e', 
        comments=''
    )


    # Salvataggio Metadati
    with open(info_filename, 'w') as f:
        f.write("PARAMETRI DI SIMULAZIONE \n")
        f.write("=====================================================\n\n")
        
        f.write("[Topologia e Hamiltoniana]\n")
        f.write(f"Velocità (SOC driving term) = {ui[0]}\n")
        f.write(f"N_siti (Bridge)      = {n_siti}\n")
        f.write(f"Fononi (Mode)        = {fon} \n")
        f.write(f"Omega_0 (Freq)       = {w_o}\n\n")
        
        f.write("[Parametri Bridge]\n")
        f.write(f"t (Hopping)          = {t} [eV]\n")
        f.write(f"t1 (Peierls)         = {t1} [eV]\n")
        f.write(f"eT (Energy Site)     = {eT} [eV]\n")
        f.write(f"so (Spin-Orbit)      = {so} [eV]\n")
        f.write(f"so_vicini (Range)    = {so_vicini} (1=NN, 2=NNN)\n")
        f.write(f"so_co (Coupling)     = {chi}\n\n")
        
        f.write("[Parametri Donatore - Accettore]\n")
        f.write(f"tA (Hopping D-B)     = {tA} [eV]\n")
        f.write(f"tB (Hopping B-A)     = {tB} [eV]\n\n")

        f.write("[Parametri Dephasing]\n")
        f.write(f"Spin Dephasing z     = {spin_dep_z} [eV]\n")
        f.write(f"Spin Dephasing xy    = {spin_dep_xy} [eV]\n")
        f.write(f"Phonon Loss     = {p_loss} [eV]\n\n")


        f.write("[Dinamica Temporale]\n")
        f.write(f"t_max                = {t_max} [ps]\n")
        f.write(f"t_int (Points)       = {t_int}\n")


# ==============================================================================
# SEZIONE 4: ESECUZIONE DEL CICLO DI SIMULAZIONE
# ==============================================================================

if __name__ == "__main__":

    dep = float(sys.argv[1])
    tra = float(sys.argv[2])
    pL = float(sys.argv[3])
    n_siti = int(sys.argv[4])

    simulazione_tight_binding(dep,tra,pL,n_siti)
