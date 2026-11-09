import numpy as np
from pyscf import gto, scf, lib

# =========================================================================
# 1. INIZIALIZZAZIONE SISTEMA MOLECOLARE E CALCOLO UKS
# =========================================================================

# Lettura della geometria e definizione del set di base
mol = gto.M(
    atom='Molecola.xyz',
    unit='Angstrom',
    basis='ma-def2-svp',  # Set di base Minimally Augmented def2-SVP
    charge=-1,
    spin=1
)

# Risoluzione Autocoerente dell'Hamiltoniana Non Perturbata (UKS)
mf = scf.UKS(mol)

# Impostazione del funzionale e della correzione di dispersione
mf.xc = 'b3lyp'
mf.disp = 'd3bj'

# Implementazione dell'approssimazione RI (Density Fitting) con base ausiliaria def2/J
mf = mf.density_fit(auxbasis='def2-universal-jfit')

mf.verbose = 0
mf.kernel()

if not mf.converged:
    print("ATTENZIONE: Convergenza SCF fallita.")

# Estrazione degli autostati per i canali Alpha e Beta
C_alpha = mf.mo_coeff[0]
mo_energy_alpha = mf.mo_energy[0]
homo_idx_alpha = mol.nelec[0] - 1 

C_beta = mf.mo_coeff[1]
mo_energy_beta = mf.mo_energy[1]

S_ao = mol.intor('int1e_ovlp')
n_ao = mol.nao


# =========================================================================
# 2. VALUTAZIONE DELL'OPERATORE SPIN-ORBITA (BARE-NUCLEUS, AO BASIS)
# =========================================================================

c_light = lib.param.LIGHT_SPEED 
soc_prefactor = 1.0 / (2.0 * c_light**2)

h_so_ao = np.zeros((3, n_ao, n_ao), dtype=complex)

for A in range(mol.natm):
    Z_A = mol.atom_charge(A)
    R_A = mol.atom_coord(A)
    mol.set_rinv_origin(R_A)
    
    # Valutazione integrale spaziale dell'operatore L
    # Nota rigorosa: +1j è necessario per la convenzione di segno di PySCF 
    # sull'integrale del gradiente incrociato (int1e_prinvxp_sph)
    integrals_rxnabla = mol.intor('int1e_prinvxp', comp=3)
    h_so_ao += Z_A * (1j) * integrals_rxnabla

h_so_ao *= soc_prefactor


# =========================================================================
# 3. TEORIA DELLE PERTURBAZIONI: COEFFICIENTI DI ADMIXTURE
# =========================================================================

def compute_admixture_coefficients_uks(
    C_alpha, mo_energy_alpha, homo_idx, 
    C_beta, mo_energy_beta, 
    h_so_ao_x, h_so_ao_y, h_so_ao_z
):
    """
    Calcola i coefficienti di mixing perturbativo al primo ordine per uno stato 
    radicale nel formalismo UKS. Scompone rigorosamente il contributo in:
    - a: Mixing orbitale (conservazione dello spin, accoppiamento alpha-alpha)
    - b: Mixing di spin (spin-flip, accoppiamento alpha-beta)
    """
    n_ao = C_alpha.shape[0]
    a = np.zeros(n_ao, dtype=complex)
    b = np.zeros(n_ao, dtype=complex)
    
    E_0 = mo_energy_alpha[homo_idx]
    c_homo_alpha = C_alpha[:, homo_idx] 
    
    eta_smearing = 1e-5
    
    # Termine 'a': Accoppiamento Alpha-Alpha governato dall'operatore L_z
    for k in range(homo_idx_alpha):
        if k == homo_idx:
            continue
            
        delta_E_norm = mo_energy_alpha[k] - E_0
        delta_E = (delta_E_norm**2 + eta_smearing**2) / delta_E_norm
        
        c_k_alpha = C_alpha[:, k] 
        mz = np.einsum('i,ij,j->', np.conj(c_k_alpha), h_so_ao_z, c_homo_alpha)
        
        a -= 0.5 * (mz / delta_E) * c_k_alpha

    # Termine 'b': Accoppiamento Beta-Alpha governato dall'operatore L_x + iL_y
    for q in range(homo_idx_alpha):
        delta_E_norm = mo_energy_beta[q] - E_0
        delta_E = (delta_E_norm**2 + eta_smearing**2) / delta_E_norm
        
        c_q_beta = C_beta[:, q]
        
        mx = np.einsum('i,ij,j->', np.conj(c_q_beta), h_so_ao_x, c_homo_alpha)
        my = np.einsum('i,ij,j->', np.conj(c_q_beta), h_so_ao_y, c_homo_alpha)
        mxy = mx + 1j * my
        
        b -= 0.5 * (mxy / delta_E) * c_q_beta
            
    return a, b

a_alpha, b_alpha = compute_admixture_coefficients_uks(
    C_alpha, mo_energy_alpha, homo_idx_alpha, 
    C_beta, mo_energy_beta,
    h_so_ao[0], h_so_ao[1], h_so_ao[2]
)


# =========================================================================
# 4. CALCOLO DEGLI INVARIANTI FISICI
# =========================================================================

def compute_gamma2_components(a, b, S):
    """
    Valuta la norma delle perturbazioni proiettandole sulla metrica dello 
    spazio di Hilbert non ortogonale (matrice di sovrapposizione S).
    """
    term_a = np.einsum('i,ij,j->', np.conj(a), S, a)
    term_b = np.einsum('i,ij,j->', np.conj(b), S, b)
    return np.real(term_a), np.real(term_b)

gamma2_up_up, gamma2_up_down = compute_gamma2_components(a_alpha, b_alpha, S_ao)
gamma2_tot = gamma2_up_up + gamma2_up_down


# =========================================================================
# 5. OUTPUT E SCRITTURA SU FILE
# =========================================================================

output_filename = "risultati_soc_uks.txt"

output_content = "="*75 + "\n"
output_content += "ANALISI PERTURBATIVA DELL'ADMIXTURE SPIN-ORBITA (UKS FORMALISM)\n"
output_content += "="*75 + "\n"
output_content += f"Funzionale: B3LYP-D3BJ | Base: ma-def2-SVP | RI-J: def2-universal-jfit\n"
output_content += "-" * 75 + "\n"
output_content += f"Mixing Orbitale (γ²_up_up):    {gamma2_up_up:.6e}\n"
output_content += f"Mixing di Spin  (γ²_up_down): {gamma2_up_down:.6e}\n"
output_content += "-" * 75 + "\n"
output_content += f"Parametro SOC Totale (γ²):     {gamma2_tot:.6e}\n"
output_content += "="*75 + "\n"

# Stampa a schermo
print(output_content)

# Scrittura su file di testo
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(output_content)

print(f"Risultati salvati correttamente nel file: {output_filename}")O
