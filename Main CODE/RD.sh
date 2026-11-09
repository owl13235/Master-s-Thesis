#!/bin/bash

# Manda mail
trap 'curl -d "La patata è germogliata alle ore $(date +%H:%M:%S)" ntfy.sh/notifiche_ssh_domenico_2026' EXIT
# ==============================================================================
# 1. SETUP DELL'AMBIENTE E DELLE DIRECTORY
# ==============================================================================
OUTDIR='DATI'

mkdir -p "$OUTDIR"
mkdir -p output_python

# Ottimizzazione multithreading
export OMP_NUM_THREADS=64
export KMP_AFFINITY=granularity=fine,compact,1,0

# Caricamenti moduli e ambiente
module load intel/oneapi/2025.2
source ~/virtual_env/bin/activate

# ==============================================================================
# 2. ESECUZIONE CICLO FONONI
# ==============================================================================
# In Bash i vettori si definiscono con le parentesi tonde e spazi
#!/bin/bash

# Compilazione Fortran spostata fuori dai cicli per efficienza
gfortran generate_bins.f90 -o generate_bins.e

# Definizione parametri
dep=($(awk 'BEGIN { 
    for(x=2e-5; x<=1e-4 + 1e-9; x+=1e-5) printf "%g ", x; 
    for(x=1e-6; x<=1e-5 + 1e-9; x+=1e-6) printf "%g ", x 
}'))
tra=($(awk 'BEGIN { for(x=1e-4; x<=2e-3 + 1e-9; x+=2e-4) printf "%g ", x }'))
pL=(0)
n_siti=(8)

echo -e "\n >>> AVVIO SIMULAZIONE CON TRONCAMENTO FONONICO: 15"

for i in "${dep[@]}"; do
    echo "---  relaxation rate: $i ---"

    for j in "${tra[@]}"; do
        echo "--- transfer rate: $j ---"

         for k in "${pL[@]}"; do
            echo "--- phonon loss rate: $k ---"

            for l in "${n_siti[@]}"; do
              echo "--- Numero siti bridge: $l ---"


                # Passaggio delle variabili come due argomenti distinti
                python DI.py "$i" "$j" "$k" "$l"

                # Esecuzione Fortran
                ./generate_bins.e

                # Lettura parametri dinamica
                read -a args < input_dynamics.inp

                echo "Dynamics START"
                solver_ode_V3 "${args[@]}"
                echo "Dynamics END"

                # Post-processing
                python convert_oss.py

            done #Chiusura ciclo l
        done #Chiusura ciclo k
    done # Chiusura ciclo j
done # Chiusura ciclo i


    # Pulizia e spostamento dati per ogni iterazione (per evitare sovrascritture)
    # Creiamo una sottocartella specifica per ogni fonone dentro DATI
    rm -f *.bin generate_bins.e input_dynamics.inp final_results.txt output.dat
    mv *.dat "$OUTDIR" 2>/dev/null

echo "Tutti i processi completati."
