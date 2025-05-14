import os
import glob
import pandas as pd

records = []

# Para cada cenário que tenha um arquivo Moroder2
for m2_path in glob.glob('*_Moroder2.txt'):
    prefix = m2_path[:-len('_Moroder2.txt')]
    scenario = prefix

    # caminhos dos arquivos
    switch_path   = f'{prefix}_NLSwitch.txt'
    local_path    = f'{prefix}_L.txt'
    m1_path       = f'{prefix}_Moroder1.txt'
    m2_path       = f'{prefix}_Moroder2.txt'

    # só processa se existir NLSwitch e Moroder1
    if not (os.path.exists(switch_path) and os.path.exists(m1_path)):
        continue

    # lê desigualdades
    with open(switch_path) as f:
        inequalities = [l.strip() for l in f if l.strip()]

    # lê bounds locais
    with open(local_path) as f:
        locals_ = [float(l.strip()) for l in f if l.strip()]

    # lê Moroder nível 2
    with open(m2_path) as f:
        m2 = [float(l.strip()) for l in f if l.strip()]

    # lê Moroder nível 1
    with open(m1_path) as f:
        m1 = [float(l.strip()) for l in f if l.strip()]

    # junta tudo
    for ineq, bl, v2, v1 in zip(inequalities, locals_, m2, m1):
        violation   = v2 - bl
        # só inclui se violação >= 1e-6
        if violation < 1e-6:
            continue

        convergence = (10 - v1 + v2) / 10 if v1 != 0 else float('nan')

        records.append({
            'scenario'       : scenario,
            'inequality'     : ineq,
            'moroder1_value' : v1,
            'moroder_value' : v2,
            'bound_local'    : bl,
            'violation'      : violation,
            'convergence'    : convergence,
        })

# monta DataFrame, ordena e salva
df = pd.DataFrame(records)
df = df.sort_values('convergence', ascending=False)  # valores menores primeiro
df.to_csv('consolidated_results.csv', index=False)

print("Gerado: consolidated_results.csv")
