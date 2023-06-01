import os


if os.getcwd() != os.path.dirname(os.path.realpath(__file__)):
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    print("Changed working directory to script directory")

if not os.path.exists('figures'):
    os.mkdir('figures')
    print("Created figures directory")

if not os.path.exists('figures/time_complexity'):
    os.mkdir('figures/time_complexity')
    print("Created figures/time_complexity directory")

if not os.path.exists('figures/FHN'):
    os.mkdir('figures/FHN')
    print("Created figures/FHN directory")

if not os.path.exists('figures/HR'):
    os.mkdir('figures/HR')
    print("Created figures/HR directory")

if not os.path.exists('figures/RPS'):
    os.mkdir('figures/RPS')
    print("Created figures/RPS directory")

if not os.path.exists('figures/RPS_ND'):
    os.mkdir('figures/RPS_ND')
    print("Created figures/RPS_ND directory")

if not os.path.exists('figures/SSN'):
    os.mkdir('figures/SSN')
    print("Created figures/SSN directory")

if not os.path.exists('figures/WC4D'):
    os.mkdir('figures/WC4D')
    print("Created figures/WC4D directory")

