for i in {0..32}; do
    ./run.sh hwfc/experiments/all_exps.py maze WFC $i 32 2>&1 | tee logs/all_exps_$i &
    ./run.sh hwfc/experiments/all_exps.py maze MWFC $i 32 2>&1 | tee logs/all_exps_$i &
    ./run.sh hwfc/experiments/all_exps.py maze HWFC $i 32 2>&1 | tee logs/all_exps_$i &
    
    ./run.sh hwfc/experiments/all_exps.py 2.5D WFC $i 32 2>&1 | tee logs/all_exps_$i &
    ./run.sh hwfc/experiments/all_exps.py 2.5D MWFC $i 32 2>&1 | tee logs/all_exps_$i &
    ./run.sh hwfc/experiments/all_exps.py 2.5D HWFC $i 32 2>&1 | tee logs/all_exps_$i &
    wait;
done