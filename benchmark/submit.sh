#!/bin/bash
nvidia-smi


# depending on your system, you might have to export CUDA paths to make KeOps work
# export PATH=/home/platforms/x86_64/appl/cuda/11.7/bin:$PATH
# export LD_LIBRARY_PATH=/home/platforms/x86_64/appl/cuda/11.7/lib64:$LD_LIBRARY_PATH
# export LIBRARY_PATH=/home/platforms/x86_64/appl/cuda/11.7/lib64:$LIBRARY_PATH

bsize=100
last_round=10
run_dir='runs'


for function in "rosenbrock" "ackley" "rastrigin" "levy" "styblinskitang" "powell"
do 


for dim in 2 10 20 50 100
do

# skip dim=2 for powell
if [ $function == "powell" ] && [ $dim == 2 ]
then
    continue
fi

for seed in 0 1 2 3 4
do


    fn="qucb"
    stdbuf -o0 -e0 python3 multi_round_experiment.py --test_function $function --dim $dim --n_start_points $bsize --batch_size $bsize --acq_fn $fn --opt torch --keops --n_rounds 10 --explore_parameter 0.1 --seed $seed
    stdbuf -o0 -e0 python3 multi_round_experiment.py --test_function $function --dim $dim --n_start_points $bsize --batch_size $bsize --acq_fn $fn --opt torch --keops --n_rounds 10 --explore_parameter 1.0 --seed $seed
    stdbuf -o0 -e0 python3 multi_round_experiment.py --test_function $function --dim $dim --n_start_points $bsize --batch_size $bsize --acq_fn $fn --opt torch --keops --n_rounds 10 --explore_parameter 10.0 --seed $seed
    

    python3 redo_last_round_full_exploit.py --out_dir ${run_dir}_${seed}/${function}${dim}_q${bsize}/${fn}_explore_parameter0.1 --round_to_rerun $last_round
    python3 redo_last_round_full_exploit.py --out_dir ${run_dir}_${seed}/${function}${dim}_q${bsize}/${fn}_explore_parameter1.0 --round_to_rerun $last_round
    python3 redo_last_round_full_exploit.py --out_dir ${run_dir}_${seed}/${function}${dim}_q${bsize}/${fn}_explore_parameter10.0 --round_to_rerun $last_round


    fn="beebo"
    stdbuf -o0 -e0 python3 multi_round_experiment.py --test_function $function --dim $dim --n_start_points $bsize --batch_size $bsize --acq_fn $fn --opt torch --keops --n_rounds 10 --explore_parameter 0.05 --seed $seed
    stdbuf -o0 -e0 python3 multi_round_experiment.py --test_function $function --dim $dim --n_start_points $bsize --batch_size $bsize --acq_fn $fn --opt torch --keops --n_rounds 10 --explore_parameter 0.5 --seed $seed
    stdbuf -o0 -e0 python3 multi_round_experiment.py --test_function $function --dim $dim --n_start_points $bsize --batch_size $bsize --acq_fn $fn --opt torch --keops --n_rounds 10 --explore_parameter 5.0 --seed $seed


    python3 redo_last_round_full_exploit.py --out_dir ${run_dir}_${seed}/${function}${dim}_q${bsize}/${fn}_explore_parameter0.05 --round_to_rerun $last_round
    python3 redo_last_round_full_exploit.py --out_dir ${run_dir}_${seed}/${function}${dim}_q${bsize}/${fn}_explore_parameter0.5 --round_to_rerun $last_round
    python3 redo_last_round_full_exploit.py --out_dir ${run_dir}_${seed}/${function}${dim}_q${bsize}/${fn}_explore_parameter5.0 --round_to_rerun $last_round




    fn="qei"
    stdbuf -o0 -e0 python3 multi_round_experiment.py --test_function $function --dim $dim --n_start_points $bsize --batch_size $bsize --acq_fn $fn --opt torch --keops --n_rounds 10 --explore_parameter 0.01 --seed $seed



    fn="thompson"
    stdbuf -o0 -e0 python3 multi_round_experiment.py --test_function $function --dim $dim --n_start_points $bsize --batch_size $bsize --acq_fn $fn --opt torch --keops --n_rounds 10 --explore_parameter 0.05 --seed $seed --n_thompson_base_samples 10000


    fn="random"
    stdbuf -o0 -e0 python3 multi_round_experiment.py --test_function $function --dim $dim --n_start_points $bsize --batch_size $bsize --acq_fn $fn --opt torch --keops --n_rounds 10 --explore_parameter 0.0 --seed $seed




done
done
done


# zipped iteration: function:dim shekel:4, hartmann:6, cosine:8
functions=(shekel hartmann cosine embeddedhartmann)
dims=(4 6 8 100)
for ((i = 0; i < ${#functions[@]}; ++i)); do

    function=${functions[$i]}
    dim=${dims[$i]}

    for seed in 0 1 2 3 4
    do


    fn="qucb"
    stdbuf -o0 -e0 python3 multi_round_experiment.py --test_function $function --dim $dim --n_start_points $bsize --batch_size $bsize --acq_fn $fn --opt torch --keops --n_rounds 10 --explore_parameter 0.1 --seed $seed
    stdbuf -o0 -e0 python3 multi_round_experiment.py --test_function $function --dim $dim --n_start_points $bsize --batch_size $bsize --acq_fn $fn --opt torch --keops --n_rounds 10 --explore_parameter 1.0 --seed $seed
    stdbuf -o0 -e0 python3 multi_round_experiment.py --test_function $function --dim $dim --n_start_points $bsize --batch_size $bsize --acq_fn $fn --opt torch --keops --n_rounds 10 --explore_parameter 10.0 --seed $seed
    

    python3 redo_last_round_full_exploit.py --out_dir ${run_dir}_${seed}/${function}${dim}_q${bsize}/${fn}_explore_parameter0.1 --round_to_rerun $last_round
    python3 redo_last_round_full_exploit.py --out_dir ${run_dir}_${seed}/${function}${dim}_q${bsize}/${fn}_explore_parameter1.0 --round_to_rerun $last_round
    python3 redo_last_round_full_exploit.py --out_dir ${run_dir}_${seed}/${function}${dim}_q${bsize}/${fn}_explore_parameter10.0 --round_to_rerun $last_round



    fn="beebo"
    stdbuf -o0 -e0 python3 multi_round_experiment.py --test_function $function --dim $dim --n_start_points $bsize --batch_size $bsize --acq_fn $fn --opt torch --keops --n_rounds 10 --explore_parameter 0.05 --seed $seed
    stdbuf -o0 -e0 python3 multi_round_experiment.py --test_function $function --dim $dim --n_start_points $bsize --batch_size $bsize --acq_fn $fn --opt torch --keops --n_rounds 10 --explore_parameter 0.5 --seed $seed
    stdbuf -o0 -e0 python3 multi_round_experiment.py --test_function $function --dim $dim --n_start_points $bsize --batch_size $bsize --acq_fn $fn --opt torch --keops --n_rounds 10 --explore_parameter 5.0 --seed $seed


    python3 redo_last_round_full_exploit.py --out_dir ${run_dir}_${seed}/${function}${dim}_q${bsize}/${fn}_explore_parameter0.05 --round_to_rerun $last_round
    python3 redo_last_round_full_exploit.py --out_dir ${run_dir}_${seed}/${function}${dim}_q${bsize}/${fn}_explore_parameter0.5 --round_to_rerun $last_round
    python3 redo_last_round_full_exploit.py --out_dir ${run_dir}_${seed}/${function}${dim}_q${bsize}/${fn}_explore_parameter5.0 --round_to_rerun $last_round




    fn="qei"
    stdbuf -o0 -e0 python3 multi_round_experiment.py --test_function $function --dim $dim --n_start_points $bsize --batch_size $bsize --acq_fn $fn --opt torch --keops --n_rounds 10 --explore_parameter 0.01 --seed $seed



    fn="thompson"
    stdbuf -o0 -e0 python3 multi_round_experiment.py --test_function $function --dim $dim --n_start_points $bsize --batch_size $bsize --acq_fn $fn --opt torch --keops --n_rounds 10 --explore_parameter 0.05 --seed $seed --n_thompson_base_samples 10000


    fn="random"
    stdbuf -o0 -e0 python3 multi_round_experiment.py --test_function $function --dim $dim --n_start_points $bsize --batch_size $bsize --acq_fn $fn --opt torch --keops --n_rounds 10 --explore_parameter 0.0 --seed $seed


    done
done



