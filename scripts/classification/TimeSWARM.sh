#!/bin/bash
model_name="TimeSWARM"
gpu_id=0
data_root=""
note=""

run_model() {
  local dataset=$1

  python -u run.py \
    --use_gpu --gpu_type cuda --gpu ${gpu_id} \
    --task_name classification \
    --is_training 1 \
    --itr 1 \
    --data UEA \
    --root_path "${data_root}/${dataset}/" \
    --model_id "${dataset}" \
    --model "${model_name}" \
    --train_epochs 100 \
    --patience 10 \
    --batch_size 16 \
    --learning_rate 0.01 \
    --num_workers 0 \
    --note "${note}"
}

# =========================
# UEA Datasets
# =========================

# run_model "ArticularyWordRecognition"
# run_model "AtrialFibrillation"
run_model "BasicMotions"
# run_model "CharacterTrajectories"
# run_model "Cricket"
# run_model "DuckDuckGeese"
# run_model "EigenWorms"
# run_model "Epilepsy"
# run_model "ERing"
# run_model "EthanolConcentration"
# run_model "FaceDetection"
# run_model "FingerMovements"
# run_model "HandMovementDirection"
# run_model "Handwriting"
# run_model "Heartbeat"
# run_model "InsectWingbeat"
# run_model "JapaneseVowels"
# run_model "Libras"
# run_model "LSST"
# run_model "MotorImagery"
# run_model "NATOPS"
# run_model "PEMS-SF"
# run_model "PenDigits"
# run_model "PhonemeSpectra"
# run_model "RacketSports"
# run_model "SelfRegulationSCP1"
# run_model "SelfRegulationSCP2"
# run_model "SpokenArabicDigits"
# run_model "StandWalkJump"
# run_model "UWaveGestureLibrary"