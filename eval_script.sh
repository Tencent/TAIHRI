export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_ATTENTION_BACKEND=FLASHINFER

python3 eval/eval_script.py \
--input_path_dir ./data/harmony4d/data_center \
--output_path outputs \
--model_path checkpoints/taihri \
--focal_length YOUR_FOCAL_LENGTH \
--test_dataset harmony \
--backend transformers