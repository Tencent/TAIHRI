export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_ATTENTION_BACKEND=FLASHINFER

python3 demo/simple_demo_kpt2d_3d_mllm.py \
--input_path YOUR_TEST_IMAGES_DIR \
--output_path outputs/demo \
--model_path checkpoints/taihri \
--focal_length YOUR_FOCAL_LENGTH \
--princpt_x YOUR_PRINCPT_X \
--princpt_y YOUR_PRINCPT_Y \
--prompts "Could you tell me the coordinates of right wrist and left wrist?" \
--backend transformers