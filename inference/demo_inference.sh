export CUDA_VISIBLE_DEVICES=7

python inference.py \
    --base-model "Qwen/Qwen-Image-Edit-2511" \
    --lora-adapter-name "default" \
    --lora-model "pretrained_weights/qwenimage_mask_flow_demo.safetensors" \
    --prompt "Modify the object masked by image 2 to white." \
    --negative-prompt "" \
    --source-image "demo/cat.png" \
    --mask-image "demo/mask.png" \
    --enable-mask-blur \
    --blur-kernel 75 \
    --blur-sigma 15.0 \
    --enable-mask-dilation \
    --dilation-kernel 75 \
    --save-dir "inference-outputs" \
    --num-inference-steps 50 \