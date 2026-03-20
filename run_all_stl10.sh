#!/bin/bash

# Tạo thư mục log nếu chưa có
mkdir -p detect_log

echo "=========================================================="
echo " BẮT ĐẦU CHẠY DECREE CHO TOÀN BỘ STL-10 (CPU - BATCH 16)"
echo "=========================================================="

# 1. Chạy mô hình Sạch (Clean) trước để làm mốc (Baseline)
echo "[*] Đang quét mô hình Sạch (Clean)..."
python3 -u main.py \
    --gpu cpu \
    --model_flag clean \
    --batch_size 16 \
    --lr 0.5 \
    --seed 80 \
    --encoder_path output/CLIP_text/clean_encoder/clean_ft_imagenet.pth \
    --mask_init rand \
    --id _CLIP_text_clean_ft \
    --encoder_usage_info CLIP \
    --arch resnet50 \
    --result_file resultfinal_cliptxt_clean.txt \
    > detect_log/cf10_80_clean_lr0.5_b16_rand_CLIP_text_clean_ft.log
echo "-> Xong mô hình Clean!"

# 2. Vòng lặp quét TOÀN BỘ mô hình bệnh STL-10
STL10_DIR="output/CLIP_text/stl10_backdoored_encoder"
COUNT=1

for ENCODER_PATH in "$STL10_DIR"/*.pth; do
    # Lấy tên file (vd: model_10_tg24_imagenet.pth) và bỏ đuôi .pth đi
    MODEL_NAME=$(basename "$ENCODER_PATH")
    MODEL_BASE="${MODEL_NAME%.pth}"

    echo "[*] Đang quét mô hình bệnh STL-10 số $COUNT: $MODEL_NAME..."
    
    python3 -u main.py \
        --gpu cpu \
        --model_flag backdoor \
        --batch_size 16 \
        --lr 0.5 \
        --seed 80 \
        --encoder_path "$ENCODER_PATH" \
        --mask_init rand \
        --id "_CLIP_text_stl10_${MODEL_BASE}" \
        --encoder_usage_info CLIP \
        --arch resnet50 \
        --result_file resultfinal_cliptxt_stl10.txt \
        > "detect_log/cf10_80_backdoor_lr0.5_b16_rand_CLIP_text_stl10_${MODEL_BASE}.log"
        
    echo "-> Xong $MODEL_NAME!"
    COUNT=$((COUNT + 1))
done

echo "=========================================================="
echo " ĐÃ HOÀN TẤT TOÀN BỘ QUÁ TRÌNH QUÉT STL-10!"
echo "=========================================================="
