#!/bin/bash

# Tạo thư mục log nếu chưa có
mkdir -p detect_log

echo "=========================================================="
echo " BẮT ĐẦU CHẠY DECREE CHO 3 MÔ HÌNH (TỐI ƯU CPU - BATCH 16)"
echo "=========================================================="

# 1. Chạy mô hình GTSRB (Backdoor)
# 2. Chạy mô hình STL-10 (Backdoor)
echo "[1/3] Đang quét mô hình bệnh STL-10 (model_17)..."
python3 -u main.py \
    --gpu cpu \
    --model_flag backdoor \
    --batch_size 16 \
    --lr 0.5 \
    --seed 80 \
    --encoder_path output/CLIP_text/stl10_backdoored_encoder/model_17_tg24_imagenet.pth \
    --mask_init rand \
    --id _CLIP_text_stl10_model_17 \
    --encoder_usage_info CLIP \
    --arch resnet50 \
    --result_file resultfinal_cliptxt_stl10.txt \
    > detect_log/cf10_80_backdoor_lr0.5_b16_rand_CLIP_text_stl10_model_17.log
echo "-> Xong mô hình 2!"

# 3. Chạy mô hình Sạch (Clean)
echo "[3/3] Đang quét mô hình Sạch (Clean)..."
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
echo "-> Xong mô hình 3!"

echo "=========================================================="
echo " ĐÃ HOÀN TẤT TOÀN BỘ QUÁ TRÌNH!"
echo "=========================================================="
