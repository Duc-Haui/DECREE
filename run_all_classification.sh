#!/bin/bash

# Cấu hình chung
GPU="cpu" 
TRIGGER="trigger/trigger_pt_white_185_24.npz"
EPOCHS=100  # Chạy full 200 epochs 
BATCH_SIZE=16 # BẮT BUỘC GIẢM BATCH SIZE XUỐNG ĐỂ KHÔNG BỊ TRÀN RAM (KILLED)

# Hàm chạy tự động cho một tập dữ liệu
run_evaluation() {
    DATASET=$1
    REF_LABEL=$2
    REF_FILE=$3
    ENCODER_DIR=$4

    echo "=================================================="
    echo " BẮT ĐẦU CHẠY THỬ TRÊN TẬP DỮ LIỆU: $DATASET"
    echo "=================================================="

    # 1. Chạy đánh giá cho Clean Encoder (Mô hình sạch)
    echo "[*] Đang huấn luyện Classifier cho: MÔ HÌNH SẠCH (Clean Encoder)"
    python training_downstream_classifier.py \
        --dataset "$DATASET" \
        --reference_label "$REF_LABEL" \
        --trigger_file "$TRIGGER" \
        --reference_file "$REF_FILE" \
        --encoder_usage_info CLIP \
        --encoder output/CLIP_text/clean_encoder/clean_ft_imagenet.pth \
        --gpu "$GPU" \
        --nn_epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --result_dir "output/classifier_${DATASET}_clean_test.pth"

    # 2. Quét qua MỘT mô hình bệnh duy nhất
    for ENCODER_PATH in "$ENCODER_DIR"/*.pth; do
        MODEL_NAME=$(basename "$ENCODER_PATH")
        
        echo "[*] Đang huấn luyện Classifier cho: MÔ HÌNH BỆNH ($MODEL_NAME)"
        python training_downstream_classifier.py \
            --dataset "$DATASET" \
            --reference_label "$REF_LABEL" \
            --trigger_file "$TRIGGER" \
            --reference_file "$REF_FILE" \
            --encoder_usage_info CLIP \
            --encoder "$ENCODER_PATH" \
            --gpu "$GPU" \
            --nn_epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --result_dir "output/classifier_${DATASET}_${MODEL_NAME}_test.pth"
        
        # DỪNG VÒNG LẶP NGAY SAU KHI CHẠY XONG 1 FILE
        break 
    done
}

# --- THỰC THI (CHẠY GTSRB) ---
run_evaluation "gtsrb" 12 "reference/CLIP/priority.npz" "output/CLIP_text/gtsrb_backdoored_encoder"

echo "ĐÃ HOÀN TẤT!"