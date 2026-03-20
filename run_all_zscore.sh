#!/bin/bash

mkdir -p vallog/clip_txt

for f in output/CLIP_text/clean_encoder/*.pth; do
  echo "Running $f"
  python -u compute_zscore.py \
    --batch_size 64 \
    --id "$(basename "$f")" \
    --encoder_path "$f" \
    --res_file valid_cliptxt_zscore.txt \
    > "vallog/clip_txt/$(basename "$f").log" 2>&1

  echo "Finished $f"
done

# for f in output/CLIP_text/gtsrb_backdoored_encoder/*.pth; do
#   echo "Running $f"
#   python -u compute_zscore.py \
#     --batch_size 64 \
#     --id "$(basename "$f")" \
#     --encoder_path "$f" \
#     --res_file valid_cliptxt_zscore.txt \
#     > "vallog/clip_txt/$(basename "$f").log" 2>&1

#   echo "Finished $f"
# done

# for f in output/CLIP_text/stl10_backdoored_encoder/*.pth; do
#   echo "Running $f"
#   python -u compute_zscore.py \
#     --batch_size 64 \
#     --id "$(basename "$f")" \
#     --encoder_path "$f" \
#     --res_file valid_cliptxt_zscore.txt \
#     > "vallog/clip_txt/$(basename "$f").log" 2>&1

#   echo "Finished $f"
# done

# for f in output/CLIP_text/svhn_backdoored_encoder/*.pth; do
#   echo "Running $f"
#   python -u compute_zscore.py \
#     --batch_size 64 \
#     --id "$(basename "$f")" \
#     --encoder_path "$f" \
#     --res_file valid_cliptxt_zscore.txt \
#     > "vallog/clip_txt/$(basename "$f").log" 2>&1

#   echo "Finished $f"
# done
