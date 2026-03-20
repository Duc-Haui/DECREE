import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# --- CẤU HÌNH ---
IMAGE_PIXELS = 224 * 224  # 50,176 pixels
THRESHOLD = 0.1

# Phân loại file để đọc
FILE_MAP = {
    'clean': 'resultfinal_cliptxt_clean.txt',
    'gtsrb': 'resultfinal_cliptxt_gtsrb.txt',
    'stl10': 'resultfinal_cliptxt_stl10.txt'
}

def extract_data():
    data = {'clean': [], 'gtsrb': [], 'stl10': []}
    
    for category, filename in FILE_MAP.items():
        if os.path.exists(filename):
            print(f"[*] Đang nạp dữ liệu từ: {filename} ({category.upper()})")
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        # Lấy giá trị L1-norm thô ở cột thứ 4 (index 3)
                        raw_l1 = float(parts[3].strip())
                        # Chuyển đổi sang PL1-norm (tỷ lệ phần trăm pixel)
                        pl1_norm = raw_l1 / IMAGE_PIXELS
                        data[category].append(pl1_norm)
        else:
            print(f"[!] Bỏ qua {filename} vì không tìm thấy.")
            
    return data

def plot_distribution(data):
    """Vẽ biểu đồ phân bố y hệt Figure 4 trong bài báo"""
    plt.figure(figsize=(7, 5))
    
    # Tính toán tọa độ trục X để dàn ngang các điểm ra
    len_clean = len(data['clean'])
    len_gtsrb = len(data['gtsrb'])
    len_stl10 = len(data['stl10'])
    
    x_clean = range(0, len_clean)
    x_gtsrb = range(len_clean, len_clean + len_gtsrb)
    x_stl10 = range(len_clean + len_gtsrb, len_clean + len_gtsrb + len_stl10)
    
    # Vẽ các cụm điểm theo đúng màu/hình khối của bài báo
    plt.scatter(x_clean, data['clean'], color='seagreen', label='clean', marker='o', s=60, alpha=0.9)
    plt.scatter(x_gtsrb, data['gtsrb'], color='olive', label='gtsrb', marker='v', s=60, alpha=0.9)
    plt.scatter(x_stl10, data['stl10'], color='darkorange', label='stl10', marker='s', s=50, alpha=0.9)
    
    # Vẽ đường Threshold = 0.1
    plt.axhline(y=THRESHOLD, color='gray', linestyle='--', linewidth=1.5, label=f'Threshold = {THRESHOLD}')
    
    # Căn chỉnh giao diện
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('$\mathcal{PL}^1$-Norm', fontsize=12)
    plt.title('Distribution of Inverted Triggers (CLIP-Text-ResNet50)', fontsize=13, fontweight='bold')
    plt.legend(loc='upper right')
    
    # Tắt số trên trục X (vì nó chỉ là index mô hình, giống bài báo)
    plt.xticks([]) 
    
    plt.tight_layout()
    plt.savefig('Figure_4_Distribution.jpg', dpi=300)
    print("-> Đã xuất: Figure_4_Distribution.jpg")
    plt.close()

def plot_roc_curve(data):
    """Vẽ đường cong ROC gộp chung tất cả mô hình bệnh (Giống Figure 8)"""
    clean_pl1 = data['clean']
    # Gộp chung GTSRB và STL-10 làm nhóm "Bệnh" (Positive class)
    bad_pl1 = data['gtsrb'] + data['stl10'] 
    
    y_true = [0] * len(clean_pl1) + [1] * len(bad_pl1)
    # Nhân với -1 vì PL1-norm càng NHỎ thì khả năng là mô hình bệnh càng CAO
    y_scores = [-x for x in clean_pl1] + [-x for x in bad_pl1]
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='crimson', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC Curve of DECREE Detection', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('Figure_8_ROC.jpg', dpi=300)
    print("-> Đã xuất: Figure_8_ROC.jpg")
    plt.close()

if __name__ == '__main__':
    data = extract_data()
    
    print(f"\n=========================================")
    print(f" TỔNG KẾT DỮ LIỆU ĐÃ NẠP")
    print(f"=========================================")
    print(f"- Mô hình Sạch (Clean) : {len(data['clean'])}")
    print(f"- Mô hình Bệnh GTSRB   : {len(data['gtsrb'])}")
    print(f"- Mô hình Bệnh STL-10  : {len(data['stl10'])}")
    print(f"=========================================\n")
    
    if len(data['clean']) > 0 and (len(data['gtsrb']) > 0 or len(data['stl10']) > 0):
        plot_distribution(data)
        plot_roc_curve(data)
        print("\nHOÀN TẤT! Hãy mở 2 file ảnh .jpg lên để chiêm ngưỡng kết quả thực nghiệm!")
    else:
        print("LỖI: Chưa đủ dữ liệu (Cần ít nhất 1 Clean và 1 Backdoor) để vẽ đồ thị.")