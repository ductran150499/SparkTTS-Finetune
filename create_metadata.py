import os
import csv

# Đường dẫn tới folder chứa cả .wav và .txt
data_folder = "/Users/tranminhduc/Downloads/630fba81"

# File output CSV
metadata_path = os.path.join(data_folder, "metadata.csv")

# Ghi metadata.csv
with open(metadata_path, mode='w', encoding='utf-8', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["audio_path", "text"])  # Header

    for file in sorted(os.listdir(data_folder)):
        if file.endswith(".wav"):
            base_name = os.path.splitext(file)[0]
            txt_path = os.path.join(data_folder, base_name + ".txt")

            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                writer.writerow([file, text])
            else:
                print(f"⚠️ Không tìm thấy file .txt cho {file}")

print(f"✅ Đã tạo xong metadata.csv tại: {metadata_path}")
