import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import argparse

def preprocess_data(raw_data_path, output_path):
    print(f"Memulai preprocessing...")

    # Load data
    try:
        df_raw = pd.read_csv(raw_data_path)
        print("Data mentah berhasil dimuat.")
    except FileNotFoundError:
        print(f"Error: File data mentah tidak ditemukan di {raw_data_path}")
        return

    features = ['income', 'credit_score', 'loan_amount', 'years_employed', 'points']
    target = 'loan_approved'
    
    # Pastikan semua kolom ada
    for col in features + [target]:
        if col not in df_raw.columns:
            print(f"Error: Kolom '{col}' tidak ditemukan di data mentah.")
            return
            
    X = df_raw[features]
    y = df_raw[target]
    print("Fitur (X) dan target (y) telah dipisahkan.")

    scaler = MinMaxScaler()
    X_scaled_array = scaler.fit_transform(X)
    
    # Konversi kembali ke DataFrame dengan nama kolom dan index yang sama
    X_scaled = pd.DataFrame(X_scaled_array, columns=features, index=X.index)
    print("MinMaxScaler telah diterapkan.")

    # Gabungkan X_scaled dan y
    df_final = pd.concat([X_scaled, y], axis=1)
    print("Fitur (X_scaled) dan target (y) telah digabungkan.")

    # Save hasil preprocessing
    # Memastikan direktori output ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"Data preprocessing selesai dan disimpan di: {output_path}")


if __name__ == "__main__":
    # Blok ini akan dieksekusi ketika file dijalankan sebagai skrip
    # python preprocessing/automate_Muhammad-Hafidz-Rizki.py
    
    # Sesuai struktur folder, data mentah ada di root ('../')
    # dan data olahan disimpan di dalam folder 'preprocessing/'
    
    # Setup parser untuk argumen (opsional tapi praktik yang baik)
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline.")
    parser.add_argument('--input', default='./loan_approval_raw.csv', help='Path ke file data mentah (raw).')
    parser.add_argument('--output', default='preprocessing/loan_approval_preprocessing.csv', help='Path untuk menyimpan file hasil preprocessing.')
    
    args = parser.parse_args()
    
    # Panggil fungsi utama
    preprocess_data(args.input, args.output)