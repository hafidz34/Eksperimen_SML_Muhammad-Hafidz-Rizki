import pandas as pd
import os
import argparse

def preprocess_data(raw_data_path, output_path):
    print(f"Memulai preprocessing...")

    try:
        df_raw = pd.read_csv(raw_data_path)
        print("Data mentah berhasil dimuat.")
    except FileNotFoundError:
        print(f"Error: File data mentah tidak ditemukan di {raw_data_path}")
        return

    # Kolom yang akan disimpan (fitur + target)
    columns_to_keep = [
        'income', 'credit_score', 'loan_amount', 'years_employed', 'points', # Fitur
        'loan_approved' # Target
    ]
    
    # Pastikan semua kolom ada
    for col in columns_to_keep:
        if col not in df_raw.columns:
            print(f"Error: Kolom '{col}' tidak ditemukan di data mentah.")
            return
            
    # Buat DataFrame final hanya dengan kolom yang diinginkan tanpa scaling
    df_final = df_raw[columns_to_keep]
    print("Kolom yang tidak relevan telah dibuang.")

    # Save hasil preprocessing
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"Data preprocessing (bersih, unscaled) selesai dan disimpan di: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline (column selection only).")
    parser.add_argument('--input', default='./loan_approval_raw.csv', help='Path ke file data mentah (raw).')
    parser.add_argument('--output', default='preprocessing/loan_approval_preprocessing.csv', help='Path untuk menyimpan file hasil preprocessing.')
    
    args = parser.parse_args()
    
    preprocess_data(args.input, args.output)