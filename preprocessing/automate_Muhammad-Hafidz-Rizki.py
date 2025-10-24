import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import argparse

def preprocess_and_split(raw_data_path, output_folder):
    """
    Memuat data mentah, membersihkan, melakukan train-test split,
    normalisasi (MinMaxScaler), dan menyimpan train/test set terpisah.
    """
    print(f"Memulai preprocessing dan split...")

    try:
        # Menggunakan path absolut dari argumen
        df_raw = pd.read_csv(raw_data_path)
        print("Data mentah berhasil dimuat.")
    except FileNotFoundError:
        print(f"Error: File data mentah tidak ditemukan di {raw_data_path}")
        return
    except Exception as e:
        print(f"Error saat memuat data: {e}")
        return

    df_processed = df_raw.copy()

    # Menghapus kolom 'name' dan 'city'
    columns_to_drop = ['name', 'city']
    # Cek apakah kolom ada sebelum drop
    columns_exist = [col for col in columns_to_drop if col in df_processed.columns]
    if columns_exist:
        df_processed = df_processed.drop(columns_exist, axis=1)
        print(f"Kolom {columns_exist} telah dibuang.")
    else:
        print("Kolom 'name' dan 'city' tidak ditemukan, melanjutkan.")

    # Mengubah 'loan_approved' (boolean) menjadi integer (0 atau 1)
    target_col = 'loan_approved'
    if target_col in df_processed.columns:
        if df_processed[target_col].dtype == 'bool':
            df_processed[target_col] = df_processed[target_col].astype(int)
            print(f"Kolom '{target_col}' diubah menjadi tipe integer.")
        elif pd.api.types.is_numeric_dtype(df_processed[target_col]):
             print(f"Kolom '{target_col}' sudah numerik.")
        else:
            print(f"Warning: Tipe data kolom '{target_col}' tidak dikenali sebagai boolean atau numerik.")

    else:
        print(f"Error: Kolom target '{target_col}' tidak ditemukan.")
        return

    # Pemisahan fitur dan target
    features = ['income', 'credit_score', 'loan_amount', 'years_employed', 'points']
    # Pastikan semua kolom fitur ada
    missing_features = [f for f in features if f not in df_processed.columns]
    if missing_features:
        print(f"Error: Kolom fitur berikut tidak ditemukan: {missing_features}")
        return

    X = df_processed[features]
    y = df_processed[target_col]

    # Train-test split (dilakukan sebelum normalisasi)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("Data berhasil dibagi menjadi train (80%) dan test (20%) set.")
        print(f"  Ukuran Train Set: {X_train.shape[0]} baris")
        print(f"  Ukuran Test Set: {X_test.shape[0]} baris")
    except Exception as e:
        print(f"Error saat melakukan train-test split: {e}")
        return

    # Normalisasi fitur menggunakan MinMaxScaler
    scaler = MinMaxScaler()

    # Fit scaler hanya pada data training
    scaler.fit(X_train)
    print("MinMaxScaler di-fit pada data training.")

    # Transform data training dan testing
    X_train_scaled_array = scaler.transform(X_train)
    X_test_scaled_array = scaler.transform(X_test)

    # Konversi kembali ke DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=features, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=features, index=X_test.index)
    print("Data training dan testing telah di-transform (dinormalisasi).")

    # Menggabungkan kembali fitur dan target
    df_train_final = pd.concat([X_train_scaled, y_train], axis=1)
    df_test_final = pd.concat([X_test_scaled, y_test], axis=1)

    # Menyimpan hasil ke file CSV
    try:
        # Membuat folder output jika belum ada
        os.makedirs(output_folder, exist_ok=True)

        # Membuat path lengkap untuk file output
        train_output_path = os.path.join(output_folder, 'loan_approval_train_preprocessing.csv')
        test_output_path = os.path.join(output_folder, 'loan_approval_test_preprocessing.csv')

        # Menyimpan DataFrame ke file CSV
        df_train_final.to_csv(train_output_path, index=False)
        df_test_final.to_csv(test_output_path, index=False)

        print("-" * 30)
        print(f"File preprocessing berhasil disimpan di folder: '{output_folder}'")
        print(f"  - {os.path.basename(train_output_path)}")
        print(f"  - {os.path.basename(test_output_path)}")
        print("-" * 30)

    except Exception as e:
        print(f"Error saat menyimpan file hasil: {e}")

if __name__ == "__main__":
    # Setup argumen command line
    parser = argparse.ArgumentParser(description="Preprocess loan approval data: clean, split, scale, and save.")

    # Argumen untuk path input data mentah
    # Defaultnya disesuaikan dengan path absolut yang kamu berikan
    parser.add_argument(
        '--input',
        default=r'C:\Users\User\Downloads\SMSML_Muhammad-Hafidz-Rizki\Eksperimen_SML_Muhammad-Hafidz-Rizki\loan_approval_raw.csv',
        help='Path lengkap ke file data mentah (raw CSV).'
    )

    # Argumen untuk folder output
    parser.add_argument(
        '--output_folder',
        default='loan_approval_preprocessing', # Nama folder output relatif terhadap lokasi skrip
        help='Nama folder untuk menyimpan file train dan test hasil preprocessing.'
    )

    # Parsing argumen yang diberikan saat menjalankan skrip
    args = parser.parse_args()

    # Memanggil fungsi utama dengan argumen yang sudah diparsing
    preprocess_and_split(args.input, args.output_folder)