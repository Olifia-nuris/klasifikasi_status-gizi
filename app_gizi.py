import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
import math
import os
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

dataset='https://drive.google.com/uc?id=138anPqj-DF8Af_1HS6FRwkubr34SsK6n'
model= r"model_gizi.sav"
label_map = {
        0: "Gizi Baik",
        1: "Gizi Kurang",
        2: "Gizi Lebih"
    }
# membaca model 
try:
    model_gizi = pickle.load(open(model, 'rb'))
except FileNotFoundError:
    st.error("File model tidak ditemukan. Pastikan path file benar.")
# Ambil semua dokumentasi preprocessing dari pickle
df_imputasi = model_gizi.get('dok_misval')
df_encode = model_gizi.get('dok_encoded')
df_scaling = model_gizi.get('dok_scaled')
df_conv = model_gizi.get('dok_usia')

# Ambil data split
split_data = model_gizi.get('split')
if split_data:
    X_train = split_data['X_train']
    X_test = split_data['X_test']
    y_train = split_data['y_train']
    y_test = split_data['y_test']

# Ambil hasil SMOTE
smote_data = model_gizi.get('smote')
if smote_data:
    X_res = smote_data['X_res']
    y_res = smote_data['y_res']
# load model preprocesing
encoders = model_gizi.get('encoder')      # label encoder
scaler = model_gizi.get('scaler')         # scaler MinMax/Standard
model_adb = model_gizi['adaboost']['models']
alphas_adb = model_gizi['adaboost']['alphas']
clas_adb = np.array(model_gizi['adaboost']['classes'])
fitur_asli = model_gizi['input_features']  



# Sidebar menu
menu = st.sidebar.selectbox(
    "Main Menu",
    options=["Home", "Model Klasifikasi", "prediksi"],
    index=0
)

# Membaca dataset jika tersedia
gizi_df = None

try:
    gizi_df = pd.read_csv(dataset)
except Exception as e:
    st.error(f"❌ Gagal memuat dataset: {e}")

# Tampilan Menu "Home" (Dokumentasi)
if menu == "Home":
    st.title("IMPLEMENTASI ENSEMBLE ADABOOST PADA ALGORITMA C5.0 UNTUK OPTIMALISASI KLASIFIKASI STATUS GIZI BALITA")
    st.markdown(
        """
        Total data yang digunakan sebesar 2.759 data status gizi pada balita yang berasal dari dua sumber yaitu :
        - UPT Puskesmas Sembayat berjumlah 1.477 balita
        - Kaggle berjumlah 1.289 balita 
        
        Data yang digunakan terdapat 6 atribut diantaranya jenis kelamin, usia saat ini, Berat, Tinggi dan LILA (Lingkar Lengan Atas) dan status gizi
        
        Kelas status gizi yang akan diklasifikasikan terdiri dari 3 kelas diantaranya gizi baik, Gizi Rendah dan gizi berlebih
        
        """
    )
    if gizi_df is not None:
        st.success("Dataset berhasil dimuat!")
        st.markdown("## sampel Dataset asli ")
        st.dataframe(gizi_df.head())
        st.markdown("## sampel Dataset yang digunakan ")
        gizi_df = gizi_df.drop(columns=["Nama", "Tgl Lahir", "Nama Ortu" , "Desa/Kel", "RT", "RW","ZS TB/U","ZS BB/U","ZS BB/TB"])
        st.dataframe(gizi_df.head())
        st.markdown("## Penyebaran data berdasarkan kelas")
        class_counts = gizi_df['STATUS GIZI'].value_counts()
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(class_counts.index, class_counts.values)
            # Tambahkan nilai di atas batang
            for bar in bars:
                yval = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    yval + 8,        # jarak teks dari batang
                    str(yval),
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    fontweight='bold'
                )

            ax.set_title("Distribusi Status Gizi")
            ax.set_xlabel("Kategori Status Gizi")
            ax.set_ylabel("Jumlah Data")

            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Gagal membuat grafik: {e}")

        st.markdown("## Preprocesing")
        st.markdown("### 1. Imputasi Missing Value dengan metode mean")
        st.markdown(""" Cek missing value tiap kolom """)
        st.dataframe(gizi_df.isna().sum())
        st.markdown(""" setelah imputasi missing value """)
        st.write(df_imputasi.isna().sum())
        st.dataframe(df_imputasi.head())


        st.markdown("### 2. Transformasi Data")
        st.markdown(""" pada proses ini kolom bulan akan ditransformasi guna seragam dan mudah diproses oleh program. 
                    jadi dari format Hari/Bulan/Tahun menjadi Bulan """)
        st.dataframe(df_conv.head())


        st.markdown("### 3. Label Encoding")
        st.markdown(""" pada proses ini kolom bulan akan ditransformasi guna seragam dan mudah diproses oleh program. 
                    jadi dari format Hari/Bulan/Tahun menjadi Bulan """)
        st.dataframe(df_encode.head())
        
        
        st.markdown("### 4. Normalisasi")
        st.markdown(""" data dinormalisasi agar nilainya dalam rentang 0-1 menggunakan metode
                    normalisasi min max. tujuan proses ini agar jarak yang dihasilkan seragam. 
                    diterapkan pada fitur numerik""")
        st.dataframe(df_scaling.head())


        st.markdown("### 5. Pembagian Data")
        st.markdown(""" Data kemudian dibagi menjadi dua bagian dengan rasio 80:20, 
                    yaitu 80% untuk data latih dan 20% untuk data uji.""")
        split_data = model_gizi['split']
        st.write("Jumlah data total :", len(gizi_df))
        st.write("Jumlah data training :", len(X_train))
        st.write("Jumlah data testing :", len(X_test))
        

        st.markdown("## SMOTE")
        st.markdown(""" Dikarenakan jumlah tiap kelas tidak seimbang. maka, disini 
                    SMOTE digunakan untuk mesintesis data pada kelas minoritas agar jumlahnya seimbang
                    dengan data mayoritas. sehingga menghasilkan model yang tidak bias""")
        st.markdown("### Jumlah tiap kelas sebelum SMOTE")
        before_counts = y_train.value_counts()
        fig_before, ax_before = plt.subplots(figsize=(6,4))
        bars = ax_before.bar(before_counts.index, before_counts.values)

        for bar in bars:
            yval = bar.get_height()
            ax_before.text(bar.get_x() + bar.get_width()/2, yval + 5, str(yval),
                        ha='center', fontsize=10, fontweight="bold")

        ax_before.set_title("Distribusi Kelas Sebelum SMOTE")
        ax_before.set_xlabel("Kelas")
        ax_before.set_ylabel("Jumlah")
        st.pyplot(fig_before)

        # --- Setelah SMOTE ---
        st.markdown("### Jumlah tiap kelas setelah SMOTE")
        sm = model_gizi['smote']
        after_counts = y_res.value_counts()
        fig_after, ax_after = plt.subplots(figsize=(6,4))
        bars = ax_after.bar(after_counts.index, after_counts.values)

        for bar in bars:
            yval = bar.get_height()
            ax_after.text(bar.get_x() + bar.get_width()/2, yval + 5, str(yval),
                        ha='center', fontsize=10, fontweight="bold")

        ax_after.set_title("Distribusi Kelas Setelah SMOTE")
        ax_after.set_xlabel("Kelas")
        ax_after.set_ylabel("Jumlah")
        st.pyplot(fig_after)

    else:
        st.warning("Dataset belum dapat dimuat.")
elif menu == "Model Klasifikasi":
    st.header("📊 Evaluasi Model C5.0 dan Adaboost dalam klasifikasi status gizi")

    # ============================
    # ======= MODEL C5.0 =========
    # ============================
    st.subheader("🌳 Model Algoritma C5.0")

    tree_c50 = model_gizi['algoritma.C5']['tree']
    depth = model_gizi['algoritma.C5']['depth']

    X_test = model_gizi['X_test']
    y_test = model_gizi['y_test']
    def predict_sample(row, tree):
        for feature in tree.keys():
            value = row[feature]
            branches = tree[feature]

            for branch, subtree in branches.items():

                branch_str = str(branch)

                # numeric threshold "<"
                if branch_str.startswith("<"):
                    thr = float(branch_str.split()[1])
                    if float(value) < thr:
                        return predict_sample(row, subtree) if isinstance(subtree, dict) else subtree

                # numeric threshold "≥"
                elif branch_str.startswith("≥"):
                    thr = float(branch_str.split()[1])
                    if float(value) >= thr:
                        return predict_sample(row, subtree) if isinstance(subtree, dict) else subtree

                # categorical (string 0/1)
                else:
                    if str(value) == branch_str:
                        return predict_sample(row, subtree) if isinstance(subtree, dict) else subtree

        return None

    # Prediksi semua data test C5.0
    y_pred = [predict_sample(X_test.iloc[i], tree_c50) for i in range(len(X_test))]

    # =======================
    #     EVALUASI C5.0
    # =======================
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    sensitivity = recall
    gmean = math.sqrt(precision * recall)

    metrik_df = pd.DataFrame({
        "Metrik": ["Accuracy", "Precision", "Recall (Sensitivity)", "F1-Score", "G-Mean"],
        "Nilai": [acc, precision, recall, f1, gmean]
    })

    st.subheader("📌 Tabel Evaluasi Model C5.0")
    st.dataframe(metrik_df)

    st.subheader("📌 Confusion Matrix")
    st.dataframe(pd.DataFrame(cm))

    # =======================
    # Sample Prediksi C5.0
    # =======================
    st.subheader("📄 Perbandingan Prediksi vs Aktual")

    sample_size = 10
    sample_df = X_test.head(sample_size).copy()
    sample_df["Actual"] = y_test[:sample_size].values
    sample_df["Predict"] = y_pred[:sample_size]
    st.dataframe(sample_df)


    # ================================================================
    # ====================    ADABOOST + C5.0    ======================
    # ================================================================
    st.subheader("🌳 Model Algoritma C5.0 + ADABOOST")
    st.markdown(""" Algoritma C5.0 digunakan sebagai base learner dari metode ensemble
                 Adaboost guna meningkatkan akurasi dibandingkan model tunggal.""")

    model_adb = model_gizi['adaboost']['models']
    alphas_adb = model_gizi['adaboost']['alphas']
    clas_adb = np.array(model_gizi['adaboost']['classes'])   # pastikan array


    # ============================
    # Fungsi prediksi ADABOOST
    # ============================
    def predict_adaboost(models, alphas, classes, X):
        M = len(models)
        n = X.shape[0]

        class_scores = np.zeros((n, len(classes)))

        for m in range(M):
            pred = np.array([predict_sample(X.iloc[i], models[m]) for i in range(n)])
            alpha = alphas[m]

            for i, p in enumerate(pred):
                if p in classes:
                    class_idx = np.where(classes == p)[0][0]
                else:
                    # fallback: jika prediksi None atau tidak termasuk kelas
                    class_idx = np.argmax(np.bincount(classes)) if len(classes) > 0 else 0  
                class_scores[i, class_idx] += alpha


        final_pred = classes[np.argmax(class_scores, axis=1)]
        return final_pred


    # Prediksi Adaboost
    y_pred_adb = predict_adaboost(model_adb, alphas_adb, clas_adb, X_test)

    # ============================
    #      EVALUASI ADABOOST
    # ============================
    acc_adb = accuracy_score(y_test, y_pred_adb)
    cm_adb = confusion_matrix(y_test, y_pred_adb)

    precision_adb, recall_adb, f1_adb, _ = precision_recall_fscore_support(
        y_test, y_pred_adb, average="macro", zero_division=0
    )

    gmean_adb = math.sqrt(precision_adb * recall_adb)

    metrik_adb_df = pd.DataFrame({
        "Metrik": ["Accuracy", "Precision", "Recall (Sensitivity)", "F1-Score", "G-Mean"],
        "Nilai": [acc_adb, precision_adb, recall_adb, f1_adb, gmean_adb]
    })

    st.subheader("📌 Tabel Evaluasi Adaboost (C5.0 Base Learner)")
    st.dataframe(metrik_adb_df)

    st.subheader("📌 Confusion Matrix Adaboost")
    st.dataframe(pd.DataFrame(cm_adb))

    # ============================
    # Sample Prediksi Adaboost
    # ============================
    st.subheader("📄 Contoh Perbandingan Prediksi vs Aktual (Adaboost)")

    sample_df_adb = X_test.head(sample_size).copy()
    sample_df_adb["Actual"] = y_test[:sample_size].values
    sample_df_adb["Predict"] = y_pred_adb[:sample_size]

    st.dataframe(sample_df_adb)



elif menu == "prediksi":
    st.header("📊 Prediksi Status Gizi Balita")
    fitur_asli = model_gizi['split']['X_train'].columns.tolist()

    # === Input Form ===
    jk = st.selectbox("Jenis Kelamin", encoders['JK'].classes_)
    usia = st.number_input("Usia Saat Ukur (bulan)", 0.0, 60.0, 12.0)
    bb = st.number_input("Berat Badan (kg)", 0.0, 50.0, 10.0)
    tb = st.number_input("Tinggi Badan (cm)", 0.0, 200.0, 50.0)
    lila = st.number_input("LiLA (cm)", 0.0, 40.0, 10.0)

    if st.button("Prediksi Status Gizi"):

        # === 1. Encoding JK ===
        jk_encoded = encoders['JK'].transform([jk])[0]

        # === 2. Buat dataframe input ===
        df_input = pd.DataFrame([{
            "JK": jk_encoded,
            "Usia Saat Ukur": usia,
            "Berat": bb,
            "Tinggi": tb,
            "LiLA": lila
        }])
        df_input = df_input.reindex(columns=fitur_asli) 

        # === 3. Scaling sesuai model ===
        kolom_scaling = scaler.feature_names_in_
        df_input_scaled = df_input[kolom_scaling]
        scaled_values = scaler.transform(df_input_scaled)
        df_scaled = pd.DataFrame(scaled_values, columns=kolom_scaling)

        def predict_sample(row, tree):
            for feature in tree.keys():
                value = row[feature]
                branches = tree[feature]

                for branch, subtree in branches.items():

                    branch_str = str(branch)

                    # numeric threshold "<"
                    if branch_str.startswith("<"):
                        thr = float(branch_str.split()[1])
                        if float(value) < thr:
                            return predict_sample(row, subtree) if isinstance(subtree, dict) else subtree

                    # numeric threshold "≥"
                    elif branch_str.startswith("≥"):
                        thr = float(branch_str.split()[1])
                        if float(value) >= thr:
                            return predict_sample(row, subtree) if isinstance(subtree, dict) else subtree

                    # categorical (string 0/1)
                    else:
                        if str(value) == branch_str:
                            return predict_sample(row, subtree) if isinstance(subtree, dict) else subtree

            return None


        def predict_adaboost(models, alphas, classes, X):
            M = len(models)
            n = X.shape[0]

            class_scores = np.zeros((n, len(classes)))

            for m in range(M):
                pred = np.array([predict_sample(X.iloc[i], models[m]) for i in range(n)])
                alpha = alphas[m]

                for i, p in enumerate(pred):
                    if p in classes:
                        class_idx = np.where(classes == p)[0][0]
                    else:
                        class_idx = 0  # fallback
                    class_scores[i, class_idx] += alpha

            final_pred = classes[np.argmax(class_scores, axis=1)]
            return final_pred

        # === 4. Prediksi dengan ADABOOST ===
        df_final = pd.DataFrame(columns=fitur_asli)
        df_final.loc[0] = 0  # default isi angka 0

        # isi kolom scaling
        for col in kolom_scaling:
            df_final.at[0, col] = df_scaled.at[0, col]

        # isi JK (yang tidak di-scaling)
        if 'JK' in df_final.columns:
            df_final.at[0, 'JK'] = jk_encoded

        # === 4. Prediksi dengan ADABOOST ===
        y_pred = predict_adaboost(model_adb, alphas_adb, clas_adb, df_final)


        # === 5. Mapping hasil ke label asli ===
        pred_label = label_map[int(y_pred[0])]

        # === 6. Tampilkan hasil ===
        st.success(f"### 🔍 Hasil Prediksi: **{pred_label}**")

