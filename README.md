## Why need Churn Prediction ?

Karena saat ini,kemungkinan pelanggan untuk berganti service provider sangat tinggi dengan segala kemudahan yg ada dan ini menjadi pekerjaan rumah yg cukup sulit bagi  sevice provider untuk mempertahan kan pelanggan nya.
Hal ini tentunya sangat berdampak besar bagi perusahaan service provider. Karena budgeting untuk pelanggan baru pastinya lebih besar dibanding kan dengan budgeting untuk memaintain pelanggan yg sudah ada.
oleh karena itu perusahaan service provider memerlukan tools yang mampu memprediksi kemungkinan pelanggan yg akan churn atau berganti ke srvice provider lain

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pickle
from pathlib import Path


### Data Understanding

Terdapat 11 Kolom & 4930 barisa data

Features
-	Dependents: Apakah custome memiliki tanggungan atau tidak
-	Tenure: Durasi pelanggan menjadi customer
-	OnlineSecurity: Apakah pelanggan menggunakan layanan pengamanan daring (online security) atau tidak
-	OnlineBackup: Apakah pelanggan menggunakan layanan online backup atau tidak
-	InternetService: Jenis layanan internet yang digunakan oleh pelanggan
-	DeviceProtection: Apakah pelanggan menggunakan layanan device protection atau tidak
-	TechSupport: Apakah pelanggan menggunakan layanan tech support atau tidak
-	Contract: Tipe kontrak berdasarkan durasi penggunaan layanan
-	PaperlessBilling: Apakah tagihan menggunakan kertas atau tidak
-	MonthlyCharges: Nominal tagihan bulanan pelanggan
-	Churn: Apakah pelanggan churn atau tidak

### Data PreProcessing
# Cek duplicate data --> 77 duplicate
# Cek Missing value --> No Missing value
# Cek anomaly data
# Cek OUtlier --> No Outlier di dataset ini


### Exploratory Data Analysis - EDA
### Univariate Analysis
Merupakan metode analisa untuk memeriksa distribusi, statistik deskriptif, dan karakteristik lain dari satu variabel tunggal. ada 3 variabel yg kita cek yaitu: Count of Churn, Count OF Interner Sevices & Distribusi Tenur

### Multivariate Analysis
Merupakan metode analisa untuk memeriksa hubungan antara dua atau lebih variabel
1. Korelasi antara Churn Rate dan Tenure
2. Korelasi antara Churn Rate dan MonthlyCharges
3. Korelasi antara Churn Rate dan internet_service
4. Korelasi antara Churn Rate dan PaperlessBilling

Feature Engineering
Menambahkan 1 kolom TotalCharges = tenure x MonthlyCharges

Feature Encoding --> LabelEncoder

Splitting Dataset
Dataset di bagi menjadi 2 bagian (70% training & 30% testing) berdasarkan variable predictor (X) dan targetnya (Y). menggunakan train_test_split() untuk membagi data tersebut. 
menggunakan value_counts untuk mengecek apakah pembagian sudah sama proporsinya dan hasil spliting data menjadi x_train, y_train, x_test & y_test
Jumlah baris dan kolom dari x_train adalah: (3397, 11) , sedangkan Jumlah baris dan kolom dari y_train adalah: (3397,)
Prosentase Churn di data Training adalah:
Churn
0    0.739476
1    0.260524
Name: proportion, dtype: float64
Jumlah baris dan kolom dari x_test adalah: (1456, 11) , sedangkan Jumlah baris dan kolom dari y_test adalah: (1456,)
Prosentase Churn di data Testing adalah:
Churn
0    0.723214
1    0.276786
Name: proportion, dtype: float64

Note : Proses encoding ini mengubah value dari data yang masih berbentuk string menjadi numeric, setelah dilakukan terlihat di persebaran datanya khususnya kolom min dan max dari masing masing variable sudah berubah menjadi 0 & 1. 
Proses selanjutnya di data splitting, dimana data di bagi menjadi 2 bagian untuk keperluan modelling, setelah dilakukan terlihat dari jumlah baris dan kolom masing-masing data sudah sesuai & prosentase kolom churn juga sama dengan data di awal, hal ini mengindikasikan bahwasannya data terpisah dengan baik dan benar.

### Modeling

Modelling : Logistic Regression
Note - Logistic Regression :
1. Dari data training terlihat bahwasannya model mampu memprediksi data dengan menghasilkan akurasi sebesar 81%, 
   dengan detil tebakan churn yang sebenernya benar churn adalah 501, tebakan tidak churn yang sebenernya tidak churn adalah 2235, 
   tebakan tidak churn yang sebenernya benar churn adalah 384 dan tebakan churn yang sebenernya tidak churn adalah 277
2. Dari data testing terlihat bahwasannya model mampu memprediksi data dengan menghasilkan akurasi sebesar 78%, 
   dengan detil tebakan churn yang sebenernya benar churn adalah 214, tebakan tidak churn yang sebenernya tidak churn adalah 928, 
   tebakan tidak churn yang sebenernya benar churn adalah 189 dan tebakan churn yang sebenernya tidak churn adalah 125
   
   
Modelling : Random Forest
Note - Random Forest :

1. Dari data training terlihat bahwasannya model mampu memprediksi data dengan menghasilkan akurasi sebesar 99%, 
dengan detil tebakan churn yang sebenernya benar churn adalah 872, tebakan tidak churn yang sebenernya tidak churn adalah 2507, 
tebakan tidak churn yang sebenernya benar churn adalah 13 dan tebakan churn yang sebenernya tidak churn adalah 5

2. Dari data testing terlihat bahwasannya model mampu memprediksi data dengan menghasilkan akurasi sebesar 78%, 
dengan detil tebakan churn yang sebenernya benar churn adalah 214, tebakan tidak churn yang sebenernya tidak churn adalah 928, 
tebakan tidak churn yang sebenernya benar churn adalah 189 dan tebakan churn yang sebenernya tidak churn adalah 214.


Modelling : Gradient Boosting
Note - Gradient Boosting :

1. Dari data training terlihat bahwasannya model mampu memprediksi data dengan menghasilkan akurasi sebesar 84%, 
dengan detil tebakan churn yang sebenernya benar churn adalah 519, tebakan tidak churn yang sebenernya tidak churn adalah 3240, 
tebakan tidak churn yang sebenernya benar churn adalah 366 dan tebakan churn yang sebenernya tidak churn adalah 519

2. Dari data testing terlihat bahwasannya model mampu memprediksi data dengan menghasilkan akurasi sebesar 78%, 
dengan detil tebakan churn yang sebenernya benar churn adalah 214, tebakan tidak churn yang sebenernya tidak churn adalah 928, 
tebakan tidak churn yang sebenernya benar churn adalah 189 dan tebakan churn yang sebenernya tidak churn adalah 125.



### Kesimpulan & Rekomendasi

Kesimpulan

Berdasarkan pemodelan yang telah dilakukan dengan menggunakan Logistic Regression, Random Forest dan Extreme Gradiant Boost, 
maka dapat disimpulkan untuk memprediksi churn dari pelanggan telco dengan menggunakan dataset ini model terbaiknya adalah menggunakan algortima Logistic Regression. 
Hal ini dikarenakan performa dari model Logistic Regression cenderung mampu memprediksi sama baiknya di fase training maupun testing (akurasi training 81%, akurasi testing 78%), 
dilain sisi algoritma lainnya cenderung Over-Fitting performanya

Limitation:
- Baris data di kisaran 5000
- Banyak variable yg di uji tidak lebih dari 10 variable (Numerik / Kategorical)
- Untuk data Churn batasan tenur maksimal di kisaran 72 bulan


Rekomendasi
1. Membuat Customer Loyalty Program yang mendorong pelanggan agar tetap bertahan dan memiliki waktu tenure yang panjang. 
   Bentuk program bisa berupa pemberian reward yang besarannya disesuaikan dengan masa tenure. Semakin panjang tenure, 
   semakin besar reward yang bisa didapat, sehingga mendorong pelanggan untuk memiliki tenure yang lebih panjang.
2. Memberikan diskon/potongan harga MonthlyCharges bagi pegawai yang terindikasi/diprediksi akan churn, 
   khususnya untuk pelanggan yang memiliki MonthlyChargesyang tinggi.
3. Menyediakan layangan InternetService Fiber optic dengan harga yang lebih murah
4. Yang terpenting adalah menjaga kualitas jaringan telekomunikasi, 
   karena jika jaringan nya prima otomatis kepuasan pelanggan akan meningkan dan kecenderungan untuk churn dapat di minimalisir


Final model --> log_model.sav
