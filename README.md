# Klasifikasi Penyakit Daun Tomat

Proyek ini merupakan implementasi model deep learning untuk **mendeteksi dan mengklasifikasi penyakit daun tomat** menggunakan **Convolutional Neural Network (CNN)**. Dataset yang digunakan berasal dari Kaggle dan telah dimodifikasi menjadi tiga subset: `train`, `validation`, dan `test`.

## Struktur Dataset

# Sumber Dataset

Dataset diunduh dari: 
`https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf`

Dataset asli dari Kaggle hanya terdiri dari dua folder: `train` dan `val`. Namun, dalam proyek ini dataset dibagi ulang menjadi tiga bagian: `train, validation, dan test`

- `train/` : 8000 gambar
- `val/` : 1000 gambar
- `test/` : 1000 gambar

Setiap folder berisi 10 kelas penyakit daun tomat, yaitu:

- Tomato___Bacterial_spot  
- Tomato___Early_blight  
- Tomato___Late_blight  
- Tomato___Leaf_Mold  
- Tomato___Septoria_leaf_spot  
- Tomato___Spider_mites Two-spotted_spider_mite  
- Tomato___Target_Spot  
- Tomato___Tomato_Yellow_Leaf_Curl_Virus  
- Tomato___Tomato_mosaic_virus  
- Tomato___healthy  

Jumlah gambar:
- Train: **8000**
- Validation: **1000**
- Test: **1000**

## Arsitektur Model

Model CNN yang digunakan terdiri dari:

- 3 lapisan Conv2D dengan MaxPooling
- 1 lapisan Dense dengan 128 neuron
- Dropout (0.5) untuk mencegah overfitting
- Output layer dengan aktivasi softmax sebanyak 10 kelas

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Pengaturan Training
Optimizer: Adam

Loss Function: Categorical Crossentropy

Batch Size: 16

Epochs: 100 (dengan EarlyStopping)

Callbacks:

EarlyStopping: untuk menghentikan pelatihan jika val_loss tidak membaik

ModelCheckpoint: menyimpan model terbaik (val_loss terendah)

ReduceLROnPlateau: mengurangi learning rate saat val_loss stagnan

# Augmentasi Data
Augmentasi dilakukan pada data training untuk meningkatkan generalisasi model:
ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# Evaluasi Model

Akurasi	91.20%
Precision Macro	0.91
Recall Macro	0.91
F1-score Macro	0.91

# Classification Report

                                               precision    recall  f1-score   support

       Tomato___Bacterial_spot                   0.92      0.90      0.91       100
       Tomato___Early_blight                     0.92      0.85      0.89       100
       Tomato___Late_blight                      0.98      0.90      0.94       100
       Tomato___Leaf_Mold                        0.94      0.91      0.92       100
       Tomato___Septoria_leaf_spot               0.84      0.92      0.88       100
       Tomato___Spider_mites                     0.86      0.90      0.88       100
       Tomato___Target_Spot                      0.83      0.80      0.81       103
       Tomato___Tomato_Yellow_Leaf_Curl_Virus    0.99      0.95      0.97        97
       Tomato___Tomato_mosaic_virus              0.88      1.00      0.94       100
       Tomato___healthy                          0.98      1.00      0.99       100

    Accuracy:                                   91.20%
    Macro avg:                                  0.91      0.91      0.91      1000
    Weighted avg:                               0.91      0.91      0.91      1000

