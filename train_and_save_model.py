# dataset: https://www.kaggle.com/datasets/datamunge/sign-language-mnist/data
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical

# Veri setlerini yükle
train_df = pd.read_csv('Sign Language MNIST/sign_mnist_train/sign_mnist_train.csv')
test_df = pd.read_csv('Sign Language MNIST/sign_mnist_test/sign_mnist_test.csv')

# Veri setlerini doğru şekilde işleme
def preprocess_data(data):
    x = data.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255
    y = to_categorical(data['label'].values, num_classes=26)
    return x, y

# Eğitim ve test veri setlerini işleme
x_train, y_train = preprocess_data(train_df)
x_test, y_test = preprocess_data(test_df)

# Machine learning modeli oluştur
model = Sequential([
    Input(shape=(28, 28, 1)),  # Giriş katmanı, her biri 28x28 boyutunda ve 1 kanallı (gri tonlamalı) görüntüler
    Conv2D(64, (3, 3), activation='relu'),  # 64 filtre içeren ve 3x3 boyutunda kernel kullanarak evrişim katmanı
    MaxPooling2D(2, 2),  # 2x2 havuzlama işlemi yaparak boyut azaltma
    Conv2D(128, (3, 3), activation='relu'),  # 128 filtre içeren ve 3x3 boyutunda kernel kullanarak bir başka evrişim katmanı
    MaxPooling2D(2, 2),  # Yeniden 2x2 havuzlama işlemi
    Flatten(),  # Çok boyutlu katman çıktılarını tek boyutlu bir vektöre düzleştirme
    Dense(256, activation='relu'),  # 256 nöronlu tam bağlantılı katman
    Dense(26, activation='softmax')  # Çıktı katmanı, her biri 26 işaret dili sınıfına ait olasılıkları hesaplayan 26 nöron
])

# Modelin derlenmesi
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğit
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Modeli diske kaydetme
model.save('sign_language_model.h5')

print("Model training complete and saved.")
