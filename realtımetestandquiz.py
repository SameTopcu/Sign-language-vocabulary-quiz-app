import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from random import choice
import time
import tkinter as tk
from tkinter import messagebox
import subprocess  # Diğer Python dosyasını çalıştırmak için subprocess modülü

# GUI için ana pencere fonksiyonu tanımlanıyor
def main_window():
    root = tk.Tk()  # Ana Tkinter penceresi oluşturulur
    root.title("Sign Language Recognition Game")  # Pencerenin başlığı ayarlanır

    # Oyunu başlatma fonksiyonu
    def start_game():
        root.destroy()  # Ana pencereyi kapatır
        start_sign_language_game()  # İşaret dili tanıma oyununu başlatır

    # Hakkında bilgisi gösteren fonksiyon
    def about():
        messagebox.showinfo("INFO", "Sign Language Recognition Quiz Game Used Tensorflow Keras Machine Learning Libraries.")  # Bilgi mesaj kutusu gösterilir

    # Oyunu sonlandırma fonksiyonu
    def exit_game():
        root.destroy()  # Ana pencereyi kapatır ve programı sonlandırır

    # Real-time test başlatma fonksiyonu
    def start_real_time_test():
        subprocess.run(["python", "real_time_test.py"])  # "real_time_test.py" dosyasını çalıştırır

    # Başlama, bilgi, çıkış ve real-time test butonları eklenir
    tk.Button(root, text="START", command=start_game, height=2, width=15).pack(pady=10)  # 'Başla' butonu
    tk.Button(root, text="INFO", command=about, height=2, width=15).pack(pady=10)  # 'Hakkında' butonu
    tk.Button(root, text="QUIT", command=exit_game, height=2, width=15).pack(pady=10)  # 'Çıkış' butonu
    tk.Button(root, text="Real-time Test", command=start_real_time_test, height=2, width=15).pack(pady=10)  # 'Real-time Test' butonu

    root.mainloop()  # Pencereyi göstermek ve etkileşime açık hale getirmek için ana döngüyü başlatır

# İşaret dili tanıma oyununu başlatan fonksiyon
def start_sign_language_game():
    model = tf.keras.models.load_model('sign_language_model.h5')
    mp_hands = mp.solutions.hands # Görüntüde elin yerini saptamak için MediaPipe kütüphanesi eller modülünü kullanır
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)   # El tanıma için yapılandırma: maksimum bir el ve %70 doğruluk
    mp_drawing = mp.solutions.drawing_utils # Çerçeve çizim yardımcı araçlar
    mp_holistic = mp.solutions.holistic # MediaPipe holistic modülü, vücut, el ve yüz bölümlerinin tanıması için yine

    # Kelime listesi ve seçim
    words = ["APP", "PRO", "QUEEN", "BIRD"]  # Quizde gelecek örnek kelimeler listesi
    target_word = choice(words)  # Rastgele bir kelime seçer
    guessed_letters = ['_' for _ in target_word] # Tahmin edilen harfleri göstermek için placeholder

    current_index = 0 # Tahmin edilmekte olan harfin indeksi
    letter_time_limit = 12  # Her harf için izin baştan denemeden önce verilecek maksimum süre
    letter_start_time = time.time()  # Üstteki süre hesabı için zamanlayıcı başlatılır

    label_map = 'ABCDEFGHIKLMNOPQRSTUVWXY' # Kullanılan harflerin listesi, 'Z' ve 'J' hariç. Bunlar fotoğraf değil video gerektiriyor hareketli gösterimleri olduğundan. 
    label_map = {i: label_map[i] for i in range(len(label_map))}  # Harf-etiket eşlemesi

    cap = cv2.VideoCapture(0) # Webcam'den video yakalamak için OpenCV nesnesi

    #Skor ve time başlatımları
    mp_start_t = time.time()
    score = 0
    start_time = time.time()
    limit = 8
    
    # Holistic modelin yapılandırması
    holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    def mediapipe_predict(hand_image):
        hand_image_rgb = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)    # El resmini RGB'ye çevir
        results = holistic.process(hand_image_rgb)  # Resmi işleyerek el bulunup bulunmadığını kontrol et
        if results.left_hand_landmarks or results.right_hand_landmarks: # Eğer el tespit edilirse ok
            if time.time() - mp_start_t > 12: # Zamanlayıcı sıfırla
                return True
            return False
        return False

    predicted_letter = '?'  # Tahmin edilen harfi ekranda başlangıçta ? olarak gösrer
    probability = 0.0   # Tahmin edilen harfin olasılığını başlangıçta 0 olarak ayarla

    print(f"Let's start with the word {target_word}")   # Oyunun başlangıcında seçilen kelimeyi yazdır

    while True:
        ret, frame = cap.read() # Kameradan sürekli çerçeve okur
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # RGBye çevir.
            results = hands.process(frame) # Eli tespitle
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Geri BGRye çevir.

            hand_detected = False # El bulundu mu

            # Bulunduysa her bir el için döngü
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    h, w, _ = frame.shape  # Çerçevenin yüksekliği, genişliği ve kanallarını al
                    min_x = min_y = max_x = max_y = 0  # El çerçevesi için köşe koordinatları başlat

                    # Her bir el işareti noktası için döngü
                    # Noktanın çerçeve içindeki koordinatları
                    for landmark in hand_landmarks.landmark:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        if x < min_x or min_x == 0:
                            min_x = x
                        if x > max_x:
                            max_x = x
                        if y < min_y or min_y == 0:
                            min_y = y
                        if y > max_y:
                            max_y = y

                    if max_x - min_x > 0 and max_y - min_y > 0: # Eğer el işareti yeterince büyükse
                        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)    # El çerçevesini çiz
                        hand_image = frame[min_y:max_y, min_x:max_x]     # El resmini çıkar
                        if hand_image.size > 0:
                            hand_image = cv2.resize(hand_image, (28, 28), interpolation=cv2.INTER_AREA) # Resmi 28x28 boyutuna getir
                            hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)   # Griye çevir
                            hand_image = hand_image.reshape(1, 28, 28, 1) / 255.0 # Normalize et
                            prediction = model.predict(hand_image)    # Modelimiz ile tahmin yap
                            class_index = np.argmax(prediction) # Modelimizden en yüksek olasılıklı sınıfı al
                            predicted_letter = label_map.get(class_index, '?')  # Sınıfı harfe çevir
                            probability = np.max(prediction)    # Tahmin olasılığını al
                            hand_detected = True    

                            # Eğer tahmin edilen harf doğruysa ve yeterince güvenilirse
                            if current_index < len(target_word) and predicted_letter == target_word[current_index] and probability > 0.7:
                                guessed_letters[current_index] = predicted_letter   # Tahmin edilen harfi güncelle,Sonraki harfe geç,  Zamanlayıcıları sıfırla
                                current_index += 1  
                                letter_start_time = time.time() 
                                mp_start_t = time.time()  
                                
                            # Eğer  uzun bir süre içinde harf tahmin edilemezse sıradakiyle denemeye devam et
                            if current_index < len(target_word) and time.time() - letter_start_time > letter_time_limit:
                                guessed_letters[current_index] = target_word[current_index]
                                current_index += 1 
                                letter_start_time = time.time()  

            # Süre sınırını kontrol et
            elapsed_time = time.time() - start_time
            if elapsed_time > limit:
                hand_image_rgb = frame[min_y:max_y, min_x:max_x] # El resmini al
                if hand_image_rgb.size > 0:
                    if mediapipe_predict(hand_image_rgb): # MediaPipe tahmini eşleştiyse skor arttır sonraki kelimeye geç
                        score += 1
                        target_word = choice(words)
                        guessed_letters = ['_' for _ in target_word]
                        current_index = 0 # Kelimeyi ilk harften başla
                        print(f"New word: {target_word}")   # Yeni kelimeyi yazdır
                start_time = time.time()     # Oyun zamanlayıcısını sıfırla

            # Kelimenin tamamlandığını kontrol et ve skoru arttırarak yeni kelimeye geç
            if current_index == len(target_word):
                print(target_word)  # Tamamlanan kelimeyi yazdır
                score += 1
                target_word = choice(words)
                guessed_letters = ['_' for _ in target_word] # Yeni kelime için placeholderları sıfırla
                current_index = 0
                letter_start_time = time.time()
                print(f"Next word: {target_word}")  # Yeni kelimeyi yazdır

            # Ekran üzerine tahmin edilen harf, skor ve mevcut kelimeyi yazdır
            cv2.putText(frame, f'Predicted: {predicted_letter} Prob: {probability:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f'Score: {score}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f'Current Word: {" ".join(guessed_letters)}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Target Word: {target_word}', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Sign Language Game', frame) # Oyun ekranını göster

            # Eğer 'q' tuşuna basılırsa oyunu kapat
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # Kamerayı release et ve tüm pencereleri kapat
    cap.release()
    cv2.destroyAllWindows()

# Programın başlatılması
main_window()
