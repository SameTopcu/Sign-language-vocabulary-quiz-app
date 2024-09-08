import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import tkinter as tk
from tkinter import messagebox
import subprocess
import sqlite3
import pyttsx3

# SQLite veritabanı bağlantısı oluştur
conn = sqlite3.connect('scores.db')
c = conn.cursor()

# Skorlar tablosunu oluştur
c.execute('''CREATE TABLE IF NOT EXISTS scores
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              score INTEGER,
              datetime DATETIME DEFAULT CURRENT_TIMESTAMP)''')
conn.commit()

# pyttsx3 motorunu başlat
engine = pyttsx3.init()

# GUI için ana pencere fonksiyonu tanımlanıyor
def main_window():
    root = tk.Tk()
    root.title("Sign Language Recognition Game")
    root.geometry("300x400")  # Pencere boyutunu ayarlayın

    # Oyun skorlarını gösteren fonksiyon
    def show_scores():
        scores = c.execute("SELECT * FROM scores ORDER BY score DESC").fetchall()
        score_text = ""
        for i, (score_id, score, datetime) in enumerate(scores, start=1):
            score_text += f"{i}. Score: {score}, Date: {datetime}\n"
        messagebox.showinfo("Scores", score_text)

    # En iyi 5 skoru gösteren fonksiyon
    def show_top_scores():
        top_scores = c.execute("SELECT * FROM scores ORDER BY score DESC LIMIT 5").fetchall()
        top_scores_text = "Top 5 Scores:\n"
        for i, (score_id, score, datetime) in enumerate(top_scores, start=1):
            top_scores_text += f"{i}. Score: {score}, Date: {datetime}\n"
        messagebox.showinfo("Top Scores", top_scores_text)

    # Oyunu başlatma fonksiyonu
    def start_game():
        root.withdraw()
        start_sign_language_game(root)

    # Hakkında bilgisi gösteren fonksiyon
    def about():
        messagebox.showinfo("INFO", "Sign Language Recognition Quiz Game Used Tensorflow Keras Machine Learning Libraries.")

    # Oyunu sonlandırma fonksiyonu
    def exit_game():
        root.destroy()

    # Real-time test başlatma fonksiyonu
    def start_real_time_test():
        subprocess.run(["python", "real_time_test.py"])

    # Butonları oluşturun
    tk.Button(root, text="Real-time Test", command=start_real_time_test, height=2, width=15, bg="purple", fg="white", font=("Arial", 12, "bold")).pack(pady=10)
    tk.Button(root, text="Quiz", command=start_game, height=2, width=15, bg="green", fg="white", font=("Arial", 12, "bold")).pack(pady=10)
    tk.Button(root, text="Scores", command=show_scores, height=2, width=15, bg="orange", fg="white", font=("Arial", 12, "bold")).pack(pady=10)
    tk.Button(root, text="INFO", command=about, height=2, width=15, bg="blue", fg="white", font=("Arial", 12, "bold")).pack(pady=10)
    tk.Button(root, text="QUIT", command=exit_game, height=2, width=15, bg="red", fg="white", font=("Arial", 12, "bold")).pack(pady=10)

    root.mainloop()

# İşaret dili tanıma oyununu başlatan fonksiyon
def start_sign_language_game(main_window):
    model = tf.keras.models.load_model('sign_language_model.h5')
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    words = ["LEG", "SAY", "CAR", "NET", "LAN","KEY","AND","ANY","PRO", "KEY", "CUP", "CAKE", "GOAT", "RED", "BAG"]
    words_cycle = iter(words)

    target_word = next(words_cycle)
    guessed_letters = ['_' for _ in target_word]

    current_index = 0
    letter_time_limit = 12
    letter_start_time = time.time()

    label_map = 'ABCDEFGHIKLMNOPQRSTUVWXY'
    label_map = {i: label_map[i] for i in range(len(label_map))}

    cap = cv2.VideoCapture(0)

    mp_start_t = time.time()
    score = 0
    start_time = time.time()
    limit = 60 #SÜRE AYARLAMASI

    holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    # Skor ekleme fonksiyonu
    def add_score(score):
        c.execute("INSERT INTO scores (score) VALUES (?)", (score,))
        conn.commit()

    def mediapipe_predict(hand_image):
        hand_image_rgb = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)
        results = holistic.process(hand_image_rgb)
        if results.left_hand_landmarks or results.right_hand_landmarks:
            if time.time() - mp_start_t > 12:
                return True
            return False
        return False

    predicted_letter = '?'
    probability = 0.0

    # Oyunun başlangıcında ilk kelimeyi sesli söyle
    engine.say(f"Next word is {target_word}")
    engine.runAndWait()

    print(f"Let's start with the word {target_word}")

    game_start_time = time.time()

    while True:
        elapsed_game_time = time.time() - game_start_time
        if elapsed_game_time > limit:
            add_score(score)
            engine.say(f"You scored {score} points.")
            engine.runAndWait()  # Skoru sesli olarak bildir
            messagebox.showinfo("Time's up!", f"Score: {score}\nYou have run out of time for the quiz.")
            break
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            hand_detected = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    h, w, _ = frame.shape
                    min_x = min_y = max_x = max_y = 0
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

                    if max_x - min_x > 0 and max_y - min_y > 0:
                        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
                        hand_image = frame[min_y:max_y, min_x:max_x]
                        if hand_image.size > 0:
                            hand_image = cv2.resize(hand_image, (28, 28), interpolation=cv2.INTER_AREA)
                            hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)
                            hand_image = hand_image.reshape(1, 28, 28, 1) / 255.0
                            prediction = model.predict(hand_image)
                            class_index = np.argmax(prediction)
                            predicted_letter = label_map.get(class_index, '?')
                            probability = np.max(prediction)
                            hand_detected = True

                            if current_index < len(target_word) and predicted_letter == target_word[current_index] and probability > 0.7:
                                guessed_letters[current_index] = predicted_letter
                                current_index += 1
                                letter_start_time = time.time()
                                mp_start_t = time.time()

                            if current_index < len(target_word) and time.time() - letter_start_time > letter_time_limit:
                                guessed_letters[current_index] = target_word[current_index]
                                current_index += 1
                                letter_start_time = time.time()

            elapsed_time = time.time() - start_time
            if elapsed_time > limit:
                hand_image_rgb = frame[min_y:max_y, min_x:max_x]
                if hand_image_rgb.size > 0:
                    if mediapipe_predict(hand_image_rgb):
                        score += 1
                        engine.say(target_word)  # Kelimeyi sesli oku
                        engine.runAndWait()
                        try:
                            target_word = next(words_cycle)
                        except StopIteration:
                            print("No more words left. Game over.")
                            break
                        guessed_letters = ['_' for _ in target_word]
                        current_index = 0
                        print(f"New word: {target_word}")
                start_time = time.time()

            if current_index == len(target_word):
                print(target_word)
                score += 1
                engine.say(target_word)  # Kelimeyi sesli oku
                engine.runAndWait()
                try:
                    target_word = next(words_cycle)
                except StopIteration:
                    print("No more words left. Game over.")
                    break
                guessed_letters = ['_' for _ in target_word]
                current_index = 0
                letter_start_time = time.time()
                # Yeni kelimeyi sesli olarak belirt
                engine.say(f"Next word is {target_word}")
                engine.runAndWait()
                print(f"Next word: {target_word}")

            cv2.putText(frame, f'Score: {score}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f'Current Word: {" ".join(guessed_letters)}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Target Word: {target_word}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Sign Language Game', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    main_window.deiconify()

# Programın başlatılması
main_window()
