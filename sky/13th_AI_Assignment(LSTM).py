import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split

# 1. text_data.txt 파일 불러오기
data_path = "sky/text_data.txt"  # 텍스트 파일 경로
df = pd.read_csv(data_path, encoding='utf-8')  # 문장, label 컬럼 포함되어야 함

# 2. 토큰화 및 시퀀스 처리
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['sentence'])
sequences = tokenizer.texts_to_sequences(df['sentence'])

vocab_size = len(tokenizer.word_index) + 1
max_len = max(len(seq) for seq in sequences)
padded = pad_sequences(sequences, maxlen=max_len, padding='post')

# 3. 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(padded, df['label'], test_size=0.2, random_state=42)

# 4. LSTM 모델 구성
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=32, input_length=max_len))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 5. 모델 학습
model.fit(X_train, y_train, epochs=10, batch_size=2, validation_split=0.2)

# 6. 평가
loss, acc = model.evaluate(X_test, y_test)
print(f"테스트 정확도: {acc:.2f}")

# 7. 새로운 문장 예측
new_sentences = [
    "너무 행복하고 기분이 좋다.",
    "시험을 망쳐서 우울했다.",
    "산책이 상쾌했다.",
    "이 영화는 너무 지루했다."
]
new_seq = tokenizer.texts_to_sequences(new_sentences)
new_pad = pad_sequences(new_seq, maxlen=max_len, padding='post')
pred = model.predict(new_pad)

for sent, p in zip(new_sentences, pred):
    print(f"'{sent}' → {'긍정' if p > 0.5 else '부정'} ({p[0]:.2f})")
