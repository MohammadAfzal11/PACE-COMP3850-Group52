import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import hashlib, matplotlib.pyplot as plt, gc

# ==========================================
# Counting Bloom Filter Encoder
# ==========================================
class CountingBloomFilter:
    def __init__(self, bf_len=1000, num_hash_func=10, q=2):
        self.bf_len = bf_len
        self.num_hash_func = num_hash_func
        self.q = q
        self.h1 = hashlib.sha1
        self.h2 = hashlib.md5

    def get_qgrams(self, text):
        if pd.isna(text):
            text = ""
        text = str(text).lower().strip()
        if len(text) < self.q:
            return [text.ljust(self.q, ' ')]
        return [text[i:i+self.q] for i in range(len(text) - self.q + 1)]

    def encode_record(self, record, fields):
        cbf = np.zeros(self.bf_len, dtype=int)
        for field in fields:
            if field in record and not pd.isna(record[field]):
                qgrams = self.get_qgrams(record[field])
                for qgram in qgrams:
                    h1_int = int(self.h1(qgram.encode()).hexdigest(), 16)
                    h2_int = int(self.h2(qgram.encode()).hexdigest(), 16)
                    for i in range(self.num_hash_func):
                        gi = (h1_int + i * h2_int) % self.bf_len
                        cbf[gi] += 1
        return cbf

# ==========================================
# Siamese Neural Network Model
# ==========================================
class SiameseCBF:
    def __init__(self, cbf_length=1000, embedding_dim=128):
        self.length = cbf_length
        self.embedding_dim = embedding_dim
        self.model = None

    def build_model(self):
        input_1 = layers.Input(shape=(self.length,), name='cbf_1')
        input_2 = layers.Input(shape=(self.length,), name='cbf_2')

        # Shared encoder
        encoder = models.Sequential([
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(self.embedding_dim, activation="relu")
        ])

        encoded_1 = encoder(input_1)
        encoded_2 = encoder(input_2)

        diff = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([encoded_1, encoded_2])
        mult = layers.Lambda(lambda x: x[0] * x[1])([encoded_1, encoded_2])
        concat = layers.Concatenate()([diff, mult])

        classifier = models.Sequential([
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ])

        output = classifier(concat)
        self.model = models.Model(inputs=[input_1, input_2], outputs=output)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                           loss="binary_crossentropy",
                           metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        return self.model


# ==========================================
# Data Handling for Unstructured Columns
# ==========================================
def generate_pairs(df, cbf_encoder, label_col="label"):
    cbf1_list, cbf2_list, y = [], [], []
    for _, row in df.iterrows():
        left = {"raw_text": row["text1"] if "text1" in row else ""}
        right = {"raw_text": row["text2"] if "text2" in row else ""}
        cbf1 = cbf_encoder.encode_record(left, fields=["raw_text"])
        cbf2 = cbf_encoder.encode_record(right, fields=["raw_text"])

        cbf1_list.append(cbf1)
        cbf2_list.append(cbf2)
        y.append(row[label_col])
    return np.array(cbf1_list), np.array(cbf2_list), np.array(y)

# ==========================================
# Main Execution
# ==========================================
def main():
    print("=== Siamese CBF Record Linkage with target_train.csv / target_test.csv ===")
    cbf_encoder = CountingBloomFilter(1000, 10, 2)
    siamese = SiameseCBF(1000)
    model = siamese.build_model()

    print("Loading training data...")
    train_df = pd.read_csv(r"C:\Users\afzal\Documents\GitHub\PACE-COMP3850-Group52\target_train.csv")
    test_df = pd.read_csv(r"C:\Users\afzal\Documents\GitHub\PACE-COMP3850-Group52\target_test.csv")

    # Generate encoded arrays
    print("Encoding training data into CBF...")
    cbf1_train, cbf2_train, y_train = generate_pairs(train_df, cbf_encoder)
    print("Encoding test data into CBF...")
    cbf1_test, cbf2_test, y_test = generate_pairs(test_df, cbf_encoder)

    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {0: weights[0], 1: weights[1]}

    print("Training Siamese model...")
    history = model.fit([cbf1_train, cbf2_train], y_train,
                        validation_data=([cbf1_test, cbf2_test], y_test),
                        epochs=50, batch_size=64, class_weight=class_weights, verbose=1)

    print("Evaluating model...")
    preds = model.predict([cbf1_test, cbf2_test])
    y_pred = (preds > 0.5).astype(int).flatten()
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Save model
    model.save("siamese_cbf_target_model.h5")

    # Plot Training History
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs'), plt.ylabel('Accuracy')
    plt.legend(), plt.show()

if __name__ == "__main__":
    main()
