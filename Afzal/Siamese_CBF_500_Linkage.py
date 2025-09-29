import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import hashlib
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import os
import time

class CountingBloomFilter:
    """Counting Bloom Filter implementation for privacy-preserving record linkage"""
    
    def __init__(self, bf_len=1000, num_hash_func=10, q=2):
        self.bf_len = bf_len
        self.num_hash_func = num_hash_func
        self.q = q
        self.h1 = hashlib.sha1
        self.h2 = hashlib.md5
    
    def get_qgrams(self, text):
        """Extract q-grams from text"""
        if pd.isna(text):
            text = ""
        text = str(text).lower().strip()
        if len(text) < self.q:
            return [text.ljust(self.q, ' ')]
        return [text[i:i+self.q] for i in range(len(text) - self.q + 1)]
    
    def encode_record(self, record_dict, fields=['first_name', 'last_name', 'city']):
        """Encode a record into a Counting Bloom Filter"""
        cbf = np.zeros(self.bf_len, dtype=int)
        
        for field in fields:
            if field in record_dict and not pd.isna(record_dict[field]):
                qgrams = self.get_qgrams(record_dict[field])
                for qgram in qgrams:
                    # Use double hashing
                    hex_str1 = self.h1(qgram.encode('utf-8')).hexdigest()
                    int1 = int(hex_str1, 16)
                    hex_str2 = self.h2(qgram.encode('utf-8')).hexdigest()
                    int2 = int(hex_str2, 16)
                    
                    for i in range(self.num_hash_func):
                        gi = (int1 + i * int2) % self.bf_len
                        cbf[gi] += 1  # Increment count instead of setting to 1
        
        return cbf
    
    def calculate_similarity(self, cbf1, cbf2):
        """Calculate Dice coefficient similarity between two CBFs"""
        sum1 = np.sum(cbf1)
        sum2 = np.sum(cbf2)
        common = np.sum(np.minimum(cbf1, cbf2))
        
        if sum1 + sum2 == 0:
            return 0.0
        
        dice_sim = (2.0 * common) / (sum1 + sum2)
        return dice_sim

class SiameseCBFModel:
    """Enhanced Siamese Neural Network for CBF-encoded records"""
    
    def __init__(self, cbf_length=1000, embedding_dim=128):
        self.cbf_length = cbf_length
        self.embedding_dim = embedding_dim
        self.model = None
        self.encoder = None
        
    def build_model(self):
        """Build enhanced Siamese Neural Network architecture for larger datasets"""
        # Input layers for CBF vectors
        input_1 = layers.Input(shape=(self.cbf_length,), name='cbf_1')
        input_2 = layers.Input(shape=(self.cbf_length,), name='cbf_2')
        
        # Enhanced shared encoder network for larger datasets
        encoder = models.Sequential([
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.embedding_dim, activation='relu', name='embedding'),
            layers.BatchNormalization(),
            layers.Dropout(0.2)
        ], name='shared_encoder')
        
        # Encode both inputs
        encoded_1 = encoder(input_1)
        encoded_2 = encoder(input_2)
        
        # Enhanced feature combination
        diff = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([encoded_1, encoded_2])
        mult = layers.Lambda(lambda x: x[0] * x[1])([encoded_1, encoded_2])
        cosine = layers.Lambda(lambda x: tf.nn.l2_normalize(x[0], axis=1) * tf.nn.l2_normalize(x[1], axis=1))([encoded_1, encoded_2])
        
        # Concatenate all features
        combined = layers.Concatenate()([diff, mult, cosine])
        
        # Enhanced classification head
        classifier = models.Sequential([
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ], name='classifier')
        
        output = classifier(combined)
        
        # Create full model
        self.model = models.Model(inputs=[input_1, input_2], outputs=output)
        self.encoder = encoder
        
        return self.model
    
    def compile_model(self, learning_rate=0.001, class_weights=None):
        """Compile the model with optional class weights"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        self.class_weights = class_weights
    
    def train(self, cbf1_train, cbf2_train, y_train, 
              cbf1_val=None, cbf2_val=None, y_val=None,
              epochs=100, batch_size=64):
        """Train the Siamese model with enhanced callbacks"""
        validation_data = None
        if cbf1_val is not None:
            validation_data = ([cbf1_val, cbf2_val], y_val)
        
        # Enhanced callbacks for larger datasets
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=8, min_lr=0.00001
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_siamese_500.h5', save_best_only=True, monitor='val_loss'
            )
        ]
        
        history = self.model.fit(
            [cbf1_train, cbf2_train], y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=self.class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, cbf1, cbf2):
        """Make predictions"""
        return self.model.predict([cbf1, cbf2], verbose=0)

def load_and_preprocess_data(alice_file, bob_file):
    """Load and preprocess Alice and Bob datasets"""
    try:
        # Load datasets
        alice_df = pd.read_csv(alice_file)
        bob_df = pd.read_csv(bob_file)
        
        print(f"Alice columns: {alice_df.columns.tolist()}")
        print(f"Bob columns: {bob_df.columns.tolist()}")
        
        # Handle different column names
        if 'givenname' in bob_df.columns:
            bob_df = bob_df.rename(columns={'givenname': 'first_name'})
        if 'surname' in bob_df.columns:
            bob_df = bob_df.rename(columns={'surname': 'last_name'})
        if 'suburb' in bob_df.columns:
            bob_df = bob_df.rename(columns={'suburb': 'city'})
        
        # Add consistent record ID column
        if 'rec_id' in alice_df.columns:
            alice_df['record_id'] = alice_df['rec_id']
        elif any(col for col in alice_df.columns if 'id' in col.lower()):
            id_col = [col for col in alice_df.columns if 'id' in col.lower()][0]
            alice_df['record_id'] = alice_df[id_col]
        
        if 'recid' in bob_df.columns:
            bob_df['record_id'] = bob_df['recid']
        elif any(col for col in bob_df.columns if 'id' in col.lower()):
            id_col = [col for col in bob_df.columns if 'id' in col.lower()][0]
            bob_df['record_id'] = bob_df[id_col]
        
        return alice_df, bob_df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def create_balanced_training_pairs(alice_df, bob_df, num_pairs=10000):
    """Create balanced training pairs with equal positive and negative samples - optimized for 500 records"""
    pairs = []
    labels = []
    
    # Get common record IDs for positive pairs
    alice_ids = set(alice_df['record_id'].astype(str))
    bob_ids = set(bob_df['record_id'].astype(str))
    common_ids = alice_ids.intersection(bob_ids)
    
    print(f"Found {len(common_ids)} matching record pairs")
    
    # Calculate balanced sizes - use more positive pairs for 500 records
    max_positive = min(len(common_ids), num_pairs // 2)
    num_negative = max_positive  # Equal number of negative pairs
    
    print(f"Creating {max_positive} positive and {num_negative} negative pairs")
    
    # Create positive pairs (matches)
    positive_count = 0
    for record_id in list(common_ids)[:max_positive]:
        alice_records = alice_df[alice_df['record_id'].astype(str) == record_id]
        bob_records = bob_df[bob_df['record_id'].astype(str) == record_id]
        
        if len(alice_records) > 0 and len(bob_records) > 0:
            alice_record = alice_records.iloc[0]
            bob_record = bob_records.iloc[0]
            
            pairs.append((alice_record.to_dict(), bob_record.to_dict()))
            labels.append(1)  # Match
            positive_count += 1
    
    # Create negative pairs (non-matches) - optimized sampling for larger dataset
    negative_count = 0
    max_attempts = num_negative * 5  # Reduced attempts ratio for efficiency
    attempts = 0
    
    # Pre-sample for efficiency
    alice_sample = alice_df.sample(n=min(num_negative * 2, len(alice_df)), replace=False)
    bob_sample = bob_df.sample(n=min(num_negative * 2, len(bob_df)), replace=False)
    
    alice_idx = 0
    bob_idx = 0
    
    while negative_count < num_negative and attempts < max_attempts:
        # Use pre-sampled records for efficiency
        alice_record = alice_sample.iloc[alice_idx % len(alice_sample)]
        bob_record = bob_sample.iloc[bob_idx % len(bob_sample)]
        
        # Ensure they're not actually matches
        if str(alice_record['record_id']) != str(bob_record['record_id']):
            pairs.append((alice_record.to_dict(), bob_record.to_dict()))
            labels.append(0)  # Non-match
            negative_count += 1
        
        alice_idx += 1
        bob_idx += 1
        attempts += 1
    
    print(f"Created {len(pairs)} training pairs ({positive_count} positive, {negative_count} negative)")
    print(f"Class balance: {positive_count/(positive_count + negative_count):.2%} positive")
    
    return pairs, np.array(labels)

def main():
    """Main execution function for 500-record experiment"""
    print("=== Siamese Neural Network + CBF: 500 Records Experiment ===\n")
    
    start_time = time.time()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 1. Load and preprocess data - CHANGED TO 500 RECORDS
    print("1. Loading 500-record datasets...")
    alice_df, bob_df = load_and_preprocess_data('Alice_numrec_500_corr_25.csv', 'Bob_numrec_500_corr_25.csv')
    print(f"Alice records: {len(alice_df)}, Bob records: {len(bob_df)}")
    
    # 2. Initialize CBF encoder
    print("\n2. Initializing Counting Bloom Filter...")
    cbf_encoder = CountingBloomFilter(bf_len=1000, num_hash_func=10, q=2)
    
    # 3. Create BALANCED training pairs - INCREASED FOR 500 RECORDS
    print("\n3. Creating balanced training pairs...")
    pairs, labels = create_balanced_training_pairs(alice_df, bob_df, num_pairs=6000)  # Increased
    
    # 4. Encode pairs into CBF vectors
    print("\n4. Encoding records into CBF vectors...")
    cbf1_list = []
    cbf2_list = []
    
    for i, pair in enumerate(pairs):
        alice_record, bob_record = pair
        cbf1 = cbf_encoder.encode_record(alice_record)
        cbf2 = cbf_encoder.encode_record(bob_record)
        cbf1_list.append(cbf1)
        cbf2_list.append(cbf2)
        
        if i % 500 == 0:  # Progress every 500 pairs
            print(f"Encoded {i}/{len(pairs)} pairs...")
    
    cbf1_array = np.array(cbf1_list)
    cbf2_array = np.array(cbf2_list)
    
    print(f"CBF array shapes: {cbf1_array.shape}, {cbf2_array.shape}")
    
    # 5. Split into train/test with stratification
    print("\n5. Splitting into train/test sets...")
    
    cbf1_train, cbf1_test, cbf2_train, cbf2_test, y_train, y_test = train_test_split(
        cbf1_array, cbf2_array, labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels
    )
    
    print(f"Training set: {len(y_train)} pairs")
    print(f"Test set: {len(y_test)} pairs")
    print(f"Training labels distribution: {np.bincount(y_train)}")
    print(f"Test labels distribution: {np.bincount(y_test)}")
    
    # 6. Calculate class weights
    print("\n6. Calculating class weights...")
    class_weights_array = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    class_weights = {0: class_weights_array[0], 1: class_weights_array[1]}
    print(f"Class weights: {class_weights}")
    
    # 7. Build and train Enhanced Siamese model
    print("\n7. Building Enhanced Siamese Neural Network for 500 Records...")
    siamese_model = SiameseCBFModel(cbf_length=1000, embedding_dim=128)
    model = siamese_model.build_model()
    siamese_model.compile_model(learning_rate=0.001, class_weights=class_weights)
    
    print(f"\nModel architecture:")
    model.summary()
    
    print("\n8. Training the model...")
    history = siamese_model.train(
        cbf1_train, cbf2_train, y_train,
        cbf1_test, cbf2_test, y_test,
        epochs=100,  # More epochs for larger dataset
        batch_size=64  # Larger batch size
    )
    
    # 9. Evaluate the model
    print("\n9. Evaluating the model...")
    y_pred_prob = siamese_model.predict(cbf1_test, cbf2_test)
    
    # Try different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    best_f1 = 0
    best_threshold = 0.5
    
    print("\nTesting different classification thresholds:")
    for threshold in thresholds:
        y_pred = (y_pred_prob > threshold).astype(int).flatten()
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Threshold {threshold}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Use best threshold
    y_pred = (y_pred_prob > best_threshold).astype(int).flatten()
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n=== SIAMESE FINAL RESULTS (Threshold: {best_threshold}) ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # 10. Privacy analysis (CBF similarity baseline)
    print("\n10. Privacy Analysis - CBF Similarity Baseline:")
    print("Computing CBF similarities for all test pairs...")
    
    cbf_similarities = []
    for i in range(len(cbf1_test)):
        sim = cbf_encoder.calculate_similarity(cbf1_test[i], cbf2_test[i])
        cbf_similarities.append(sim)
        if i % 100 == 0:
            print(f"Processed {i}/{len(cbf1_test)} similarities...")
    
    # Test CBF thresholds
    cbf_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    best_cbf_f1 = 0
    best_cbf_threshold = 0.3
    
    print("\nTesting CBF similarity thresholds:")
    for threshold in cbf_thresholds:
        cbf_pred = (np.array(cbf_similarities) > threshold).astype(int)
        cbf_accuracy = accuracy_score(y_test, cbf_pred)
        cbf_f1 = f1_score(y_test, cbf_pred)
        print(f"CBF Threshold {threshold}: Accuracy={cbf_accuracy:.4f}, F1={cbf_f1:.4f}")
        
        if cbf_f1 > best_cbf_f1:
            best_cbf_f1 = cbf_f1
            best_cbf_threshold = threshold
    
    print(f"\n=== CBF BASELINE RESULTS (Threshold: {best_cbf_threshold}) ===")
    cbf_pred = (np.array(cbf_similarities) > best_cbf_threshold).astype(int)
    cbf_accuracy = accuracy_score(y_test, cbf_pred)
    cbf_f1 = f1_score(y_test, cbf_pred)
    print(f"CBF Dice Similarity Baseline:")
    print(f"Accuracy: {cbf_accuracy:.4f}")
    print(f"F1 Score: {cbf_f1:.4f}")
    
    # 11. Final comparison and timing
    total_time = time.time() - start_time
    
    print(f"\n=== FINAL COMPARISON: 500 RECORDS EXPERIMENT ===")
    print(f"Dataset Size: 500 records each (Alice & Bob)")
    print(f"Training Pairs: {len(y_train)}")
    print(f"Test Pairs: {len(y_test)}")
    print(f"Total Runtime: {total_time:.2f} seconds")
    print(f"")
    print(f"Siamese + CBF: Accuracy={accuracy:.4f}, F1={f1:.4f}")
    print(f"CBF Baseline:  Accuracy={cbf_accuracy:.4f}, F1={cbf_f1:.4f}")
    print(f"Improvement:   Accuracy={accuracy-cbf_accuracy:+.4f}, F1={f1-cbf_f1:+.4f}")
    
    if accuracy > cbf_accuracy:
        print(f"🎉 SUCCESS: Siamese network outperformed CBF baseline!")
    else:
        print(f"📊 INSIGHT: CBF baseline still superior - may need more data or tuning")
    
    # 12. Save results
    results = {
        'dataset_size': 500,
        'training_pairs': len(y_train),
        'test_pairs': len(y_test),
        'siamese_accuracy': float(accuracy),
        'siamese_f1': float(f1),
        'cbf_accuracy': float(cbf_accuracy),
        'cbf_f1': float(cbf_f1),
        'best_siamese_threshold': float(best_threshold),
        'best_cbf_threshold': float(best_cbf_threshold),
        'total_runtime': float(total_time)
    }
    
    import json
    with open('results_500_records.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to 'results_500_records.json'")
    
    return siamese_model, cbf_encoder, history, results

if __name__ == "__main__":
    model, encoder, history, results = main()
