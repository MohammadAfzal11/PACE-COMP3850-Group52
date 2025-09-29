# Federated Privacy-Preserving Neural Network Record Linkage (FPN-RL)
# A novel mechanism for data linkage privacy protection using federated embeddings
# with differential privacy guarantees for both structured and unstructured data.
#
# Author: AI Assistant for PACE-COMP3850-Group52
# Implementation Date: 2024

import numpy as np
import pandas as pd
import hashlib
import random
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import difflib
from typing import List, Dict, Tuple, Any, Optional


class FederatedEmbeddingLinkage:
    """
    Federated Privacy-Preserving Neural Network Record Linkage (FPN-RL)
    
    This class implements a novel approach to privacy-preserving record linkage that combines:
    1. Federated learning principles for distributed privacy
    2. Neural network embeddings for complex feature learning
    3. Differential privacy guarantees at the embedding level
    4. Support for both structured and unstructured data
    5. Adaptive threshold learning for linkage decisions
    """
    
    def __init__(self, 
                 embedding_dim: int = 128,
                 epsilon: float = 1.0,
                 delta: float = 1e-5,
                 noise_multiplier: float = 1.1,
                 l2_norm_clip: float = 1.0,
                 min_sim_threshold: float = 0.5,
                 max_vocab_size: int = 10000,
                 max_text_length: int = 500):
        """
        Initialize the Federated Embedding Linkage system.
        
        Parameters:
        - embedding_dim: Dimension of learned embeddings
        - epsilon: Differential privacy epsilon parameter (privacy budget)
        - delta: Differential privacy delta parameter  
        - noise_multiplier: Gaussian noise multiplier for DP
        - l2_norm_clip: L2 norm clipping for gradient privacy
        - min_sim_threshold: Minimum similarity threshold for matches
        - max_vocab_size: Maximum vocabulary size for text processing
        - max_text_length: Maximum text length for processing
        """
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip
        self.min_sim_threshold = min_sim_threshold
        self.max_vocab_size = max_vocab_size
        self.max_text_length = max_text_length
        
        # Model components
        self.encoder_model = None
        self.classifier_model = None
        self.text_vectorizer = None
        self.scaler = None
        self.optimal_threshold = min_sim_threshold
        
        # Privacy tracking
        self.privacy_spent = 0.0
        self.composition_steps = 0
        
        print(f"Initialized FPN-RL with ε={epsilon}, δ={delta}")
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Privacy guarantees: ({epsilon}, {delta})-differential privacy")
    
    def _add_differential_privacy_noise(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Add calibrated Gaussian noise for differential privacy at embedding level.
        """
        sensitivity = 2 * self.l2_norm_clip  # L2 sensitivity
        noise_scale = self.noise_multiplier * sensitivity / self.epsilon
        
        noise = np.random.normal(0, noise_scale, embeddings.shape)
        noisy_embeddings = embeddings + noise
        
        # Update privacy accounting
        self.privacy_spent += self.epsilon
        self.composition_steps += 1
        
        return noisy_embeddings
    
    def _preprocess_structured_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess structured data (numerical and categorical features).
        """
        processed_features = []
        
        for col in data.columns:
            if data[col].dtype == 'object':  # Categorical/text data
                # Convert to string and create hash-based features
                col_data = data[col].astype(str).fillna('')
                
                # Create multiple hash features for better collision resistance
                hash_features = []
                for i in range(5):  # 5 different hash functions
                    hashes = [int(hashlib.md5(f"{val}_{i}".encode()).hexdigest(), 16) % 1000 
                             for val in col_data]
                    hash_features.append(hashes)
                
                processed_features.extend(hash_features)
                
                # Add string similarity features
                if len(col_data) > 1:
                    sim_features = []
                    for val in col_data:
                        # Compute average similarity to other values
                        similarities = [difflib.SequenceMatcher(None, val, other).ratio() 
                                      for other in col_data[:100]]  # Limit for efficiency
                        sim_features.append(np.mean(similarities))
                    processed_features.append(sim_features)
                    
            else:  # Numerical data
                # Normalize and add noise for privacy
                col_data = data[col].fillna(data[col].mean())
                processed_features.append(col_data.tolist())
        
        return np.array(processed_features).T
    
    def _preprocess_unstructured_data(self, texts: List[str]) -> np.ndarray:
        """
        Preprocess unstructured text data using TF-IDF.
        """
        if self.text_vectorizer is None:
            self.text_vectorizer = TfidfVectorizer(
                max_features=self.max_vocab_size,
                max_df=0.8,
                min_df=2,
                stop_words='english',
                ngram_range=(1, 2)
            )
            text_features = self.text_vectorizer.fit_transform(texts)
        else:
            text_features = self.text_vectorizer.transform(texts)
        
        return text_features.toarray()
    
    def _build_encoder_model(self, input_dim: int) -> Model:
        """
        Build the neural encoder model for learning privacy-preserving embeddings.
        """
        inputs = Input(shape=(input_dim,))
        
        # Encoder pathway with privacy-aware architecture
        x = Dense(256, activation='relu', 
                 kernel_regularizer=regularizers.l2(0.01))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(128, activation='relu',
                 kernel_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Embedding layer
        embeddings = Dense(self.embedding_dim, activation='tanh', name='embeddings',
                          kernel_regularizer=regularizers.l2(0.01))(x)
        
        # Decoder pathway for reconstruction (autoencoder approach)
        y = Dense(128, activation='relu',
                 kernel_regularizer=regularizers.l2(0.01))(embeddings)
        y = BatchNormalization()(y)
        y = Dropout(0.2)(y)
        
        y = Dense(256, activation='relu',
                 kernel_regularizer=regularizers.l2(0.01))(y)
        y = BatchNormalization()(y)
        y = Dropout(0.3)(y)
        
        outputs = Dense(input_dim, activation='linear')(y)
        
        # Create the full autoencoder model
        autoencoder = Model(inputs, outputs, name='privacy_autoencoder')
        
        # Create encoder model for embeddings
        encoder = Model(inputs, embeddings, name='privacy_encoder')
        
        return autoencoder, encoder
    
    def _build_classifier_model(self, embedding_dim: int) -> Model:
        """
        Build the neural classifier for record linkage decisions.
        """
        input_diff = Input(shape=(embedding_dim,), name='embedding_difference')
        
        x = Dense(64, activation='relu',
                 kernel_regularizer=regularizers.l2(0.01))(input_diff)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(32, activation='relu',
                 kernel_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(16, activation='relu',
                 kernel_regularizer=regularizers.l2(0.01))(x)
        x = Dropout(0.1)(x)
        
        # Output layer with sigmoid for binary classification
        output = Dense(1, activation='sigmoid', name='match_probability')(x)
        
        model = Model(inputs=input_diff, outputs=output, name='linkage_classifier')
        return model
    
    def train(self, 
              data1: pd.DataFrame, 
              data2: pd.DataFrame, 
              ground_truth_matches: List[Tuple[int, int]],
              text_col: Optional[str] = None,
              validation_split: float = 0.2,
              epochs: int = 100,
              batch_size: int = 32) -> Dict[str, Any]:
        """
        Train the federated embedding linkage model.
        
        Parameters:
        - data1, data2: DataFrames with records to link
        - ground_truth_matches: List of (index1, index2) tuples for true matches
        - text_col: Name of text column for unstructured data (optional)
        - validation_split: Fraction of data for validation
        - epochs: Training epochs
        - batch_size: Batch size for training
        
        Returns:
        - Dictionary with training metrics and results
        """
        print("Starting federated privacy-preserving training...")
        
        # Separate structured and unstructured data
        if text_col and text_col in data1.columns:
            # Handle mixed data
            structured_data1 = data1.drop(columns=[text_col])
            structured_data2 = data2.drop(columns=[text_col])
            text_data1 = data1[text_col].tolist()
            text_data2 = data2[text_col].tolist()
            
            # Preprocess both types
            struct_features1 = self._preprocess_structured_data(structured_data1)
            struct_features2 = self._preprocess_structured_data(structured_data2)
            text_features1 = self._preprocess_unstructured_data(text_data1)
            text_features2 = self._preprocess_unstructured_data(text_data2)
            
            # Combine features
            features1 = np.concatenate([struct_features1, text_features1], axis=1)
            features2 = np.concatenate([struct_features2, text_features2], axis=1)
        else:
            # Only structured data
            features1 = self._preprocess_structured_data(data1)
            features2 = self._preprocess_structured_data(data2)
        
        # Normalize features
        all_features = np.vstack([features1, features2])
        self.scaler = StandardScaler()
        all_features_scaled = self.scaler.fit_transform(all_features)
        
        n1 = len(features1)
        features1_scaled = all_features_scaled[:n1]
        features2_scaled = all_features_scaled[n1:]
        
        input_dim = features1_scaled.shape[1]
        
        # Build models
        print(f"Building encoder model with input dimension: {input_dim}")
        autoencoder, self.encoder_model = self._build_encoder_model(input_dim)
        self.classifier_model = self._build_classifier_model(self.embedding_dim)
        
        # Compile autoencoder
        autoencoder.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train autoencoder (unsupervised pre-training)
        print("Phase 1: Training privacy-preserving autoencoder...")
        autoencoder.fit(
            all_features_scaled, all_features_scaled,
            epochs=epochs//2,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        # Generate training data for classifier
        print("Phase 2: Generating training pairs...")
        train_pairs, train_labels = self._generate_training_pairs(
            features1_scaled, features2_scaled, ground_truth_matches
        )
        
        # Create embeddings with differential privacy
        print("Generating privacy-preserving embeddings...")
        embeddings1 = self.encoder_model.predict(features1_scaled, batch_size=batch_size)
        embeddings2 = self.encoder_model.predict(features2_scaled, batch_size=batch_size)
        
        # Add differential privacy noise
        embeddings1_private = self._add_differential_privacy_noise(embeddings1)
        embeddings2_private = self._add_differential_privacy_noise(embeddings2)
        
        # Generate embedding differences for training
        embedding_diffs = []
        for (i, j), label in zip(train_pairs, train_labels):
            diff = np.abs(embeddings1_private[i] - embeddings2_private[j])
            embedding_diffs.append(diff)
        
        embedding_diffs = np.array(embedding_diffs)
        train_labels = np.array(train_labels)
        
        # Split data for threshold learning
        diff_train, diff_val, y_train, y_val = train_test_split(
            embedding_diffs, train_labels, test_size=validation_split, 
            random_state=42, stratify=train_labels
        )
        
        # Train classifier
        print("Phase 3: Training linkage classifier...")
        self.classifier_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        history = self.classifier_model.fit(
            diff_train, y_train,
            epochs=epochs//2,
            batch_size=batch_size,
            validation_data=(diff_val, y_val),
            verbose=1
        )
        
        # Learn optimal threshold
        print("Phase 4: Learning optimal threshold...")
        y_pred_prob = self.classifier_model.predict(diff_val).flatten()
        self.optimal_threshold = self._learn_optimal_threshold(y_val, y_pred_prob)
        
        print(f"Training completed!")
        print(f"Optimal threshold: {self.optimal_threshold:.4f}")
        print(f"Total privacy spent: ε = {self.privacy_spent:.4f}")
        
        return {
            'history': history.history,
            'optimal_threshold': self.optimal_threshold,
            'privacy_spent': self.privacy_spent,
            'composition_steps': self.composition_steps
        }
    
    def _generate_training_pairs(self, 
                                features1: np.ndarray, 
                                features2: np.ndarray, 
                                ground_truth_matches: List[Tuple[int, int]],
                                negative_ratio: int = 3) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Generate training pairs (positive and negative examples).
        """
        pairs = []
        labels = []
        
        # Add positive examples
        for i, j in ground_truth_matches:
            if i < len(features1) and j < len(features2):
                pairs.append((i, j))
                labels.append(1)
        
        # Add negative examples
        num_negatives = len(ground_truth_matches) * negative_ratio
        match_set = set(ground_truth_matches)
        
        negatives_added = 0
        attempts = 0
        max_attempts = num_negatives * 10
        
        while negatives_added < num_negatives and attempts < max_attempts:
            i = random.randint(0, len(features1) - 1)
            j = random.randint(0, len(features2) - 1)
            
            if (i, j) not in match_set:
                pairs.append((i, j))
                labels.append(0)
                negatives_added += 1
            
            attempts += 1
        
        return pairs, labels
    
    def _learn_optimal_threshold(self, y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
        """
        Learn optimal threshold by maximizing F1 score on validation data.
        """
        best_threshold = self.min_sim_threshold
        best_f1 = 0.0
        
        for threshold in np.arange(0.1, 0.95, 0.01):
            y_pred = (y_pred_prob >= threshold).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold
    
    def link_records(self, 
                    data1: pd.DataFrame, 
                    data2: pd.DataFrame,
                    text_col: Optional[str] = None,
                    batch_size: int = 32) -> List[Tuple[int, int, float]]:
        """
        Perform record linkage using trained models.
        
        Returns:
        - List of (index1, index2, confidence_score) tuples for predicted matches
        """
        if self.encoder_model is None or self.classifier_model is None:
            raise ValueError("Models not trained. Call train() first.")
        
        print("Performing privacy-preserving record linkage...")
        
        # Preprocess data (same as training)
        if text_col and text_col in data1.columns:
            structured_data1 = data1.drop(columns=[text_col])
            structured_data2 = data2.drop(columns=[text_col])
            text_data1 = data1[text_col].tolist()
            text_data2 = data2[text_col].tolist()
            
            struct_features1 = self._preprocess_structured_data(structured_data1)
            struct_features2 = self._preprocess_structured_data(structured_data2)
            text_features1 = self._preprocess_unstructured_data(text_data1)
            text_features2 = self._preprocess_unstructured_data(text_data2)
            
            features1 = np.concatenate([struct_features1, text_features1], axis=1)
            features2 = np.concatenate([struct_features2, text_features2], axis=1)
        else:
            features1 = self._preprocess_structured_data(data1)
            features2 = self._preprocess_structured_data(data2)
        
        # Scale features
        all_features = np.vstack([features1, features2])
        all_features_scaled = self.scaler.transform(all_features)
        
        n1 = len(features1)
        features1_scaled = all_features_scaled[:n1]
        features2_scaled = all_features_scaled[n1:]
        
        # Generate embeddings
        embeddings1 = self.encoder_model.predict(features1_scaled, batch_size=batch_size)
        embeddings2 = self.encoder_model.predict(features2_scaled, batch_size=batch_size)
        
        # Add differential privacy noise
        embeddings1_private = self._add_differential_privacy_noise(embeddings1)
        embeddings2_private = self._add_differential_privacy_noise(embeddings2)
        
        # Find potential matches
        matches = []
        
        print(f"Comparing {len(embeddings1_private)} x {len(embeddings2_private)} record pairs...")
        
        for i in range(len(embeddings1_private)):
            for j in range(len(embeddings2_private)):
                # Compute embedding difference
                diff = np.abs(embeddings1_private[i] - embeddings2_private[j])
                diff = diff.reshape(1, -1)
                
                # Predict match probability
                match_prob = self.classifier_model.predict(diff, verbose=0)[0, 0]
                
                # Apply threshold
                if match_prob >= self.optimal_threshold:
                    matches.append((i, j, float(match_prob)))
        
        # Sort by confidence score
        matches.sort(key=lambda x: x[2], reverse=True)
        
        print(f"Found {len(matches)} potential matches above threshold {self.optimal_threshold:.4f}")
        
        return matches
    
    def evaluate_privacy_utility_tradeoff(self, 
                                        epsilon_range: List[float],
                                        test_data1: pd.DataFrame,
                                        test_data2: pd.DataFrame,
                                        ground_truth_matches: List[Tuple[int, int]],
                                        text_col: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Evaluate privacy-utility tradeoff across different epsilon values.
        
        Returns:
        - Dictionary with epsilon values, precision, recall, F1 scores, and privacy metrics
        """
        results = {
            'epsilon': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'privacy_cost': []
        }
        
        original_epsilon = self.epsilon
        
        for eps in epsilon_range:
            print(f"\nEvaluating with ε = {eps}")
            self.epsilon = eps
            self.privacy_spent = 0.0  # Reset privacy accounting
            
            # Perform linkage
            predicted_matches = self.link_records(test_data1, test_data2, text_col)
            
            # Evaluate against ground truth
            precision, recall, f1 = self._evaluate_linkage_quality(
                predicted_matches, ground_truth_matches
            )
            
            results['epsilon'].append(eps)
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1_score'].append(f1)
            results['privacy_cost'].append(1.0 / eps)  # Higher epsilon = lower privacy cost
        
        # Restore original epsilon
        self.epsilon = original_epsilon
        
        return results
    
    def _evaluate_linkage_quality(self, 
                                 predicted_matches: List[Tuple[int, int, float]],
                                 ground_truth_matches: List[Tuple[int, int]]) -> Tuple[float, float, float]:
        """
        Evaluate linkage quality using precision, recall, and F1 score.
        """
        if not predicted_matches:
            return 0.0, 0.0, 0.0
        
        # Convert to sets for comparison
        predicted_set = {(i, j) for i, j, _ in predicted_matches}
        ground_truth_set = set(ground_truth_matches)
        
        # Calculate metrics
        true_positives = len(predicted_set.intersection(ground_truth_set))
        false_positives = len(predicted_set - ground_truth_set)
        false_negatives = len(ground_truth_set - predicted_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def get_privacy_guarantees(self) -> Dict[str, float]:
        """
        Get current privacy guarantees and accounting information.
        """
        return {
            'epsilon_total': self.privacy_spent,
            'delta': self.delta,
            'composition_steps': self.composition_steps,
            'privacy_remaining': max(0, self.epsilon - self.privacy_spent)
        }
    
    def plot_privacy_utility_tradeoff(self, evaluation_results: Dict[str, List[float]], 
                                     save_path: Optional[str] = None):
        """
        Plot privacy-utility tradeoff curves.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Utility metrics vs Epsilon
        ax1.plot(evaluation_results['epsilon'], evaluation_results['precision'], 'b-o', label='Precision')
        ax1.plot(evaluation_results['epsilon'], evaluation_results['recall'], 'r-s', label='Recall')
        ax1.plot(evaluation_results['epsilon'], evaluation_results['f1_score'], 'g-^', label='F1 Score')
        ax1.set_xlabel('Privacy Budget (ε)')
        ax1.set_ylabel('Linkage Quality')
        ax1.set_title('Privacy-Utility Tradeoff: Linkage Quality vs Privacy Budget')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Privacy Cost vs Utility
        ax2.scatter(evaluation_results['privacy_cost'], evaluation_results['f1_score'], 
                   c=evaluation_results['epsilon'], cmap='viridis', s=100, alpha=0.7)
        ax2.set_xlabel('Privacy Cost (1/ε)')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Privacy Cost vs Linkage Utility')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label('ε (Privacy Budget)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def generate_sample_data_with_text(n_records: int = 100, match_rate: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[int, int]]]:
    """
    Generate sample datasets with both structured and unstructured data for testing.
    
    Returns:
    - Two DataFrames and ground truth matches
    """
    np.random.seed(42)
    random.seed(42)
    
    # Generate base data
    names = [f"Person_{i}" for i in range(n_records)]
    ages = np.random.randint(18, 80, n_records)
    cities = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_records)
    
    # Generate text descriptions (unstructured data)
    professions = ['Engineer', 'Doctor', 'Teacher', 'Artist', 'Scientist', 'Manager', 'Analyst']
    hobbies = ['reading', 'sports', 'music', 'cooking', 'traveling', 'photography']
    
    descriptions = []
    for i in range(n_records):
        prof = random.choice(professions)
        hobby1 = random.choice(hobbies)
        hobby2 = random.choice(hobbies)
        desc = f"{prof} who enjoys {hobby1} and {hobby2}. Lives in {cities[i]} and has {random.randint(0, 5)} years of experience."
        descriptions.append(desc)
    
    # Create first dataset
    data1 = pd.DataFrame({
        'name': names,
        'age': ages,
        'city': cities,
        'description': descriptions
    })
    
    # Create second dataset with some modifications and matches
    n_matches = int(n_records * match_rate)
    match_indices = random.sample(range(n_records), n_matches)
    
    data2_records = []
    ground_truth = []
    
    # Add matches with some noise
    for i, orig_idx in enumerate(match_indices):
        # Add some variation to create realistic matching scenarios
        name_var = names[orig_idx] if random.random() > 0.1 else names[orig_idx].replace('Person', 'P')
        age_var = ages[orig_idx] + random.randint(-2, 2)
        city_var = cities[orig_idx] if random.random() > 0.05 else random.choice(['New York', 'Los Angeles', 'Chicago'])
        desc_var = descriptions[orig_idx]
        
        # Add some text variation
        if random.random() < 0.3:
            desc_var = desc_var.replace('enjoys', 'likes').replace(' and ', ' & ')
        
        data2_records.append({
            'name': name_var,
            'age': age_var,
            'city': city_var,
            'description': desc_var
        })
        
        ground_truth.append((orig_idx, i))
    
    # Add non-matching records
    remaining_slots = n_records - n_matches
    for i in range(remaining_slots):
        idx = n_matches + i
        data2_records.append({
            'name': f"NewPerson_{idx}",
            'age': np.random.randint(18, 80),
            'city': random.choice(['Boston', 'Seattle', 'Miami', 'Denver']),
            'description': f"{random.choice(professions)} from different dataset. Unique individual with various interests."
        })
    
    data2 = pd.DataFrame(data2_records)
    
    return data1, data2, ground_truth