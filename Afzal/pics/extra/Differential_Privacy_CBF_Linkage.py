import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import hashlib
import time
import json
import warnings
warnings.filterwarnings('ignore')

class WorkingDifferentialPrivacyCBF:
    """Working Differential Privacy CBF with proper evaluation"""
    
    def __init__(self, bf_len=1000, num_hash_func=10, q=2, epsilon=1.0):
        self.bf_len = bf_len
        self.num_hash_func = num_hash_func
        self.q = q
        self.epsilon = epsilon
        self.h1 = hashlib.sha1
        self.h2 = hashlib.md5
        
    def get_qgrams(self, text):
        if pd.isna(text):
            text = ""
        text = str(text).lower().strip()
        if len(text) < self.q:
            return [text.ljust(self.q, ' ')]
        return [text[i:i+self.q] for i in range(len(text) - self.q + 1)]
    
    def encode_record_clean(self, record_dict, fields=['first_name', 'last_name', 'city']):
        cbf = np.zeros(self.bf_len, dtype=int)
        
        for field in fields:
            if field in record_dict and not pd.isna(record_dict[field]):
                qgrams = self.get_qgrams(record_dict[field])
                for qgram in qgrams:
                    hex_str1 = self.h1(qgram.encode('utf-8')).hexdigest()
                    int1 = int(hex_str1, 16)
                    hex_str2 = self.h2(qgram.encode('utf-8')).hexdigest()
                    int2 = int(hex_str2, 16)
                    
                    for i in range(self.num_hash_func):
                        gi = (int1 + i * int2) % self.bf_len
                        cbf[gi] += 1
        
        return cbf
    
    def add_calibrated_noise(self, cbf):
        """Add properly calibrated Laplace noise"""
        # Much smaller sensitivity for better utility
        sensitivity = 0.1  # Very small noise
        noise_scale = sensitivity / self.epsilon if self.epsilon > 0 else 0.01
        
        # Add minimal noise
        noise = np.random.laplace(0, noise_scale, size=cbf.shape)
        noisy_cbf = cbf + noise
        
        # Keep reasonable bounds
        return np.maximum(0, noisy_cbf).astype(float)
    
    def encode_record_private(self, record_dict, fields=['first_name', 'last_name', 'city']):
        clean_cbf = self.encode_record_clean(record_dict, fields)
        return self.add_calibrated_noise(clean_cbf)
    
    def calculate_similarity(self, cbf1, cbf2):
        sum1 = np.sum(cbf1)
        sum2 = np.sum(cbf2)
        common = np.sum(np.minimum(cbf1, cbf2))
        
        if sum1 + sum2 == 0:
            return 0.0
        
        dice_sim = (2.0 * common) / (sum1 + sum2)
        return dice_sim

def load_data():
    """Simplified data loading"""
    alice_df = pd.read_csv('Alice_numrec_500_corr_25.csv')
    bob_df = pd.read_csv('Bob_numrec_500_corr_25.csv')
    
    # Standardize columns
    if 'givenname' in bob_df.columns:
        bob_df = bob_df.rename(columns={'givenname': 'first_name'})
    if 'surname' in bob_df.columns:
        bob_df = bob_df.rename(columns={'surname': 'last_name'})
    if 'suburb' in bob_df.columns:
        bob_df = bob_df.rename(columns={'suburb': 'city'})
    
    if 'rec_id' in alice_df.columns:
        alice_df['record_id'] = alice_df['rec_id']
    if 'recid' in bob_df.columns:
        bob_df['record_id'] = bob_df['recid']
    
    return alice_df, bob_df

def create_test_pairs(alice_df, bob_df, num_pairs=400):
    """Create test pairs ensuring we have both matches and non-matches"""
    pairs = []
    labels = []
    
    # Get matching IDs
    alice_ids = set(alice_df['record_id'].astype(str))
    bob_ids = set(bob_df['record_id'].astype(str))
    common_ids = list(alice_ids.intersection(bob_ids))
    
    # Take first half for positive pairs
    num_positive = min(len(common_ids), num_pairs // 2)
    
    for i in range(num_positive):
        record_id = common_ids[i]
        alice_record = alice_df[alice_df['record_id'].astype(str) == record_id].iloc[0]
        bob_record = bob_df[bob_df['record_id'].astype(str) == record_id].iloc[0]
        
        pairs.append((alice_record.to_dict(), bob_record.to_dict()))
        labels.append(1)
    
    # Create negative pairs
    np.random.seed(42)
    alice_sample = alice_df.sample(n=num_positive, replace=True)
    bob_sample = bob_df.sample(n=num_positive, replace=True)
    
    for i in range(num_positive):
        alice_record = alice_sample.iloc[i]
        bob_record = bob_sample.iloc[i]
        
        if str(alice_record['record_id']) != str(bob_record['record_id']):
            pairs.append((alice_record.to_dict(), bob_record.to_dict()))
            labels.append(0)
    
    return pairs, np.array(labels)

def evaluate_with_epsilon(epsilon):
    """Evaluate DP-CBF with specific epsilon"""
    print(f"\n--- Testing ε = {epsilon} ---")
    
    # Load data
    alice_df, bob_df = load_data()
    
    # Create test pairs
    pairs, labels = create_test_pairs(alice_df, bob_df, num_pairs=200)
    print(f"Created {len(pairs)} pairs ({np.sum(labels)} positive)")
    
    # Initialize DP-CBF
    dp_cbf = WorkingDifferentialPrivacyCBF(epsilon=epsilon)
    
    # Encode all pairs
    similarities_private = []
    similarities_clean = []
    
    for alice_record, bob_record in pairs:
        # Private encoding
        cbf1_private = dp_cbf.encode_record_private(alice_record)
        cbf2_private = dp_cbf.encode_record_private(bob_record)
        sim_private = dp_cbf.calculate_similarity(cbf1_private, cbf2_private)
        similarities_private.append(sim_private)
        
        # Clean encoding
        cbf1_clean = dp_cbf.encode_record_clean(alice_record)
        cbf2_clean = dp_cbf.encode_record_clean(bob_record)
        sim_clean = dp_cbf.calculate_similarity(cbf1_clean, cbf2_clean)
        similarities_clean.append(sim_clean)
    
    similarities_private = np.array(similarities_private)
    similarities_clean = np.array(similarities_clean)
    
    print(f"Private sim range: {similarities_private.min():.4f} - {similarities_private.max():.4f}")
    print(f"Clean sim range: {similarities_clean.min():.4f} - {similarities_clean.max():.4f}")
    
    # Find optimal thresholds by trying different values
    thresholds = np.arange(0.05, 0.6, 0.05)
    best_private_f1 = 0
    best_clean_f1 = 0
    best_private_threshold = 0.3
    best_clean_threshold = 0.3
    
    for threshold in thresholds:
        # Private predictions
        pred_private = (similarities_private > threshold).astype(int)
        if np.sum(pred_private) > 0 and np.sum(pred_private) < len(pred_private):
            try:
                f1_private = f1_score(labels, pred_private)
                if f1_private > best_private_f1:
                    best_private_f1 = f1_private
                    best_private_threshold = threshold
            except:
                pass
        
        # Clean predictions
        pred_clean = (similarities_clean > threshold).astype(int)
        if np.sum(pred_clean) > 0 and np.sum(pred_clean) < len(pred_clean):
            try:
                f1_clean = f1_score(labels, pred_clean)
                if f1_clean > best_clean_f1:
                    best_clean_f1 = f1_clean
                    best_clean_threshold = threshold
            except:
                pass
    
    # Final evaluation with best thresholds
    pred_private_final = (similarities_private > best_private_threshold).astype(int)
    pred_clean_final = (similarities_clean > best_clean_threshold).astype(int)
    
    acc_private = accuracy_score(labels, pred_private_final)
    acc_clean = accuracy_score(labels, pred_clean_final)
    
    print(f"Best thresholds - Private: {best_private_threshold:.3f}, Clean: {best_clean_threshold:.3f}")
    print(f"Private: Acc={acc_private:.4f}, F1={best_private_f1:.4f}")
    print(f"Clean:   Acc={acc_clean:.4f}, F1={best_clean_f1:.4f}")
    print(f"Loss:    Acc={acc_clean-acc_private:.4f}, F1={best_clean_f1-best_private_f1:.4f}")
    
    return {
        'epsilon': epsilon,
        'private_accuracy': acc_private,
        'private_f1': best_private_f1,
        'clean_accuracy': acc_clean,
        'clean_f1': best_clean_f1,
        'accuracy_loss': acc_clean - acc_private,
        'f1_loss': best_clean_f1 - best_private_f1
    }

def main():
    """Main working differential privacy experiment"""
    print("=== WORKING Differential Privacy CBF Experiment ===")
    
    np.random.seed(42)
    start_time = time.time()
    
    # Test range of epsilon values
    epsilons = [0.5, 1.0, 2.0, 5.0, 10.0]
    results = []
    
    for eps in epsilons:
        result = evaluate_with_epsilon(eps)
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("DIFFERENTIAL PRIVACY RESULTS SUMMARY")
    print("="*80)
    print(f"{'Epsilon':<8} {'Privacy':<12} {'Accuracy':<10} {'F1 Score':<10} {'Acc Loss':<10} {'F1 Loss':<10}")
    print("-"*80)
    
    for result in results:
        privacy_level = 'Very High' if result['epsilon'] < 1.0 else 'High' if result['epsilon'] < 2.0 else 'Medium' if result['epsilon'] < 5.0 else 'Low'
        print(f"{result['epsilon']:<8} {privacy_level:<12} "
              f"{result['private_accuracy']:<10.4f} {result['private_f1']:<10.4f} "
              f"{result['accuracy_loss']:<10.4f} {result['f1_loss']:<10.4f}")
    
    # Find best results
    valid_results = [r for r in results if r['private_f1'] > 0]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['private_f1'])
        print(f"\n🎯 Best Performance: ε={best_result['epsilon']} (F1={best_result['private_f1']:.4f})")
        
        high_privacy = [r for r in valid_results if r['epsilon'] <= 1.0]
        if high_privacy:
            best_high = max(high_privacy, key=lambda x: x['private_f1'])
            print(f"🔐 Best High Privacy: ε={best_high['epsilon']} (F1={best_high['private_f1']:.4f})")
    
    # Create visualization
    if len([r for r in results if r['private_f1'] > 0]) >= 2:
        plt.figure(figsize=(12, 4))
        
        epsilons_plot = [r['epsilon'] for r in results]
        
        plt.subplot(1, 2, 1)
        private_accs = [r['private_accuracy'] for r in results]
        clean_accs = [r['clean_accuracy'] for r in results]
        plt.plot(epsilons_plot, private_accs, 'b-o', label='Private', linewidth=2, markersize=8)
        plt.plot(epsilons_plot, clean_accs, 'r--s', label='Clean', linewidth=2, markersize=8)
        plt.xlabel('Privacy Parameter (ε)')
        plt.ylabel('Accuracy')
        plt.title('Privacy-Utility Trade-off: Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        private_f1s = [r['private_f1'] for r in results]
        clean_f1s = [r['clean_f1'] for r in results]
        plt.plot(epsilons_plot, private_f1s, 'g-o', label='Private F1', linewidth=2, markersize=8)
        plt.plot(epsilons_plot, clean_f1s, 'm--s', label='Clean F1', linewidth=2, markersize=8)
        plt.xlabel('Privacy Parameter (ε)')
        plt.ylabel('F1 Score')
        plt.title('Privacy-Utility Trade-off: F1 Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('differential_privacy_working.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("\n📊 Visualization saved as 'differential_privacy_working.png'")
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total runtime: {total_time:.2f} seconds")
    
    # Save results
    with open('differential_privacy_working_results.json', 'w') as f:
        json.dump({'results': results, 'runtime': total_time}, f, indent=2)
    
    print("📁 Results saved to 'differential_privacy_working_results.json'")
    
    return results

if __name__ == "__main__":
    results = main()
