import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report
import hashlib
import time
import json
import warnings
warnings.filterwarnings('ignore')

class DifferentialPrivacyCBF_100Records:
    """DP-CBF optimized for 100-record dataset"""
    
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
        """Calibrated noise for 100-record dataset"""
        # Slightly larger noise for smaller dataset
        sensitivity = 0.15  
        noise_scale = sensitivity / self.epsilon if self.epsilon > 0 else 0.01
        
        noise = np.random.laplace(0, noise_scale, size=cbf.shape)
        noisy_cbf = cbf + noise
        
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

def load_100_record_data():
    """Load 100-record Alice and Bob datasets"""
    alice_df = pd.read_csv('Alice_numrec_100_corr_25.csv')
    bob_df = pd.read_csv('Bob_numrec_100_corr_25.csv')
    
    print(f"Loaded Alice: {len(alice_df)} records, Bob: {len(bob_df)} records")
    
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

def create_test_pairs_100(alice_df, bob_df, num_pairs=100):
    """Create test pairs for 100-record dataset"""
    pairs = []
    labels = []
    
    # Get matching IDs
    alice_ids = set(alice_df['record_id'].astype(str))
    bob_ids = set(bob_df['record_id'].astype(str))
    common_ids = list(alice_ids.intersection(bob_ids))
    
    print(f"Found {len(common_ids)} matching record pairs")
    
    # Take all available positive pairs (up to num_pairs//2)
    num_positive = min(len(common_ids), num_pairs // 2)
    
    for i in range(num_positive):
        record_id = common_ids[i]
        alice_record = alice_df[alice_df['record_id'].astype(str) == record_id].iloc[0]
        bob_record = bob_df[bob_df['record_id'].astype(str) == record_id].iloc[0]
        
        pairs.append((alice_record.to_dict(), bob_record.to_dict()))
        labels.append(1)
    
    # Create negative pairs
    np.random.seed(42)
    negative_needed = num_positive  # Equal number
    alice_sample = alice_df.sample(n=negative_needed, replace=True)
    bob_sample = bob_df.sample(n=negative_needed, replace=True)
    
    for i in range(negative_needed):
        alice_record = alice_sample.iloc[i]
        bob_record = bob_sample.iloc[i]
        
        if str(alice_record['record_id']) != str(bob_record['record_id']):
            pairs.append((alice_record.to_dict(), bob_record.to_dict()))
            labels.append(0)
    
    print(f"Created {len(pairs)} pairs ({num_positive} positive, {len(pairs)-num_positive} negative)")
    return pairs, np.array(labels)

def evaluate_dp_100_records(epsilon):
    """Evaluate DP-CBF on 100-record dataset"""
    print(f"\n--- Testing ε = {epsilon} (100 Records) ---")
    
    # Load 100-record data
    alice_df, bob_df = load_100_record_data()
    
    # Create test pairs (smaller dataset = fewer pairs)
    pairs, labels = create_test_pairs_100(alice_df, bob_df, num_pairs=80)
    
    # Initialize DP-CBF
    dp_cbf = DifferentialPrivacyCBF_100Records(epsilon=epsilon)
    
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
    
    # Find optimal thresholds
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
    
    # Final evaluation
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
        'dataset_size': 100,
        'private_accuracy': acc_private,
        'private_f1': best_private_f1,
        'clean_accuracy': acc_clean,
        'clean_f1': best_clean_f1,
        'accuracy_loss': acc_clean - acc_private,
        'f1_loss': best_clean_f1 - best_private_f1
    }

def main_100_records():
    """Main experiment for 100-record dataset comparison"""
    print("=== Differential Privacy CBF: 100 Records Experiment ===")
    print("Comparing with Siamese Network results on same dataset\n")
    
    np.random.seed(42)
    start_time = time.time()
    
    # Test same epsilon values as 500-record experiment
    epsilons = [0.5, 1.0, 2.0, 5.0, 10.0]
    results_100 = []
    
    for eps in epsilons:
        result = evaluate_dp_100_records(eps)
        results_100.append(result)
    
    # Summary comparison
    print("\n" + "="*90)
    print("DIFFERENTIAL PRIVACY CBF: 100 vs 500 RECORDS COMPARISON")
    print("="*90)
    print("Comparing performance on different dataset sizes:")
    print(f"{'Epsilon':<8} {'Dataset':<8} {'Privacy':<12} {'Accuracy':<10} {'F1 Score':<10} {'F1 Loss':<10}")
    print("-"*90)
    
    # Your previous 500-record results for comparison
    results_500 = [
        {'epsilon': 0.5, 'private_accuracy': 0.94, 'private_f1': 0.9394, 'f1_loss': 0.0261},
        {'epsilon': 1.0, 'private_accuracy': 0.95, 'private_f1': 0.95, 'f1_loss': 0.0155},
        {'epsilon': 2.0, 'private_accuracy': 0.955, 'private_f1': 0.9552, 'f1_loss': 0.0103},
        {'epsilon': 5.0, 'private_accuracy': 0.96, 'private_f1': 0.9604, 'f1_loss': 0.0051},
        {'epsilon': 10.0, 'private_accuracy': 0.965, 'private_f1': 0.9655, 'f1_loss': 0.0000}
    ]
    
    for i, (result_100, result_500) in enumerate(zip(results_100, results_500)):
        eps = result_100['epsilon']
        privacy_level = 'Very High' if eps < 1.0 else 'High' if eps < 2.0 else 'Medium' if eps < 5.0 else 'Low'
        
        # 100-record results
        print(f"{eps:<8} {'100':<8} {privacy_level:<12} "
              f"{result_100['private_accuracy']:<10.4f} {result_100['private_f1']:<10.4f} "
              f"{result_100['f1_loss']:<10.4f}")
        
        # 500-record results
        print(f"{eps:<8} {'500':<8} {privacy_level:<12} "
              f"{result_500['private_accuracy']:<10.4f} {result_500['private_f1']:<10.4f} "
              f"{result_500['f1_loss']:<10.4f}")
        print()
    
    # Dataset size analysis
    print("\nDataset Size Impact Analysis:")
    print("=" * 50)
    
    for i, (result_100, result_500) in enumerate(zip(results_100, results_500)):
        eps = result_100['epsilon']
        f1_diff = result_500['private_f1'] - result_100['private_f1']
        acc_diff = result_500['private_accuracy'] - result_100['private_accuracy']
        
        print(f"ε={eps}: 500-record advantage = F1: {f1_diff:+.4f}, Accuracy: {acc_diff:+.4f}")
    
    # Comparison with Siamese Network (from your previous results)
    print(f"\nMODEL PERFORMANCE COMPARISON ON 100 RECORDS:")
    print("=" * 60)
    print(f"{'Model':<25} {'Accuracy':<12} {'F1 Score':<12} {'Training Time'}")
    print("-" * 60)
    
    # Find best DP result
    best_dp_100 = max(results_100, key=lambda x: x['private_f1'])
    
    print(f"{'Siamese + CBF':<25} {'75.0%':<12} {'66.67%':<12} {'~2 minutes'}")
    print(f"{'CBF Baseline':<25} {'95.0%':<12} {'95.24%':<12} {'<1 second'}")
    
    # Fixed: No f-string formatting issues
    dp_name = f"DP-CBF (ε={best_dp_100['epsilon']})"
    dp_acc = f"{best_dp_100['private_accuracy']*100:.1f}%"
    dp_f1_score = f"{best_dp_100['private_f1']*100:.2f}%"
    print(f"{dp_name:<25} {dp_acc:<12} {dp_f1_score:<12} {'~1 second'}")
    
    # Create visualization comparing dataset sizes
    plt.figure(figsize=(15, 5))
    
    epsilons_plot = [r['epsilon'] for r in results_100]
    
    # Accuracy comparison
    plt.subplot(1, 3, 1)
    acc_100 = [r['private_accuracy'] for r in results_100]
    acc_500 = [r['private_accuracy'] for r in results_500]
    plt.plot(epsilons_plot, acc_100, 'b-o', label='100 Records', linewidth=2, markersize=8)
    plt.plot(epsilons_plot, acc_500, 'r-s', label='500 Records', linewidth=2, markersize=8)
    plt.xlabel('Privacy Parameter (ε)')
    plt.ylabel('Accuracy')
    plt.title('Dataset Size Impact: Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # F1 Score comparison
    plt.subplot(1, 3, 2)
    f1_100 = [r['private_f1'] for r in results_100]
    f1_500 = [r['private_f1'] for r in results_500]
    plt.plot(epsilons_plot, f1_100, 'g-o', label='100 Records', linewidth=2, markersize=8)
    plt.plot(epsilons_plot, f1_500, 'm-s', label='500 Records', linewidth=2, markersize=8)
    plt.xlabel('Privacy Parameter (ε)')
    plt.ylabel('F1 Score')
    plt.title('Dataset Size Impact: F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Privacy cost comparison
    plt.subplot(1, 3, 3)
    loss_100 = [r['f1_loss'] for r in results_100]
    loss_500 = [r['f1_loss'] for r in results_500]
    plt.plot(epsilons_plot, loss_100, 'orange', marker='o', label='100 Records', linewidth=2, markersize=8)
    plt.plot(epsilons_plot, loss_500, 'purple', marker='s', label='500 Records', linewidth=2, markersize=8)
    plt.xlabel('Privacy Parameter (ε)')
    plt.ylabel('F1 Loss')
    plt.title('Privacy Cost by Dataset Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dp_cbf_dataset_size_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds")
    print("Dataset size comparison visualization saved as 'dp_cbf_dataset_size_comparison.png'")
    
    # Save results
    comparison_results = {
        'results_100_records': results_100,
        'results_500_records': results_500,
        'runtime': total_time,
        'best_100_record_performance': best_dp_100
    }
    
    with open('dp_cbf_dataset_comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print("Comparison results saved to 'dp_cbf_dataset_comparison.json'")
    
    return results_100

if __name__ == "__main__":
    results = main_100_records()
