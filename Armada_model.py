"""
ARMADA: Adversarially Robust Adaptive Malware Detection Architecture
======================================================================
Human Immune System-Inspired Malware Detection

Pipeline:
  Data Collection → Feature Extraction → Signature Filter →
  Behavioral Analysis → Risk Scoring → Adversarial Shield →
  Concept Drift Detection → Decision Engine
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
import joblib
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────────────────────
# 1. SYNTHETIC DATASET GENERATOR
# Simulates: File, Process, Network, Registry features
# ──────────────────────────────────────────────────────────────

def generate_dataset(n_samples=2000, random_state=42):
    """
    Generate synthetic malware/benign samples with realistic feature distributions.
    
    Feature Groups:
      File     : entropy, size, packed, imports, hash_anomaly
      Process  : cpu_spike, mem_usage, child_procs, injection, api_calls
      Network  : dns_requests, ip_rep_score, domain_age, exfil_volume, c2_pattern
      Registry : reg_modifications, startup_persist, service_create, key_deletion
    """
    np.random.seed(random_state)
    N = n_samples
    half = N // 2

    # ── Benign samples ──
    benign = {
        # File features
        'file_entropy':       np.random.normal(4.5, 0.8, half).clip(0, 8),
        'file_size_kb':       np.random.exponential(200, half).clip(1, 5000),
        'is_packed':          np.random.binomial(1, 0.05, half),
        'import_count':       np.random.randint(5, 50, half).astype(float),
        'hash_anomaly':       np.random.binomial(1, 0.02, half),

        # Process features
        'cpu_spike':          np.random.normal(15, 10, half).clip(0, 100),
        'mem_usage_mb':       np.random.normal(80, 40, half).clip(1, 500),
        'child_proc_count':   np.random.randint(0, 3, half).astype(float),
        'proc_injection':     np.random.binomial(1, 0.01, half),
        'api_call_freq':      np.random.normal(200, 80, half).clip(0, 1000),

        # Network features
        'dns_requests':       np.random.randint(0, 30, half).astype(float),
        'ip_reputation':      np.random.normal(85, 10, half).clip(0, 100),
        'domain_age_days':    np.random.exponential(500, half).clip(1, 5000),
        'exfil_volume_kb':    np.random.exponential(10, half).clip(0, 200),
        'c2_pattern_score':   np.random.normal(5, 5, half).clip(0, 100),

        # Registry / System
        'reg_modifications':  np.random.randint(0, 5, half).astype(float),
        'startup_persist':    np.random.binomial(1, 0.05, half),
        'service_created':    np.random.binomial(1, 0.02, half),
        'key_deletion':       np.random.binomial(1, 0.01, half),
    }

    # ── Malware samples ──
    malware = {
        # File features — high entropy (packed/encrypted), import anomaly
        'file_entropy':       np.random.normal(7.2, 0.5, half).clip(0, 8),
        'file_size_kb':       np.random.exponential(500, half).clip(1, 10000),
        'is_packed':          np.random.binomial(1, 0.80, half),
        'import_count':       np.random.randint(1, 200, half).astype(float),
        'hash_anomaly':       np.random.binomial(1, 0.85, half),

        # Process — CPU spikes, injection, many child procs
        'cpu_spike':          np.random.normal(75, 20, half).clip(0, 100),
        'mem_usage_mb':       np.random.normal(300, 100, half).clip(1, 1000),
        'child_proc_count':   np.random.randint(3, 20, half).astype(float),
        'proc_injection':     np.random.binomial(1, 0.75, half),
        'api_call_freq':      np.random.normal(700, 200, half).clip(0, 2000),

        # Network — many DNS requests, bad IPs, fresh domains, exfil
        'dns_requests':       np.random.randint(50, 500, half).astype(float),
        'ip_reputation':      np.random.normal(25, 15, half).clip(0, 100),
        'domain_age_days':    np.random.exponential(30, half).clip(1, 365),
        'exfil_volume_kb':    np.random.exponential(500, half).clip(0, 5000),
        'c2_pattern_score':   np.random.normal(75, 15, half).clip(0, 100),

        # Registry — persistence, service creation
        'reg_modifications':  np.random.randint(10, 100, half).astype(float),
        'startup_persist':    np.random.binomial(1, 0.90, half),
        'service_created':    np.random.binomial(1, 0.70, half),
        'key_deletion':       np.random.binomial(1, 0.60, half),
    }

    df_benign  = pd.DataFrame(benign);  df_benign['label']  = 0
    df_malware = pd.DataFrame(malware); df_malware['label'] = 1

    df = pd.concat([df_benign, df_malware]).reset_index(drop=True)
    df = shuffle(df, random_state=random_state).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────
# 2. ARMADA RISK SCORE CALCULATOR
# ──────────────────────────────────────────────────────────────

def compute_risk_score(row):
    """
    ARMADA Risk Formula:
      Risk = 0.25·S + 0.35·B + 0.25·N + 0.15·A

    Where:
      S = Signature Score  (file features)
      B = Behavioral Score (process features)
      N = Network Score    (network features)
      A = Adversarial Confidence
    """
    # Signature Score (S) — 0-100
    S = (
        (row['file_entropy'] / 8.0) * 40 +
        row['is_packed'] * 30 +
        row['hash_anomaly'] * 30
    )

    # Behavioral Score (B) — 0-100
    B = (
        (row['cpu_spike'] / 100.0) * 25 +
        (row['mem_usage_mb'] / 1000.0) * 15 +
        (row['child_proc_count'] / 20.0) * 20 +
        row['proc_injection'] * 25 +
        (row['api_call_freq'] / 2000.0) * 15
    )

    # Network Score (N) — 0-100
    N = (
        (row['dns_requests'] / 500.0) * 25 +
        ((100 - row['ip_reputation']) / 100.0) * 30 +
        ((365 - min(row['domain_age_days'], 365)) / 365.0) * 20 +
        (row['exfil_volume_kb'] / 5000.0) * 15 +
        (row['c2_pattern_score'] / 100.0) * 10
    )

    # Adversarial Confidence (A) — based on anomaly flags
    A = (
        row['hash_anomaly'] * 40 +
        row['proc_injection'] * 35 +
        row['startup_persist'] * 25
    )

    risk = 0.25 * S + 0.35 * B + 0.25 * N + 0.15 * A
    return round(min(risk, 100), 2), round(S, 2), round(B, 2), round(N, 2), round(A, 2)


def decision(risk_score):
    if risk_score < 30:
        return "SAFE"
    elif risk_score < 60:
        return "SUSPICIOUS"
    else:
        return "MALWARE"


# ──────────────────────────────────────────────────────────────
# 3. ARMADA ML ENGINE
# Hybrid: Random Forest (Phase 1) + Gradient Boosting (Phase 2)
# ──────────────────────────────────────────────────────────────

FEATURES = [
    'file_entropy', 'file_size_kb', 'is_packed', 'import_count', 'hash_anomaly',
    'cpu_spike', 'mem_usage_mb', 'child_proc_count', 'proc_injection', 'api_call_freq',
    'dns_requests', 'ip_reputation', 'domain_age_days', 'exfil_volume_kb', 'c2_pattern_score',
    'reg_modifications', 'startup_persist', 'service_created', 'key_deletion'
]


class ARMADADetector:
    """
    ARMADA Malware Detector
    Implements the full ARMADA pipeline with immune-system-inspired architecture.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        # Phase 1: Random Forest (innate immunity)
        self.rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=10,
            random_state=42, n_jobs=-1
        )
        # Phase 2: Gradient Boosting (adaptive immunity)
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1,
            max_depth=5, random_state=42
        )
        self.threat_memory = []   # Adaptive Immune Memory
        self.is_trained = False

    # ── Layer 1: Signature Filter ──────────────────────────────
    def signature_filter(self, features: dict) -> dict:
        """
        Innate immunity: fast rule-based pre-screening.
        Returns early MALWARE verdict for obvious cases.
        """
        flags = []
        if features.get('file_entropy', 0) > 7.5:
            flags.append("HIGH_ENTROPY_PACKING")
        if features.get('hash_anomaly', 0) == 1:
            flags.append("HASH_MISMATCH")
        if features.get('proc_injection', 0) == 1:
            flags.append("PROCESS_INJECTION")
        if features.get('ip_reputation', 100) < 20:
            flags.append("BLACKLISTED_IP")
        return {'flags': flags, 'blocked': len(flags) >= 3}

    # ── Layer 2: Behavioral Analysis (ML) ─────────────────────
    def train(self, df: pd.DataFrame):
        """Train both ML models on labeled data."""
        X = df[FEATURES].values
        y = df['label'].values
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Training Random Forest (Phase 1 - Innate Immunity)...")
        self.rf_model.fit(X_train, y_train)

        print("Training Gradient Boosting (Phase 2 - Adaptive Immunity)...")
        self.gb_model.fit(X_train, y_train)

        self.is_trained = True

        # Evaluate
        rf_preds  = self.rf_model.predict(X_test)
        gb_preds  = self.gb_model.predict(X_test)

        print("\n=== Phase 1: Random Forest ===")
        print(classification_report(y_test, rf_preds, target_names=['Benign', 'Malware']))

        print("=== Phase 2: Gradient Boosting ===")
        print(classification_report(y_test, gb_preds, target_names=['Benign', 'Malware']))

        return X_test, y_test

    def predict(self, features: dict) -> dict:
        """
        Full ARMADA pipeline prediction for a single sample.
        Returns complete analysis with risk score and decision.
        """
        # Step 1: Signature Filter
        sig = self.signature_filter(features)
        if sig['blocked']:
            return {
                'verdict': 'MALWARE',
                'risk_score': 95.0,
                'method': 'SIGNATURE_FILTER',
                'flags': sig['flags'],
                'confidence': 0.99,
                'memory_match': False,
                'components': {}
            }

        # Step 2: Risk Score calculation
        row = pd.Series(features)
        risk, S, B, N, A = compute_risk_score(row)

        # Step 3: ML prediction (if trained)
        ml_confidence = 0.5
        if self.is_trained:
            X = np.array([[features.get(f, 0) for f in FEATURES]])
            X_scaled = self.scaler.transform(X)
            rf_prob  = self.rf_model.predict_proba(X_scaled)[0][1]
            gb_prob  = self.gb_model.predict_proba(X_scaled)[0][1]
            ml_confidence = 0.5 * rf_prob + 0.5 * gb_prob
            # Blend risk score with ML confidence
            risk = 0.6 * risk + 0.4 * (ml_confidence * 100)

        # Step 4: Adversarial check
        adv_boost = 0
        if features.get('is_packed') and features.get('proc_injection'):
            adv_boost = 10  # Likely evasion attempt

        final_risk = min(round(risk + adv_boost, 2), 100)

        # Step 5: Memory check (concept drift awareness)
        memory_match = self._check_threat_memory(features)

        verdict = decision(final_risk)

        # Step 6: Store in threat memory if malware
        if verdict == 'MALWARE':
            self._store_threat_memory(features, final_risk)

        return {
            'verdict': verdict,
            'risk_score': final_risk,
            'method': 'FULL_PIPELINE',
            'flags': sig['flags'],
            'confidence': round(ml_confidence, 3),
            'memory_match': memory_match,
            'components': {
                'signature_score': S,
                'behavioral_score': B,
                'network_score': N,
                'adversarial_score': A,
            }
        }

    # ── Adaptive Immune Memory ─────────────────────────────────
    def _store_threat_memory(self, features: dict, risk: float):
        """Store behavioral fingerprint (memory cell)."""
        fingerprint = {
            'entropy':       features.get('file_entropy', 0),
            'proc_inject':   features.get('proc_injection', 0),
            'c2_score':      features.get('c2_pattern_score', 0),
            'persist':       features.get('startup_persist', 0),
            'risk':          risk,
        }
        self.threat_memory.append(fingerprint)

    def _check_threat_memory(self, features: dict) -> bool:
        """Check if pattern matches known threat in memory."""
        for mem in self.threat_memory:
            if (abs(mem['entropy'] - features.get('file_entropy', 0)) < 0.5 and
                    mem['proc_inject'] == features.get('proc_injection', 0) and
                    abs(mem['c2_score'] - features.get('c2_pattern_score', 0)) < 10):
                return True
        return False

    def save(self, path='armada_model.pkl'):
        joblib.dump({'rf': self.rf_model, 'gb': self.gb_model,
                     'scaler': self.scaler, 'memory': self.threat_memory}, path)
        print(f"Model saved to {path}")

    def load(self, path='armada_model.pkl'):
        data = joblib.load(path)
        self.rf_model     = data['rf']
        self.gb_model     = data['gb']
        self.scaler       = data['scaler']
        self.threat_memory = data['memory']
        self.is_trained   = True
        print(f"Model loaded from {path}")


# ──────────────────────────────────────────────────────────────
# 4. MAIN: Train, Evaluate & Demo
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  ARMADA: Adversarially Robust Adaptive Malware Detector")
    print("=" * 60)

    # Generate dataset
    print("\n[1] Generating synthetic dataset (2000 samples)...")
    df = generate_dataset(n_samples=2000)
    print(f"    Benign: {(df.label==0).sum()} | Malware: {(df.label==1).sum()}")

    # Train model
    print("\n[2] Training ARMADA models...")
    armada = ARMADADetector()
    armada.train(df)

    # Save model
    armada.save('../models/armada_model.pkl')

    # Demo predictions
    print("\n[3] Demo: Running ARMADA on sample inputs")
    print("-" * 60)

    samples = {
        "Ransomware (WannaCry-like)": {
            'file_entropy': 7.8, 'file_size_kb': 3200, 'is_packed': 1,
            'import_count': 15, 'hash_anomaly': 1,
            'cpu_spike': 90, 'mem_usage_mb': 450, 'child_proc_count': 8,
            'proc_injection': 1, 'api_call_freq': 1200,
            'dns_requests': 300, 'ip_reputation': 10, 'domain_age_days': 5,
            'exfil_volume_kb': 2000, 'c2_pattern_score': 85,
            'reg_modifications': 50, 'startup_persist': 1,
            'service_created': 1, 'key_deletion': 1,
        },
        "Normal Application (Chrome)": {
            'file_entropy': 4.2, 'file_size_kb': 250, 'is_packed': 0,
            'import_count': 35, 'hash_anomaly': 0,
            'cpu_spike': 12, 'mem_usage_mb': 120, 'child_proc_count': 1,
            'proc_injection': 0, 'api_call_freq': 180,
            'dns_requests': 10, 'ip_reputation': 92, 'domain_age_days': 3000,
            'exfil_volume_kb': 5, 'c2_pattern_score': 2,
            'reg_modifications': 2, 'startup_persist': 0,
            'service_created': 0, 'key_deletion': 0,
        },
        "Suspicious Script (Polymorphic)": {
            'file_entropy': 6.8, 'file_size_kb': 800, 'is_packed': 1,
            'import_count': 5, 'hash_anomaly': 1,
            'cpu_spike': 45, 'mem_usage_mb': 200, 'child_proc_count': 3,
            'proc_injection': 0, 'api_call_freq': 500,
            'dns_requests': 60, 'ip_reputation': 45, 'domain_age_days': 30,
            'exfil_volume_kb': 100, 'c2_pattern_score': 40,
            'reg_modifications': 10, 'startup_persist': 1,
            'service_created': 0, 'key_deletion': 0,
        },
    }

    for name, features in samples.items():
        result = armada.predict(features)
        print(f"\nSample: {name}")
        print(f"  Verdict     : {result['verdict']}")
        print(f"  Risk Score  : {result['risk_score']} / 100")
        print(f"  ML Confidence: {result['confidence']:.1%}")
        if result['flags']:
            print(f"  Flags       : {', '.join(result['flags'])}")
        c = result['components']
        if c:
            print(f"  Scores      : S={c['signature_score']} B={c['behavioral_score']} "
                  f"N={c['network_score']} A={c['adversarial_score']}")
        print(f"  Memory Match: {result['memory_match']}")

    print("\n" + "=" * 60)
    print("ARMADA pipeline complete.")