"""
ECG Arrhythmia Classification Pipeline
Implementation of Zheng et al. (2020) multi-stage approach with novel analysis

References:
    Zheng, J., Zhang, J., Danioko, S., Yao, H., Guo, H., & Rakovski, C. (2020).
    A 12-lead electrocardiogram database for arrhythmia research covering more
    than 10,000 patients. Scientific Data, 7(1), 48.
    
    Zheng, J., Chu, H., Struppa, D., et al. (2020).
    Optimal multi-stage arrhythmia classification approach.
    Scientific Reports, 10, 2898.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class ECGDataLoader:
    """Load and validate 12-lead ECG data"""
    
    def __init__(self, sampling_rate=500):
        self.sampling_rate = sampling_rate
        self.duration = 10
        self.n_samples = sampling_rate * self.duration
        self.n_leads = 12
        
    def load_ecg(self, data_path):
        """Load ECG data from CSV/numpy file"""
        try:
            if data_path.endswith('.npy'):
                data = np.load(data_path)
            else:
                data = np.genfromtxt(data_path, delimiter=',')
            
            if data.ndim == 2:
                data = data[np.newaxis, :, :]
            return data
        except Exception as e:
            print(f"Error loading ECG data: {e}")
            return None
    
    def validate_ecg(self, ecg_signal):
        """Check for missing values and signal integrity"""
        return not (np.any(np.isnan(ecg_signal)) or np.any(np.isinf(ecg_signal)))


class NoiseReduction:
    """Three-stage noise reduction pipeline (Zheng et al. 2020)"""
    
    def __init__(self, sampling_rate=500):
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2
    
    def butterworth_lowpass(self, signal_data, cutoff=100, order=4):
        """Stage 1: Butterworth Low-Pass Filter (removes >100Hz noise)"""
        normalized_cutoff = cutoff / self.nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        
        filtered = np.zeros_like(signal_data)
        for lead in range(signal_data.shape[-1]):
            filtered[..., lead] = signal.filtfilt(b, a, signal_data[..., lead], axis=-1)
        return filtered
    
    def robust_loess(self, signal_data, frac=0.15):
        """Stage 2: Robust LOESS (removes baseline wandering)"""
        from scipy.signal import savgol_filter
        
        window_length = max(11, int(2 * len(signal_data[0]) * frac))
        if window_length % 2 == 0:
            window_length += 1
        
        smoothed = np.zeros_like(signal_data)
        for lead in range(signal_data.shape[-1]):
            smoothed[..., lead] = savgol_filter(
                signal_data[..., lead], window_length, polyorder=3, axis=-1
            )
        return signal_data - smoothed
    
    def nonlocal_means(self, signal_data, h=0.1, patch_size=5):
        """Stage 3: Non-Local Means Denoising"""
        denoised = np.zeros_like(signal_data, dtype=float)
        
        for lead in range(signal_data.shape[-1]):
            lead_signal = signal_data[..., lead]
            
            for i in range(lead_signal.shape[-1]):
                start_idx = max(0, i - patch_size)
                end_idx = min(lead_signal.shape[-1], i + patch_size + 1)
                patch_i = lead_signal[..., start_idx:end_idx]
                
                weights = np.zeros(lead_signal.shape[-1])
                for j in range(lead_signal.shape[-1]):
                    start_j = max(0, j - patch_size)
                    end_j = min(lead_signal.shape[-1], j + patch_size + 1)
                    patch_j = lead_signal[..., start_j:end_j]
                    
                    dist = np.sqrt(np.sum((patch_i - patch_j) ** 2))
                    weights[j] = np.exp(-(dist ** 2) / (h ** 2))
                
                denoised[..., lead, i] = np.sum(
                    weights * lead_signal[..., i]
                ) / np.sum(weights)
        
        return denoised
    
    def apply_pipeline(self, signal_data):
        """Apply complete 3-stage noise reduction"""
        stage1 = self.butterworth_lowpass(signal_data)
        stage2 = self.robust_loess(stage1)
        stage3 = self.nonlocal_means(stage2)
        return stage3


class ECGFeatureExtractor:
    """Extract ~150 features from preprocessed ECG signals"""
    
    def __init__(self, sampling_rate=500):
        self.sampling_rate = sampling_rate
    
    def detect_peaks_and_valleys(self, signal_1d):
        """Detect QRS complexes and peaks/valleys"""
        peaks, _ = signal.find_peaks(signal_1d, distance=50)
        valleys, _ = signal.find_peaks(-signal_1d, distance=50)
        return peaks, valleys
    
    def extract_wave_measurements(self, ecg_signal):
        """Extract ECG wave measurements"""
        features_dict = {}
        
        for lead_idx in range(ecg_signal.shape[-1]):
            lead_signal = ecg_signal[..., lead_idx]
            peaks, valleys = self.detect_peaks_and_valleys(lead_signal)
            
            features_dict[f'lead_{lead_idx}_n_peaks'] = len(peaks)
            features_dict[f'lead_{lead_idx}_n_valleys'] = len(valleys)
            
            if len(peaks) > 0:
                features_dict[f'lead_{lead_idx}_peak_mean'] = np.mean(lead_signal[peaks])
                features_dict[f'lead_{lead_idx}_peak_std'] = np.std(lead_signal[peaks])
                features_dict[f'lead_{lead_idx}_peak_max'] = np.max(lead_signal[peaks])
            
            if len(valleys) > 0:
                features_dict[f'lead_{lead_idx}_valley_mean'] = np.mean(lead_signal[valleys])
                features_dict[f'lead_{lead_idx}_valley_std'] = np.std(lead_signal[valleys])
        
        return features_dict
    
    def extract_interval_features(self, ecg_signal):
        """Extract interval and ratio features"""
        features_dict = {}
        
        for lead_idx in range(ecg_signal.shape[-1]):
            lead_signal = ecg_signal[..., lead_idx]
            peaks, valleys = self.detect_peaks_and_valleys(lead_signal)
            
            if len(peaks) > 1:
                peak_intervals = np.diff(peaks)
                features_dict[f'lead_{lead_idx}_peak_interval_mean'] = np.mean(peak_intervals)
                features_dict[f'lead_{lead_idx}_peak_interval_std'] = np.std(peak_intervals)
                features_dict[f'lead_{lead_idx}_peak_interval_cv'] = (
                    np.std(peak_intervals) / (np.mean(peak_intervals) + 1e-6)
                )
            
            if len(peaks) > 0 and len(valleys) > 0:
                avg_peak = np.mean(np.abs(lead_signal[peaks]))
                avg_valley = np.mean(np.abs(lead_signal[valleys]))
                features_dict[f'lead_{lead_idx}_peak_valley_ratio'] = (
                    avg_peak / (avg_valley + 1e-6)
                )
        
        return features_dict
    
    def extract_statistical_features(self, ecg_signal):
        """Extract statistical and spectral features"""
        features_dict = {}
        
        for lead_idx in range(ecg_signal.shape[-1]):
            lead_signal = ecg_signal[..., lead_idx]
            
            features_dict[f'lead_{lead_idx}_mean'] = np.mean(lead_signal)
            features_dict[f'lead_{lead_idx}_std'] = np.std(lead_signal)
            features_dict[f'lead_{lead_idx}_min'] = np.min(lead_signal)
            features_dict[f'lead_{lead_idx}_max'] = np.max(lead_signal)
            features_dict[f'lead_{lead_idx}_range'] = np.ptp(lead_signal)
            
            fft_vals = np.abs(fft(lead_signal))
            features_dict[f'lead_{lead_idx}_spectral_energy'] = np.sum(fft_vals ** 2)
            features_dict[f'lead_{lead_idx}_spectral_entropy'] = (
                -np.sum((fft_vals / np.sum(fft_vals)) * 
                np.log(fft_vals / np.sum(fft_vals) + 1e-10))
            )
            features_dict[f'lead_{lead_idx}_entropy'] = self._compute_entropy(lead_signal)
        
        return features_dict
    
    def _compute_entropy(self, signal_1d, bins=10):
        """Compute Shannon entropy"""
        hist, _ = np.histogram(signal_1d, bins=bins)
        hist = hist / np.sum(hist)
        return -np.sum(hist * np.log(hist + 1e-10))
    
    def extract_all_features(self, ecg_signal):
        """Extract all feature groups"""
        features = {}
        features.update(self.extract_wave_measurements(ecg_signal))
        features.update(self.extract_interval_features(ecg_signal))
        features.update(self.extract_statistical_features(ecg_signal))
        return features


class ArrhythmiaClassifier:
    """XGBoost classifier for 4-class arrhythmia classification"""
    
    CLASS_MAPPING = {
        'SB': 0,      # Sinus Bradycardia
        'SR': 1,      # Sinus Rhythm
        'AFIB': 2,    # Atrial Fibrillation
        'GSVT': 3     # Supraventricular Tachycardia
    }
    
    def __init__(self, use_rescaling=True):
        self.model = None
        self.scaler = StandardScaler()
        self.use_rescaling = use_rescaling
        self.is_fitted = False
    
    def create_model(self, max_depth=6, learning_rate=0.1, n_estimators=100):
        """Create XGBoost model"""
        self.model = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            objective='multi:softmax',
            num_class=4,
            random_state=42,
            eval_metric='mlogloss',
            verbosity=0
        )
        return self.model
    
    def fit(self, X, y):
        """Train the classifier"""
        if self.use_rescaling:
            X = self._rescale_signals(X)
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.use_rescaling:
            X = self._rescale_signals(X)
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.use_rescaling:
            X = self._rescale_signals(X)
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def _rescale_signals(self, X):
        """Rescale to maximum peak value of 1"""
        X_rescaled = X.copy()
        for i in range(X.shape[0]):
            max_val = np.max(np.abs(X[i]))
            if max_val > 0:
                X_rescaled[i] = X[i] / max_val
        return X_rescaled
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        y_pred = self.predict(X)
        
        results = {
            'f1_weighted': f1_score(y, y_pred, average='weighted'),
            'f1_macro': f1_score(y, y_pred, average='macro'),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classification_report': classification_report(y, y_pred, zero_division=0)
        }
        
        return results, y_pred


class ECGArrhythmiaPipeline:
    """Complete ECG arrhythmia classification pipeline"""
    
    def __init__(self, sampling_rate=500):
        self.sampling_rate = sampling_rate
        self.data_loader = ECGDataLoader(sampling_rate)
        self.noise_reducer = NoiseReduction(sampling_rate)
        self.feature_extractor = ECGFeatureExtractor(sampling_rate)
        self.classifier = ArrhythmiaClassifier()
    
    def preprocess(self, ecg_data):
        """Apply noise reduction pipeline"""
        if ecg_data.ndim == 2:
            ecg_data = ecg_data[np.newaxis, :, :]
        
        n_records = ecg_data.shape[0]
        processed = np.zeros_like(ecg_data)
        
        for i in range(n_records):
            processed[i] = self.noise_reducer.apply_pipeline(ecg_data[i:i+1])[0]
        
        return processed
    
    def extract_features(self, ecg_data):
        """Extract features from preprocessed ECG"""
        if ecg_data.ndim == 2:
            ecg_data = ecg_data[np.newaxis, :, :]
        
        features_list = []
        for i in range(ecg_data.shape[0]):
            features_dict = self.feature_extractor.extract_all_features(ecg_data[i])
            features_list.append(features_dict)
        
        return pd.DataFrame(features_list)
    
    def train_classifier(self, X, y):
        """Train classification model"""
        self.classifier.create_model()
        self.classifier.fit(X.values, y)
        
        cv_scores = cross_val_score(
            self.classifier.model,
            self.classifier.scaler.transform(X.values),
            y,
            cv=10,
            scoring='f1_weighted'
        )
        
        print(f"10-fold CV F1-Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        return self.classifier
    
    def predict(self, X):
        """Make predictions"""
        return self.classifier.predict(X.values)
    
    def evaluate(self, X, y):
        """Evaluate classifier"""
        results, y_pred = self.classifier.evaluate(X.values, y)
        
        print(f"\nModel Performance:")
        print(f"Weighted F1-Score: {results['f1_weighted']:.4f}")
        print(f"Macro F1-Score: {results['f1_macro']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        
        return results, y_pred
