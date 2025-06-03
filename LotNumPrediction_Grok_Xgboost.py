import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import random
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import itertools


class LotteryPredictor:
    def __init__(self, file_path="data.txt", num_range=(1, 50), nums_per_draw=5):
        self.file_path = file_path
        self.min_num = num_range[0]
        self.max_num = num_range[1]
        self.nums_per_draw = nums_per_draw
        self.data = []
        self.frequency = {}
        self.pairs = {}
        self.delta_patterns = {}
        self.last_actual = None
        self.last_prediction = None
        self.scaler = StandardScaler()
        self.models = {}
        self.performance_history = []

    def load_data(self):
        """Load the lottery data."""
        try:
            data = []
            with open(self.file_path, 'r') as file:
                for line in file:
                    numbers = [int(num) for num in line.strip().split('\t')]
                    if len(numbers) == self.nums_per_draw:
                        data.append(sorted(numbers))
            print(f"Loaded {len(data)} sets of lottery numbers")
            self.data = data
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return []

    def analyze_data(self):
        """Perform analysis on the lottery data."""
        self._analyze_frequency()
        self._analyze_combinations()
        self._analyze_deltas()
        self._analyze_hit_rate()
        return self

    def _analyze_frequency(self):
        """Analyze frequency with trend over last 5 draws."""
        all_numbers = [num for sublist in self.data for num in sublist]
        self.frequency = Counter(all_numbers)
        self.frequency_trend = {}
        if len(self.data) >= 5:
            recent_numbers = [num for draw in self.data[-5:] for num in draw]
            recent_freq = Counter(recent_numbers)
            total_draws = len(self.data)
            for num in range(self.min_num, self.max_num + 1):
                recent_count = recent_freq.get(num, 0) / 5
                long_count = self.frequency.get(num, 0) / total_draws
                self.frequency_trend[num] = recent_count - long_count
        return self.frequency

    def _analyze_combinations(self):
        """Analyze pairs for pattern detection."""
        self.pairs = defaultdict(int)
        for draw in self.data[-10:]:  # Focus on last 10 draws
            for pair in itertools.combinations(draw, 2):
                self.pairs[pair] += 1
        self.pair_significance = {pair: count for pair, count in self.pairs.items() if count > 1}
        return self.pairs

    def _analyze_deltas(self):
        """Enhanced delta analysis for clustering."""
        self.delta_patterns = defaultdict(int)
        self.cluster_score = defaultdict(int)
        for draw in self.data[-10:]:  # Last 10 draws
            deltas = [draw[i + 1] - draw[i] for i in range(len(draw) - 1)]
            self.delta_patterns[tuple(deltas)] += 1
            for i in range(len(draw) - 1):
                if draw[i + 1] - draw[i] <= 3:
                    self.cluster_score[draw[i]] += 1
                    self.cluster_score[draw[i + 1]] += 1
        self.common_deltas = sorted(self.delta_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
        return self.delta_patterns

    def _analyze_hit_rate(self):
        """Analyze hit rate."""
        if len(self.data) < 5:
            return []
        self.historical_hits = [len(set(self.data[i]).intersection(self.data[i + 1]))
                                for i in range(len(self.data) - 1)]
        self.avg_hit_count = np.mean(self.historical_hits) if self.historical_hits else 0
        return self.avg_hit_count

    def prepare_ml_features(self):
        """Feature engineering with clustering focus."""
        if len(self.data) < 10:
            print("Not enough data for ML prediction")
            return None, None

        window_size = 5
        X, y = [], []
        for i in range(len(self.data) - window_size):
            features = []
            for j in range(window_size):
                features.extend(self.data[i + j])
            for num in range(self.min_num, self.max_num + 1):
                features.append(sum(1 for draw in self.data[i:i + window_size] if num in draw))
            for j in range(window_size):
                draw = self.data[i + j]
                deltas = [draw[k + 1] - draw[k] for k in range(len(draw) - 1)]
                features.extend(deltas)
                features.append(sum(1 for d in deltas if d <= 3))  # Count tight clusters
            X.append(features)
            y.append(self.data[i + window_size])

        X = self.scaler.fit_transform(X)
        return np.array(X), np.array(y)

    def train_ml_models(self):
        """Train simplified ML models."""
        X, y = self.prepare_ml_features()
        if X is None or not len(X):
            return None

        self.models = {}
        for pos in range(self.nums_per_draw):
            y_pos = [draw[pos] for draw in y]
            xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

            models = {'xgb': xgb_model, 'rf': rf_model}
            for name, model in models.items():
                model.fit(X, y_pos)
                cv_score = np.mean(cross_val_score(model, X, y_pos, cv=3, scoring='neg_mean_squared_error'))
                print(f"Position {pos} - {name} CV MSE: {-cv_score:.2f}")
            self.models[pos] = models
        return self.models

    def _get_ml_prediction(self):
        """ML prediction with clustering awareness."""
        if not self.models or len(self.data) < 5:
            return None

        window_size = 5
        features = []
        for i in range(window_size):
            features.extend(self.data[-window_size + i])
        for num in range(self.min_num, self.max_num + 1):
            features.append(sum(1 for draw in self.data[-window_size:] if num in draw))
        for i in range(window_size):
            draw = self.data[-window_size + i]
            deltas = [draw[k + 1] - draw[k] for k in range(len(draw) - 1)]
            features.extend(deltas)
            features.append(sum(1 for d in deltas if d <= 3))

        features = self.scaler.transform([features])
        predictions = []
        for pos in range(self.nums_per_draw):
            pos_preds = [int(round(model.predict(features)[0])) for model in self.models[pos].values()]
            pred = max(self.min_num, min(self.max_num, int(np.median(pos_preds))))
            predictions.append(pred)

        prediction_set = set(predictions)
        while len(prediction_set) < self.nums_per_draw:
            high_freq = [num for num, _ in Counter(self.frequency).most_common()]
            for num in high_freq:
                if num not in prediction_set:
                    prediction_set.add(num)
                    break
        return sorted(list(prediction_set))

    def _get_statistical_prediction(self):
        """Statistical prediction with balanced range."""
        if not self.frequency or not self.frequency_trend:
            return None
        candidates = list(range(self.min_num, self.max_num + 1))
        weights = [max(0.1, 1 + (self.frequency_trend.get(num, 0) * 10) + self.cluster_score.get(num, 0) * 5)
                   for num in candidates]
        recent_nums = set(num for draw in self.data[-5:] for num in draw)
        for i, num in enumerate(candidates):
            if any(abs(num - n) <= 3 for n in recent_nums):
                weights[i] *= 2
            if num > 40:
                weights[i] *= 1.5
        prediction = set()
        while len(prediction) < self.nums_per_draw:
            choice = random.choices(candidates, weights=weights, k=1)[0]
            prediction.add(choice)
            weights[candidates.index(choice)] = 0.1
        return sorted(list(prediction))

    def _get_pattern_prediction(self):
        """Pattern prediction based on recent pairs."""
        if not self.pairs:
            return None
        number_scores = defaultdict(float)
        for pair, count in self.pair_significance.items():
            number_scores[pair[0]] += count * 3
            number_scores[pair[1]] += count * 3
        candidates = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
        prediction = [num for num, _ in candidates[:self.nums_per_draw]]
        while len(prediction) < self.nums_per_draw:
            available = [n for n in range(self.min_num, self.max_num + 1) if n not in prediction]
            if available:
                prediction.append(random.choice(available))
        return sorted(prediction)

    def _get_delta_prediction(self):
        """Delta prediction with tight mid-range cluster."""
        if not self.common_deltas or not self.data:
            return None
        recent_nums = set(num for draw in self.data[-5:] for num in draw)
        # Start with a strong mid-range cluster seed (15-25)
        mid_seeds = [n for n in recent_nums if 15 <= n <= 25 and n in self.cluster_score]
        start_num = random.choice(mid_seeds) if mid_seeds else random.choice(
            [n for n in recent_nums if 15 <= n <= 25] or list(recent_nums))
        prediction = [start_num]
        # Build a tight 3-number cluster
        for _ in range(2):  # 3 numbers total in cluster
            last_num = prediction[-1]
            cluster_options = [n for n in range(max(15, last_num - 3), min(26, last_num + 4))
                               if n not in prediction and (
                                           n in self.cluster_score or any(abs(n - r) <= 3 for r in recent_nums))]
            next_num = random.choice(cluster_options) if cluster_options else \
                random.choice([n for n in range(15, 26) if n not in prediction])
            prediction.append(next_num)
        # Add a low number (1-10)
        low_options = [n for n in range(1, 11) if n in recent_nums and n not in prediction]
        prediction.append(random.choice(low_options) if low_options else random.choice(
            [n for n in range(1, 11) if n not in prediction]))
        # Add a high number (40-50)
        high_options = [n for n in range(40, self.max_num + 1) if n in recent_nums and n not in prediction]
        prediction.append(random.choice(high_options) if high_options else random.choice(
            [n for n in range(40, 51) if n not in prediction]))
        return sorted(prediction)

    def predict_next_draw(self):
        """Prediction with tuned weights."""
        if not self.data:
            print("No data available. Please load data first.")
            return []

        if not self.models and len(self.data) >= 10:
            self.train_ml_models()

        weights = {'ml': 2, 'stat': 2, 'pattern': 3, 'delta': 5}  # Boost delta for clustering
        if len(self.data) < 10:
            weights['ml'] = 0
        if self.performance_history and np.mean([h for _, h in self.performance_history]) < 1:
            weights['delta'] += 2

        prediction_components = [
            (self._get_ml_prediction(), weights['ml']),
            (self._get_statistical_prediction(), weights['stat']),
            (self._get_pattern_prediction(), weights['pattern']),
            (self._get_delta_prediction(), weights['delta'])
        ]

        all_votes = defaultdict(float)
        for pred, weight in prediction_components:
            if pred:
                for num in pred:
                    all_votes[num] += weight * (0.5 if num in self.data[-1] else 1)  # Reduce last-draw repeats

        top_nums = [num for num, _ in sorted(all_votes.items(), key=lambda x: x[1], reverse=True)][
                   :self.nums_per_draw * 2]
        final_prediction = self._balance_prediction(top_nums)
        self.last_prediction = sorted(final_prediction)
        return sorted(final_prediction)

    def _balance_prediction(self, candidates):
        """Balance with tight mid-range cluster."""
        if len(candidates) <= self.nums_per_draw:
            return candidates[:self.nums_per_draw]
        prediction = []
        recent_nums = set(num for draw in self.data[-5:] for num in draw)

        # Build a 3-number mid-range cluster (15-25)
        mid_seeds = [n for n in candidates if
                     n in self.cluster_score and 15 <= n <= 25 and any(abs(n - r) <= 3 for r in recent_nums)]
        if mid_seeds:
            seed = random.choice(mid_seeds)
            prediction.append(seed)
            for _ in range(2):  # 3-number cluster
                nearby = [n for n in candidates if
                          abs(n - prediction[-1]) <= 3 and n not in prediction and 15 <= n <= 25]
                if nearby:
                    prediction.append(random.choice(nearby))
                elif len(prediction) < 3:
                    fallback = [n for n in range(15, 26) if
                                n not in prediction and any(abs(n - r) <= 5 for r in recent_nums)]
                    prediction.append(random.choice(fallback) if fallback else random.choice(
                        [n for n in range(15, 26) if n not in prediction]))

        # Add a low number (1-10)
        low_candidates = [n for n in candidates if 1 <= n <= 10 and n not in prediction]
        if low_candidates and len(prediction) < self.nums_per_draw:
            prediction.append(random.choice(low_candidates))
        elif len(prediction) < self.nums_per_draw:
            prediction.append(random.choice([n for n in range(1, 11) if n not in prediction]))

        # Add a high number (40-50)
        high_candidates = [n for n in candidates if n >= 40 and n not in prediction]
        if high_candidates and len(prediction) < self.nums_per_draw:
            prediction.append(random.choice(high_candidates))
        elif len(prediction) < self.nums_per_draw:
            prediction.append(random.choice([n for n in range(40, 51) if n not in prediction]))

        return sorted(prediction[:self.nums_per_draw])

    def evaluate_last_prediction(self, actual_result):
        """Evaluate prediction."""
        if not self.last_prediction:
            print("No previous prediction to evaluate")
            return 0
        self.last_actual = sorted(actual_result)
        matches = len(set(self.last_prediction).intersection(actual_result))
        match_percentage = (matches / self.nums_per_draw) * 100
        print(f"\nPrediction evaluation:")
        print(f"Predicted: {self.last_prediction}")
        print(f"Actual:    {sorted(actual_result)}")
        print(f"Matches:   {matches}/{self.nums_per_draw} ({match_percentage:.1f}%)")
        self.performance_history.append((self.last_prediction, matches))
        if sorted(actual_result) not in self.data:
            self.data.append(sorted(actual_result))
            self.analyze_data()
            if len(self.data) >= 10:
                self.train_ml_models()
        return matches

    def run(self, evaluate_with=None):
        """Run the predictor."""
        self.load_data()
        self.analyze_data()
        if len(self.data) >= 10:
            self.train_ml_models()
        prediction = self.predict_next_draw()

        print("\nData Analysis Results:")
        print(f"Total draws analyzed: {len(self.data)}")
        print(f"Average carry-over numbers: {self.avg_hit_count:.2f}")

        print("\nFINAL PREDICTION:")
        print(f"Predicted numbers: {prediction}")

        if evaluate_with:
            self.evaluate_last_prediction(evaluate_with)
        return prediction


def main(file_path="data.txt", actual_result=None):
    predictor = LotteryPredictor(file_path=file_path)
    if actual_result:
        if isinstance(actual_result, str):
            actual_result = [int(x) for x in actual_result.strip().split()]
        predictor.run(evaluate_with=actual_result)
    else:
        predictor.run()
    return predictor


if __name__ == "__main__":
    import sys

    file_path = "data.txt"
    actual_result = None
    if len(sys.argv) > 1:
        actual_result = [int(x) for x in sys.argv[1:]]
    main(file_path, actual_result)