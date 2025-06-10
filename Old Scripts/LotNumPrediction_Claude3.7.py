import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import random
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
import itertools
import math


class LotteryPredictor:
    def __init__(self, file_path="data.txt", num_range=(1, 50), nums_per_draw=5):
        self.file_path = file_path
        self.min_num = num_range[0]
        self.max_num = num_range[1]
        self.nums_per_draw = nums_per_draw
        self.data = []
        self.frequency = {}
        self.pairs = {}
        self.triplets = {}
        self.historical_hits = []
        self.patterns = {}
        self.delta_patterns = {}
        self.last_actual = None  # Store the last actual result
        self.last_prediction = None  # Store the last prediction
        self.scaler = StandardScaler()

    def load_data(self):
        """Load the lottery data from a tab-separated file."""
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
        """Perform comprehensive analysis on the lottery data."""
        self._analyze_frequency()
        self._analyze_combinations()
        self._analyze_deltas()
        self._analyze_hot_cold()
        self._analyze_hit_rate()
        self._create_advanced_features()
        return self

    def _analyze_frequency(self):
        """Analyze the frequency of each number in the historical data."""
        all_numbers = [num for sublist in self.data for num in sublist]
        self.frequency = Counter(all_numbers)

        total_draws = len(self.data)
        # Calculate expected frequency if distribution was uniform
        expected_freq = total_draws * self.nums_per_draw / (self.max_num - self.min_num + 1)

        # Calculate deviation from expected frequency
        self.frequency_deviation = {
            num: (count / expected_freq) - 1
            for num, count in self.frequency.items()
        }

        # Calculate how frequency changes over time (trend)
        self.frequency_trend = {}
        if len(self.data) >= 10:
            recent_data = self.data[-10:]
            old_data = self.data[:-10]

            recent_nums = [num for sublist in recent_data for num in sublist]
            old_nums = [num for sublist in old_data for num in sublist]

            recent_freq = Counter(recent_nums)
            old_freq = Counter(old_nums)

            for num in range(self.min_num, self.max_num + 1):
                recent_count = recent_freq.get(num, 0) / len(recent_data)
                old_count = old_freq.get(num, 0) / len(old_data) if old_data else 0
                self.frequency_trend[num] = recent_count - old_count

        return self.frequency

    def _analyze_combinations(self):
        """Analyze combinations of numbers that appear together."""
        # Analyze pairs and triplets
        self.pairs = defaultdict(int)
        self.triplets = defaultdict(int)

        for draw in self.data:
            # Look at pairs
            for pair in itertools.combinations(draw, 2):
                self.pairs[pair] += 1

            # Look at triplets
            for triplet in itertools.combinations(draw, 3):
                self.triplets[triplet] += 1

        # Calculate significance of combinations compared to random chance
        total_draws = len(self.data)

        # Expected probability of a pair occurring by chance
        pair_prob = self.nums_per_draw * (self.nums_per_draw - 1) / 2
        pair_prob /= (self.max_num * (self.max_num - 1) / 2)
        expected_pair_count = total_draws * pair_prob

        # Calculate pair significance (how many times more/less likely than random)
        self.pair_significance = {
            pair: count / expected_pair_count
            for pair, count in self.pairs.items()
        }

        # Do the same for triplets
        triplet_prob = (self.nums_per_draw * (self.nums_per_draw - 1) * (self.nums_per_draw - 2)) / 6
        triplet_prob /= ((self.max_num * (self.max_num - 1) * (self.max_num - 2)) / 6)
        expected_triplet_count = total_draws * triplet_prob

        self.triplet_significance = {
            triplet: count / expected_triplet_count
            for triplet, count in self.triplets.items() if count > 1  # Only keep meaningful ones
        }

        return self.pairs, self.triplets

    def _analyze_deltas(self):
        """Analyze the differences between consecutive numbers in a draw."""
        self.delta_patterns = defaultdict(int)

        for draw in self.data:
            deltas = [draw[i + 1] - draw[i] for i in range(len(draw) - 1)]
            delta_tuple = tuple(deltas)
            self.delta_patterns[delta_tuple] += 1

        # Find the most common delta patterns
        self.common_deltas = sorted(
            self.delta_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return self.delta_patterns

    def _analyze_hot_cold(self):
        """Analyze hot (frequently drawn) and cold (rarely drawn) numbers."""
        if not self.frequency:
            self._analyze_frequency()

        # Define hot numbers as top 30% by frequency
        sorted_freq = sorted(self.frequency.items(), key=lambda x: x[1], reverse=True)
        hot_cutoff = int(len(sorted_freq) * 0.3)
        self.hot_numbers = [num for num, _ in sorted_freq[:hot_cutoff]]

        # Define cold numbers as bottom 30% by frequency
        cold_cutoff = int(len(sorted_freq) * 0.7)
        self.cold_numbers = [num for num, _ in sorted_freq[cold_cutoff:]]

        # Analyze if hot/cold numbers tend to appear together
        hot_cold_distribution = []
        for draw in self.data:
            hot_count = sum(1 for num in draw if num in self.hot_numbers)
            cold_count = sum(1 for num in draw if num in self.cold_numbers)
            hot_cold_distribution.append((hot_count, cold_count))

        self.hot_cold_dist = Counter(hot_cold_distribution)

        return self.hot_numbers, self.cold_numbers

    def _analyze_hit_rate(self):
        """Analyze the hit rate of each number position."""
        if len(self.data) < 10:
            return []

        # Track hits per position
        position_hits = [[] for _ in range(self.nums_per_draw)]

        for i in range(len(self.data) - 1):
            current_draw = self.data[i]
            next_draw = self.data[i + 1]

            for pos, num in enumerate(current_draw):
                if num in next_draw:
                    position_hits[pos].append(1)
                else:
                    position_hits[pos].append(0)

        # Calculate hit rate per position
        self.position_hit_rates = [sum(hits) / len(hits) if hits else 0 for hits in position_hits]

        # Calculate overall hit rate (how many numbers from previous draw appear in next draw)
        self.historical_hits = []
        for i in range(len(self.data) - 1):
            current_draw = set(self.data[i])
            next_draw = set(self.data[i + 1])
            hit_count = len(current_draw.intersection(next_draw))
            self.historical_hits.append(hit_count)

        self.avg_hit_count = sum(self.historical_hits) / len(self.historical_hits) if self.historical_hits else 0

        return self.position_hit_rates

    def _create_advanced_features(self):
        """Create advanced features for machine learning."""
        if len(self.data) < 10:
            return

        # Look for cyclical patterns
        self.cycles = {}
        max_cycle_length = min(10, len(self.data) // 3)

        for cycle_length in range(2, max_cycle_length + 1):
            matches = []
            for i in range(len(self.data) - cycle_length):
                current_set = set(self.data[i])
                future_set = set(self.data[i + cycle_length])
                overlap = len(current_set.intersection(future_set))
                matches.append(overlap)

            if matches:
                self.cycles[cycle_length] = sum(matches) / len(matches) / self.nums_per_draw

        # Find the most promising cycle length
        self.best_cycle = max(self.cycles.items(), key=lambda x: x[1])[0] if self.cycles else None

        return self.cycles

    def prepare_ml_features(self):
        """Prepare features for machine learning models."""
        if len(self.data) < 15:
            print("Not enough data for ML prediction")
            return None, None

        # Create features from the last N draws
        window_size = 5  # Look at last 5 draws
        X = []
        y = []

        for i in range(len(self.data) - window_size):
            # Use window_size consecutive draws as features
            features = []

            # Basic features - the numbers themselves
            for j in range(window_size):
                features.extend(self.data[i + j])

            # Add frequency features
            for j in range(window_size):
                draw = self.data[i + j]
                for num in draw:
                    features.append(self.frequency.get(num, 0))

            # Add positional features
            for j in range(window_size):
                for pos, num in enumerate(self.data[i + j]):
                    features.append(pos * num)

            # Add delta features
            for j in range(window_size):
                draw = self.data[i + j]
                for k in range(len(draw) - 1):
                    features.append(draw[k + 1] - draw[k])

            X.append(features)
            y.append(self.data[i + window_size])

        # Normalize features
        X = self.scaler.fit_transform(X)

        return X, y

    def train_ml_models(self):
        """Train machine learning models for prediction."""
        X, y = self.prepare_ml_features()
        if X is None or not len(X):
            return None

        # Train separate models for each position
        self.models = []
        for pos in range(self.nums_per_draw):
            y_pos = [draw[pos] for draw in y]

            # Train multiple models for ensemble
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            nn_model = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)

            # Train all models
            rf_model.fit(X, y_pos)
            gb_model.fit(X, y_pos)
            try:
                nn_model.fit(X, y_pos)
            except:
                nn_model = None  # Neural network might fail on small datasets

            self.models.append((rf_model, gb_model, nn_model))

        return self.models

    def predict_next_draw(self):
        """Generate a prediction for the next lottery draw."""
        if not self.data:
            print("No data available. Please load data first.")
            return []

        # Create models if not already created
        if not hasattr(self, 'models'):
            self.train_ml_models()

        prediction_components = []

        # 1. ML-based prediction if we have enough data
        ml_prediction = self._get_ml_prediction()
        if ml_prediction:
            prediction_components.append(ml_prediction)

        # 2. Statistical prediction based on frequency and trends
        stat_prediction = self._get_statistical_prediction()
        prediction_components.append(stat_prediction)

        # 3. Pattern-based prediction
        pattern_prediction = self._get_pattern_prediction()
        prediction_components.append(pattern_prediction)

        # 4. Cycle-based prediction if applicable
        cycle_prediction = self._get_cycle_prediction()
        if cycle_prediction:
            prediction_components.append(cycle_prediction)

        # 5. Delta-based prediction
        delta_prediction = self._get_delta_prediction()
        prediction_components.append(delta_prediction)

        # Combine all predictions with weighted voting
        all_votes = []
        weights = [3, 2, 2, 2, 1]  # Weights for each prediction method

        for i, prediction in enumerate(prediction_components):
            if prediction:
                weight = weights[i] if i < len(weights) else 1
                all_votes.extend([(num, weight) for num in prediction])

        # Count weighted votes
        vote_counts = defaultdict(float)
        for num, weight in all_votes:
            vote_counts[num] += weight

        # Get top voted numbers
        top_numbers = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
        top_nums = [num for num, _ in top_numbers[:self.nums_per_draw * 2]]

        # Final filtering step - ensure diversity and balance
        final_prediction = self._balance_prediction(top_nums)

        # Store the prediction
        self.last_prediction = sorted(final_prediction)

        return sorted(final_prediction)

    def _get_ml_prediction(self):
        """Get prediction from ML models."""
        if not hasattr(self, 'models') or not self.models:
            return None

        # Prepare features for prediction
        window_size = 5
        if len(self.data) < window_size:
            return None

        features = []
        for i in range(min(window_size, len(self.data))):
            features.extend(self.data[-window_size + i])

        # Add frequency features
        for i in range(min(window_size, len(self.data))):
            draw = self.data[-window_size + i]
            for num in draw:
                features.append(self.frequency.get(num, 0))

        # Add positional features
        for i in range(min(window_size, len(self.data))):
            for pos, num in enumerate(self.data[-window_size + i]):
                features.append(pos * num)

        # Add delta features
        for i in range(min(window_size, len(self.data))):
            draw = self.data[-window_size + i]
            for j in range(len(draw) - 1):
                features.append(draw[j + 1] - draw[j])

        # Normalize features
        features = self.scaler.transform([features])

        # Get predictions from each model
        predictions = []
        for pos in range(self.nums_per_draw):
            pos_predictions = []
            rf_model, gb_model, nn_model = self.models[pos]

            # Get predictions from each model
            rf_pred = int(round(rf_model.predict(features)[0]))
            gb_pred = int(round(gb_model.predict(features)[0]))

            pos_predictions.append(rf_pred)
            pos_predictions.append(gb_pred)

            if nn_model:
                try:
                    nn_pred = int(round(nn_model.predict(features)[0]))
                    pos_predictions.append(nn_pred)
                except:
                    pass

            # Take the median prediction for this position
            median_pred = int(round(np.median(pos_predictions)))
            predictions.append(max(self.min_num, min(self.max_num, median_pred)))

        # Ensure no duplicates
        prediction_set = set(predictions)
        while len(prediction_set) < self.nums_per_draw:
            # If we have duplicates, add high frequency numbers
            high_freq = [num for num, _ in Counter(self.frequency).most_common()]
            for num in high_freq:
                if num not in prediction_set:
                    prediction_set.add(num)
                    if len(prediction_set) >= self.nums_per_draw:
                        break

        return sorted(list(prediction_set))

    def _get_statistical_prediction(self):
        """Get prediction based on statistical properties."""
        if not self.frequency:
            return None

        # Use frequency deviation to choose numbers
        candidates = list(range(self.min_num, self.max_num + 1))

        # Weight by frequency deviation (numbers that appear more often than expected)
        weights = []
        for num in candidates:
            dev = self.frequency_deviation.get(num, 0)
            # Make sure all weights are positive
            weight = max(0.1, 1 + dev)

            # Add trend component if available
            if hasattr(self, 'frequency_trend') and num in self.frequency_trend:
                trend = self.frequency_trend[num]
                weight *= (1 + max(0, trend * 5))  # Boost weights for numbers trending up

            weights.append(weight)

        # Choose numbers based on weights
        prediction = set()
        while len(prediction) < self.nums_per_draw:
            chosen = random.choices(candidates, weights=weights, k=1)[0]
            prediction.add(chosen)

        return sorted(list(prediction))

    def _get_pattern_prediction(self):
        """Get prediction based on number patterns."""
        if not hasattr(self, 'pair_significance') or not self.pair_significance:
            return None

        # Use significant pairs to construct a prediction
        sig_pairs = sorted(
            self.pair_significance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]

        # Count how many times each number appears in significant pairs
        num_counts = Counter()
        for pair, _ in sig_pairs:
            num_counts[pair[0]] += 1
            num_counts[pair[1]] += 1

        # Choose numbers that appear in multiple significant pairs
        prediction = [num for num, _ in num_counts.most_common(self.nums_per_draw)]

        # Fill in if needed
        while len(prediction) < self.nums_per_draw:
            # Add from triplet patterns if available
            if hasattr(self, 'triplet_significance') and self.triplet_significance:
                for triplet, _ in sorted(
                        self.triplet_significance.items(),
                        key=lambda x: x[1],
                        reverse=True
                ):
                    for num in triplet:
                        if num not in prediction:
                            prediction.append(num)
                            if len(prediction) >= self.nums_per_draw:
                                break
                    if len(prediction) >= self.nums_per_draw:
                        break

            # If still need more, add from frequency
            if len(prediction) < self.nums_per_draw:
                for num, _ in Counter(self.frequency).most_common():
                    if num not in prediction:
                        prediction.append(num)
                        if len(prediction) >= self.nums_per_draw:
                            break

        return sorted(prediction[:self.nums_per_draw])

    def _get_cycle_prediction(self):
        """Get prediction based on cyclical patterns."""
        if not hasattr(self, 'best_cycle') or not self.best_cycle:
            return None

        cycle = self.best_cycle
        if len(self.data) <= cycle:
            return None

        # Get numbers from cycle_length draws ago
        reference_draw = self.data[-cycle]

        # Start with those numbers but ensure we have enough
        prediction = list(reference_draw)

        # If we need more numbers, add from the draw before or after
        if len(self.data) > cycle + 1:
            additional_draw = self.data[-(cycle + 1)]
            for num in additional_draw:
                if num not in prediction:
                    prediction.append(num)
                    if len(prediction) >= self.nums_per_draw:
                        break

        # If we still need more, add from frequency
        while len(prediction) < self.nums_per_draw:
            for num, _ in Counter(self.frequency).most_common():
                if num not in prediction:
                    prediction.append(num)
                    if len(prediction) >= self.nums_per_draw:
                        break

        return sorted(prediction[:self.nums_per_draw])

    def _get_delta_prediction(self):
        """Get prediction based on delta patterns."""
        if not hasattr(self, 'common_deltas') or not self.common_deltas:
            return None

        # Use the most common delta pattern
        delta_pattern = self.common_deltas[0][0]

        # Apply this delta pattern to the last draw
        if not self.data:
            return None

        last_draw = self.data[-1]
        prediction = set()

        # Try different starting points
        for start_idx in range(len(last_draw)):
            if start_idx + len(delta_pattern) >= len(last_draw):
                continue

            # Generate prediction based on this starting point
            current_pred = [last_draw[start_idx]]
            for delta in delta_pattern:
                next_num = current_pred[-1] + delta
                if self.min_num <= next_num <= self.max_num:
                    current_pred.append(next_num)

            # Add valid numbers to our prediction
            for num in current_pred:
                if self.min_num <= num <= self.max_num:
                    prediction.add(num)

        # Fill in if needed
        prediction_list = sorted(list(prediction))
        while len(prediction_list) < self.nums_per_draw:
            # Add from frequency
            for num, _ in Counter(self.frequency).most_common():
                if num not in prediction_list:
                    prediction_list.append(num)
                    if len(prediction_list) >= self.nums_per_draw:
                        break

        return sorted(prediction_list[:self.nums_per_draw])

    def _balance_prediction(self, candidates):
        """Balance the prediction to ensure diversity."""
        if len(candidates) <= self.nums_per_draw:
            return candidates

        prediction = []

        # 1. Try to include both hot and cold numbers
        hot_count = 0
        cold_count = 0

        # target distribution based on historical analysis
        target_hot_cold = None
        if hasattr(self, 'hot_cold_dist') and self.hot_cold_dist:
            target_hot_cold = self.hot_cold_dist.most_common(1)[0][0]

        # First pass - add numbers according to hot/cold distribution
        for num in candidates:
            in_hot = hasattr(self, 'hot_numbers') and num in self.hot_numbers
            in_cold = hasattr(self, 'cold_numbers') and num in self.cold_numbers

            if target_hot_cold:
                target_hot, target_cold = target_hot_cold

                if in_hot and hot_count < target_hot:
                    prediction.append(num)
                    hot_count += 1
                elif in_cold and cold_count < target_cold:
                    prediction.append(num)
                    cold_count += 1

                if len(prediction) >= self.nums_per_draw:
                    break

        # 2. Second pass - make sure we include numbers from last draw based on hit rate
        if hasattr(self, 'avg_hit_count') and self.avg_hit_count > 0 and self.data:
            target_hits = round(self.avg_hit_count)
            hit_count = 0

            last_draw = set(self.data[-1])
            for num in candidates:
                if num in last_draw and num not in prediction and hit_count < target_hits:
                    prediction.append(num)
                    hit_count += 1

                if len(prediction) >= self.nums_per_draw:
                    break

        # 3. Third pass - fill with remaining top candidates
        for num in candidates:
            if num not in prediction:
                prediction.append(num)
                if len(prediction) >= self.nums_per_draw:
                    break

        return prediction[:self.nums_per_draw]

    def evaluate_last_prediction(self, actual_result):
        """Evaluate how well the last prediction performed."""
        if not self.last_prediction:
            print("No previous prediction to evaluate")
            return 0

        self.last_actual = sorted(actual_result)

        # Calculate number of matches
        matches = len(set(self.last_prediction).intersection(set(actual_result)))
        match_percentage = (matches / self.nums_per_draw) * 100

        print(f"\nPrediction evaluation:")
        print(f"Predicted: {self.last_prediction}")
        print(f"Actual:    {sorted(actual_result)}")
        print(f"Matches:   {matches}/{self.nums_per_draw} ({match_percentage:.1f}%)")

        # Add this result to our data for future predictions
        if sorted(actual_result) not in self.data:
            self.data.append(sorted(actual_result))
            # Re-analyze with new data
            self.analyze_data()
            self.train_ml_models()

        return matches

    def run(self, evaluate_with=None):
        """Run the full prediction process."""
        self.load_data()
        self.analyze_data()
        self.train_ml_models()

        prediction = self.predict_next_draw()

        print("\nData Analysis Results:")
        print(f"Total draws analyzed: {len(self.data)}")

        if hasattr(self, 'best_cycle') and self.best_cycle:
            print(f"Best cycle length detected: {self.best_cycle} draws")

        if hasattr(self, 'avg_hit_count'):
            print(f"Average carry-over numbers: {self.avg_hit_count:.2f} per draw")

        print("\nPREDICTION DETAILS:")
        ml_pred = self._get_ml_prediction()
        if ml_pred:
            print(f"Machine learning prediction: {ml_pred}")

        stat_pred = self._get_statistical_prediction()
        print(f"Statistical prediction: {stat_pred}")

        pattern_pred = self._get_pattern_prediction()
        print(f"Pattern-based prediction: {pattern_pred}")

        cycle_pred = self._get_cycle_prediction()
        if cycle_pred:
            print(f"Cycle-based prediction: {cycle_pred}")

        delta_pred = self._get_delta_prediction()
        print(f"Delta-based prediction: {delta_pred}")

        print("\nFINAL PREDICTION:")
        print(f"The next 5 numbers are likely to be: {prediction}")

        # Evaluate if actual result is provided
        if evaluate_with:
            self.evaluate_last_prediction(evaluate_with)

        return prediction


def main(file_path="data.txt", actual_result=None):
    """Main function to orchestrate the prediction process."""
    predictor = LotteryPredictor(file_path=file_path)

    if actual_result:
        # Convert string input to list if needed
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

    # Check if actual result is provided as command line argument
    if len(sys.argv) > 1:
        actual_result = [int(x) for x in sys.argv[1:]]

    main(file_path, actual_result)