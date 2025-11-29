import numpy as np
from collections import Counter

def find_best_split(feature_vector, target_vector):
    sorted_idx = np.argsort(feature_vector)
    f_sorted = feature_vector[sorted_idx]
    t_sorted = target_vector[sorted_idx]

    diffs = f_sorted[1:] != f_sorted[:-1]
    if not np.any(diffs):
        return np.array([]), np.array([]), None, None

    thresholds = (f_sorted[1:][diffs] + f_sorted[:-1][diffs]) / 2

    t_cumsum = np.cumsum(t_sorted)
    n_total = len(t_sorted)
    total_1 = t_cumsum[-1]
    total_0 = n_total - total_1

    left_counts = t_cumsum[:-1][diffs]
    left_sizes = np.arange(1, n_total)[diffs]
    left_1 = left_counts
    left_0 = left_sizes - left_1

    right_1 = total_1 - left_1
    right_0 = total_0 - left_0
    right_sizes = n_total - left_sizes

    H_left = 1 - (left_1 / left_sizes)**2 - (left_0 / left_sizes)**2
    H_right = 1 - (right_1 / right_sizes)**2 - (right_0 / right_sizes)**2

    ginis = - (left_sizes / n_total) * H_left - (right_sizes / n_total) * H_right

    idx = np.argmax(ginis)
    return thresholds, ginis, thresholds[idx], ginis[idx]


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        if sub_X.shape[0] < self._min_samples_split or \
           (self._max_depth is not None and depth >= self._max_depth):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best = None
        threshold_best = None
        gini_best = None
        split_best = None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature]

            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                target_class_for_ratio = 1
                clicks = Counter(sub_X[sub_y == target_class_for_ratio, feature])

                ratios = {cat: clicks.get(cat, 0)/counts[cat] for cat in counts}
                sorted_categories = [cat for cat, r in sorted(ratios.items(), key=lambda x: x[1])]
                category_to_num = {cat: i for i, cat in enumerate(sorted_categories)}
                feature_vector = np.array([category_to_num[x] for x in sub_X[:, feature]])
            else:
                raise ValueError("Unknown feature type")

            if len(np.unique(feature_vector)) < 2:
                continue

            thresholds, ginis, _, _ = find_best_split(feature_vector, sub_y)

            if len(ginis) == 0: continue

            idx = np.argmax(ginis)
            gini = ginis[idx]
            threshold = thresholds[idx]

            if feature_type == "real":
                split_tmp = feature_vector < threshold
            else:
                left_cats = [cat for cat, num in category_to_num.items() if num < threshold]
                split_tmp = np.array([x in left_cats for x in sub_X[:, feature]])

            if split_tmp.sum() < self._min_samples_leaf or (~split_tmp).sum() < self._min_samples_leaf:
                continue

            if gini_best is None or gini > gini_best:
                gini_best = gini
                feature_best = feature
                threshold_best = threshold if feature_type == "real" else left_cats
                split_best = split_tmp

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        else:
            node["categories_split"] = threshold_best

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split_best], sub_y[split_best], node["left_child"], depth+1)
        self._fit_node(sub_X[~split_best], sub_y[~split_best], node["right_child"], depth+1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]
        # Защита от отсутствующих данных
        try:
            value = x[feature]
        except IndexError:
            return node.get("class", 0)

        if self._feature_types[feature] == "real":
            if value < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            if value in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, depth=0)

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in X])