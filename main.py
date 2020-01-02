from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
from sklearn.preprocessing.data import MinMaxScaler
from utils import load_data
from algorithm import SelfAdaptiveShifting
import numpy as np
import random
import argparse
from tqdm import tqdm


def parse_arguments():
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")
    parser.add_argument('--data', type=str, default='fourclass')
    parser.add_argument('--scale', default=False, action='store_true')
    parser.add_argument('--kernel', default='rbf')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_arguments()
    random.seed(args.seed)
    X, y = load_data(args)
    if args.scale:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    y_value = np.unique(y)

    f_index = np.where(y == y_value[0])[0]
    s_index = np.where(y == y_value[1])[0]

    target_X, target_y = X[f_index], np.ones(len(f_index))
    outlier_X, outlier_y = X[s_index], -np.ones(len(s_index))
    target_X_train, target_X_test, target_y_train, target_y_test = train_test_split(target_X, target_y, shuffle=True,
                                                                                    random_state=args.seed, test_size=1/3)

    self_adaptive_shifting = SelfAdaptiveShifting(target_X_train)
    self_adaptive_shifting.edge_pattern_detection(args.threshold)
    pseudo_outlier_X = self_adaptive_shifting.generate_pseudo_outliers()
    pseudo_target_X = self_adaptive_shifting.generate_pseudo_targets()
    pseudo_outlier_y = -np.ones(len(pseudo_outlier_X))
    pseudo_target_y = np.ones(len(pseudo_target_X))

    gamma_candidates = [1e-4, 1e-3, 1e-2, 1e-1, 1e-0, 1e+1, 1e+2, 1e+3, 1/np.size(target_X, -1)]
    nu_candidates = [0.005, 0.01, 0.05, 0.1, 0.5]

    best_err = 1.0
    best_gamma, best_nu = 1/np.size(target_X, -1), 0.5
    for gamma in tqdm(gamma_candidates):
        for nu in tqdm(nu_candidates):
            model = OneClassSVM(gamma=gamma, nu=nu).fit(target_X_train)
            err_o = 1 - np.mean(model.predict(pseudo_outlier_X) == pseudo_outlier_y)
            err_t = 1 - np.mean(model.predict(pseudo_target_X) == pseudo_target_y)
            err = float((err_o + err_t) / 2)
            if err < best_err:
                best_err = err
                best_gamma = gamma
                best_nu = nu

    best_model = OneClassSVM(kernel=args.kernel, gamma=best_gamma, nu=best_nu).fit(target_X_train)
    target_pred = best_model.predict(target_X_test)
    outlier_pred = best_model.predict(outlier_X)
    y_pred = np.concatenate((target_pred, outlier_pred))
    y_true = np.concatenate((target_y_test, outlier_y))
    f1 = f1_score(y_true, y_pred, average="binary")
    mcc = matthews_corrcoef(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    print("\n[%s] (gamma: %.4f, nu: %.4f, err: %.4f) \nf1-score: %.4f, mcc: %.4f, acc: %.4f" % (args.data, best_gamma, best_nu, best_err, f1, mcc, acc))

    model = OneClassSVM(kernel=args.kernel).fit(target_X_train)
    target_pred = model.predict(target_X_test)
    outlier_pred = model.predict(outlier_X)
    y_pred = np.concatenate((target_pred, outlier_pred))
    y_true = np.concatenate((target_y_test, outlier_y))
    f1 = f1_score(y_true, y_pred, average="binary")
    mcc = matthews_corrcoef(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    print("\n[%s] (default setting) \nf1-score: %.4f, mcc: %.4f, acc: %.4f" % (args.data, f1, mcc, acc))

    self_adaptive_shifting.visualize()


if __name__ == "__main__":
    main()
