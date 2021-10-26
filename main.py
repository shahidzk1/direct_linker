from HiggsRanking.models.DirectRanker import DirectRanker
from HiggsRanking.models.Cnn import cnn
from HiggsRanking.models.ListNet import ListNet
from HiggsRanking.models.Stacker import stacker
from HiggsRanking.models.MLP import MLP
from HiggsRanking.helpers import auc_cls, read_higgs_data, s_0_cls, nDCG_cls, higgs_ranking_metric

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, PredefinedSplit, GridSearchCV

import argparse
import json
import os

from datetime import datetime
import pandas as pd
import numpy as np
from functools import partial

TIMESTR = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
GRID_ROOT = 'gridsearch_results'

def write_results(path, dictionary):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, indent=4)

def load_params(d):
    with open(d, 'r') as f:
        d = json.load(f)
    datasets = d["dataset"]
    d.pop('dataset')
    preprocess_fn = d["preprocess_fn"]
    d.pop('preprocess_fn')
    models = d["model"]
    d.pop("model")
    return d, datasets, preprocess_fn, models

def prepare_gird_results(results, test_auc, test_s0, model_name, dataset_name, random_seed, use_weights, checkpoint, score_name="auc_s0"):
    df = pd.concat([
        pd.DataFrame(results["params"]),
        pd.DataFrame(results["mean_test_{}".format(score_name)], columns=["mean_vali_{}".format(score_name)]),
        pd.DataFrame({'test_auc': [test_auc for i in results["mean_test_{}".format(score_name)]]}),
        pd.DataFrame({'test_s0': [test_s0 for i in results["mean_test_{}".format(score_name)]]}),
        pd.DataFrame({'model_name': [model_name for i in results["mean_test_{}".format(score_name)]]}),
        pd.DataFrame({'dataset_name': [dataset_name for i in results["mean_test_{}".format(score_name)]]}),
        pd.DataFrame({'random_seed': [random_seed for i in results["mean_test_{}".format(score_name)]]}),
        pd.DataFrame({'use_weights': [use_weights for i in results["mean_test_{}".format(score_name)]]}),
        pd.DataFrame({'checkpoint': [checkpoint for i in results["mean_test_{}".format(score_name)]]}),
    ], axis=1)
    print(df)
    return df

def get_best_params(results, score_name="auc_s0"):
    df = pd.concat([
        pd.DataFrame(results["params"]),
        pd.DataFrame(results["mean_test_{}".format(score_name)], columns=["mean_vali_{}".format(score_name)])
    ], axis=1)
    df = df.iloc[df["mean_vali_{}".format(score_name)].idxmax()].to_dict()
    del df["mean_vali_auc_s0"]
    return df
    
def get_higgs_challenge_data(impute='mean', scaler='quantile', pca=True, use_weights=True):
    path = '~/ranking/direct_ranker_paper/data/higgs_challenge/atlas-higgs-challenge-2014-v2.csv'
    df = pd.read_csv(path)
    
    if impute == 'cut':
        df = df[df["PRI_jet_num"] == 2]
    x_names = set(df.columns) - {'Weight', 'Label', 'KaggleSet', 'KaggleWeight', 'EventId'}
    x = df.loc[:, x_names].values
    
    if impute == 'mean':
        imp = SimpleImputer(missing_values=-999, strategy='mean')
        x = imp.fit_transform(x)
    elif impute == 'zeros':
        x[x == -999] = 0
    
    if scaler == 'quantile':
        preprocess_fn = QuantileTransformer(random_state=42)
        x = preprocess_fn.fit_transform(x)
    elif scaler == 'standard':
        scaler = StandardScaler()
        scaler.fit_transform(x)
        
    if pca:
        pca = decomposition.PCA(whiten=True)
        pca.fit(x)
        x = pca.transform(x)
    
    y = [1 if v == "s" else 0 for v in df['Label'].values]
    w = df['Weight'].values
    dataset_info = {
        "name": 'higgs_challenge',
        "num_features": len(x[0]),
        "use_weights": use_weights
    }
    
    dataset1 = x, np.array(y), w
    dataset2 = x, np.array(y), w
    data = dataset1, dataset2
    
    return data, dataset_info

def prepare_data(dataset_name, preprocess_str, test_data=False, random_state=42, use_weights=True):
    dataset_str = dataset_name.lower()
    
    if dataset_str == "higgs_challenge":
    	return get_higgs_challenge_data(impute='cut', pca=False, use_weights=use_weights)

    higgs_real = '../../../direct_ranker_paper/data/new_higgs_data/higgs_real.csv'
    higgs_unreal = '../../../direct_ranker_paper/data/new_higgs_data/higgs_unreal.csv'
    ttbar_real = '../../../direct_ranker_paper/data/new_higgs_data/ttbar_real.csv'
    ttbar_unreal = '../../../direct_ranker_paper/data/new_higgs_data/ttbar_unreal.csv'

    if 'minmax' in preprocess_str:
        preprocess_fn = MinMaxScaler()
    elif 'standard' in preprocess_str:
        preprocess_fn = StandardScaler()
    elif 'quantile' in preprocess_str:
        preprocess_fn = QuantileTransformer(random_state=random_state)
    else:
        raise ValueError('Preprocessing identifier {} not recognized.'.format(preprocess_str))

    dataset_real = read_higgs_data(
        higgs_real,
        ttbar_real,
        test_data=test_data,
        luminosity=156000,
        preprocess_fn=preprocess_fn
    )
    dataset_unreal = read_higgs_data(
        higgs_unreal,
        ttbar_unreal,
        test_data=test_data,
        luminosity=156000,
        preprocess_fn=preprocess_fn
    )

    data = dataset_real, dataset_unreal

    dataset_info = {
        "name": dataset_str,
        "num_features": len(data[0][0][0]),
        "use_weights": use_weights
    }
    return data, dataset_info

def reweight(w_subset, samples_subset, samples_dataset):
    return w_subset * samples_dataset/samples_subset

def run_experiment(model_str, parameters, dataset, dataset_info, out_dir, num_jobs, gridsearch):
    # prepare output dirs
    target_dir = 'results/{}/{}/{}'.format(out_dir, model_str, dataset_info["name"])
    os.makedirs(target_dir)
    # get cross validation split
    skf = StratifiedKFold(n_splits=parameters["n_splits"], shuffle=True, random_state=parameters["random_seed"][0])
    n_splits = parameters["n_splits"]
    parameters.pop("n_splits")
    random_seed = parameters["random_seed"]
    parameters.pop("random_seed")
    # generate score for gridsearch
    if 'Tree' == model_str:
        bdt = True
    else:
        bdt = False

    # prepare model
    if 'DirectRanker' == model_str:
        model = DirectRanker(
            out_dir=target_dir,
            num_features=dataset_info['num_features']
        )
        parameters.pop("early_stopping_lookback")
        parameters.pop("hidden_layers_conv")
        parameters.pop("load_save_path")
        parameters.pop("layer_filters_conv")
        parameters.pop("kernel_sizes_conv")
        parameters.pop("num_last_conv_layer")
        parameters.pop("n_estimators")
        parameters.pop("kernel_regularizer_conv")
        parameters.pop("permutation_layer_size")
        parameters.pop("conv_activations_conv")
        parameters.pop("norm")
    elif 'Stacker' == model_str:
        model = stacker(
            out_dir=target_dir,
            num_features=dataset_info['num_features']
        )
        parameters.pop("n_estimators")
    elif 'Cnn' == model_str:
        model = cnn(
            out_dir=target_dir,
            num_features=dataset_info['num_features']
        )
        parameters.pop("early_stopping_lookback")
        parameters.pop('hidden_layers_dr')
        parameters.pop("load_save_path")
        parameters.pop('scale_factor_train_sample')
        parameters.pop('num_last_conv_layer')
        parameters.pop("n_estimators")
        parameters.pop("kernel_regularizer_dr")
        parameters.pop("drop_out")
        parameters.pop("norm")
    elif 'MLP' == model_str:
        model = MLP(
            out_dir=target_dir,
            num_features=dataset_info['num_features']
        )
        parameters.pop("early_stopping_lookback")
        parameters.pop("hidden_layers_conv")
        parameters.pop("load_save_path")
        parameters.pop("layer_filters_conv")
        parameters.pop("kernel_sizes_conv")
        parameters.pop("num_last_conv_layer")
        parameters.pop("n_estimators")
        parameters.pop("kernel_regularizer_conv")
        parameters.pop("permutation_layer_size")
        parameters.pop("norm")
        parameters.pop("conv_activations_conv")
        parameters.pop("scale_factor_train_sample")
    elif 'ListNet' == model_str:
        model = ListNet(
            out_dir=target_dir,
            num_features=dataset_info['num_features']
        )
        parameters.pop("early_stopping_lookback")
        parameters.pop("hidden_layers_conv")
        parameters.pop("load_save_path")
        parameters.pop("layer_filters_conv")
        parameters.pop("kernel_sizes_conv")
        parameters.pop("num_last_conv_layer")
        parameters.pop('scale_factor_train_sample')
        parameters.pop("n_estimators")
        parameters.pop("drop_out")
        parameters.pop("norm")
        parameters.pop("permutation_layer_size")
    elif 'Tree' == model_str:
        tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=10)
        model = AdaBoostClassifier(base_estimator=tree, n_estimators=150,
                                 learning_rate=0.25, random_state=random_seed[0])
        parameters.pop("hidden_layers_dr")
        parameters.pop("load_save_path")
        parameters.pop("hidden_layers_conv")
        parameters.pop("layer_filters_conv")
        parameters.pop("kernel_sizes_conv")
        parameters.pop('num_last_conv_layer')
        parameters.pop('epoch')
        parameters.pop('scale_factor_train_sample')
        parameters.pop('batch_size')
        parameters.pop("early_stopping_lookback")
        parameters.pop('verbose')
        parameters.pop("kernel_regularizer_conv")
        parameters.pop("kernel_regularizer_dr")
        parameters.pop("validation_size")
        parameters.pop("drop_out")
        parameters.pop("norm")
        parameters.pop("permutation_layer_size")
        parameters.pop("conv_activations_conv")
        parameters.pop("print_summary")
        # TODO: the learning rate here is changed to 0.25 since 0.001 is not performing good for a tree
        parameters.pop('learning_rate')
    else:
        raise ValueError('Parameter "model" {} not recognized.'.format(model_str))

    counter = 0
    x_real, y_real, w_real = dataset[0]
    x_unreal, y_unreal, w_unreal = dataset[1]
    if gridsearch:
        real_idx = list(skf.split(x_real, y_real))
        unreal_idx = list(skf.split(x_unreal, y_unreal))
        for r_idx, ur_idx in zip(real_idx, unreal_idx):
            train_real, test_real = r_idx[0], r_idx[1]
            train_unreal, test_unreal = ur_idx[0], ur_idx[1]
            counter += 1
            print("[FOLD{}] start".format(counter))
            if dataset_info["name"] == "unreal_real":
                x_train = x_unreal[train_unreal]
                y_train = y_unreal[train_unreal]
                w_train = w_unreal[train_unreal]
                x_test = x_real[test_real]
                y_test = y_real[test_real]
                w_test = reweight(w_real[test_real], len(y_test), len(y_real))
            elif dataset_info["name"] == "real_real":
                x_train = x_real[train_real]
                y_train = y_real[train_real]
                w_train = w_real[train_real]
                x_test = x_real[test_real]
                y_test = y_real[test_real]
                w_test = reweight(w_real[test_real], len(y_test), len(y_real))
            elif dataset_info["name"] == "unreal_unreal":
                x_train = x_unreal[train_unreal]
                y_train = y_unreal[train_unreal]
                w_train = w_unreal[train_unreal]
                x_test = x_unreal[test_unreal]
                y_test = y_unreal[test_unreal]
                w_test = reweight(w_unreal[test_unreal], len(y_test), len(y_unreal))
            elif dataset_info["name"] == "real_unreal":
                x_train = x_real[train_real]
                y_train = y_real[train_real]
                w_train = w_real[train_real]
                x_test = x_unreal[test_unreal]
                y_test = y_unreal[test_unreal]
                w_test = reweight(w_unreal[test_unreal], len(y_test), len(y_unreal))
            elif dataset_info["name"] == "higgs_challenge":
                x_train = x_unreal[train_unreal]
                y_train = y_unreal[train_unreal]
                w_train = w_unreal[train_unreal]
                x_test = x_unreal[test_unreal]
                y_test = y_unreal[test_unreal]
                w_test = w_unreal[test_unreal]
            else:
                raise ValueError('No dataset {}.'.format(dataset_info["name"]))

            higgs_ranking_metric_f = partial(
                higgs_ranking_metric,
                bdt=bdt,
                w_0=w_train[y_train == 1][0],
                w_1=w_train[y_train == 0][0]
            )
            scoring = {'auc_s0': higgs_ranking_metric_f}

            cv_split = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_seed[0])
            refit = 'auc_s0'

            cv = GridSearchCV(model,
                              param_grid=parameters,
                              verbose=10,
                              scoring=scoring,
                              cv=cv_split,
                              refit=refit,
                              n_jobs=num_jobs,
                              return_train_score=True)
            cv.fit(x_train, y_train)
            best_estimator = cv.best_estimator_

            auc = auc_cls(best_estimator, x_test, y_test, w_test, cnn_bdt=bdt, use_weights=dataset_info["use_weights"])
            s_0 = s_0_cls(best_estimator, x_test, y_test, w_test, cnn_bdt=bdt, use_weights=dataset_info["use_weights"])
            print("AUC {} S_0 {} from {}".format(auc, s_0, model_str))
            results = cv.cv_results_
            if model_str == "Tree":
                checkpoint = "Tree"
            else:
                checkpoint = best_estimator.checkpoint_path
            results = prepare_gird_results(
                results,
                test_auc=auc,
                test_s0=s_0,
                model_name=model_str,
                dataset_name=dataset_info["name"],
                random_seed=random_seed[0],
                use_weights=dataset_info["use_weights"],
                checkpoint=checkpoint
            )
            os.makedirs(target_dir + "/fold_{}".format(counter))
            write_results(target_dir + "/fold_{}/grid_results.json".format(counter), results.to_dict())
    else:
        if dataset_info["name"] == "unreal_real":
            x_train = x_unreal
            y_train = y_unreal
            x_test = x_real
            y_test = y_real
            w_test = w_real
        elif dataset_info["name"] == "unreal_unreal":
            x = x_unreal
            y = y_unreal
            w = w_unreal
            x_train, x_test, y_train, y_test, w_train, w_test = \
                train_test_split(x, y, w, random_state=random_seed[0], shuffle=True)
            w_test = reweight(w_test, len(w_test), len(y))
        elif dataset_info["name"] == "real_real":
            x = x_real
            y = y_real
            w = w_real
            x_train, x_test, y_train, y_test, w_train, w_test = \
                train_test_split(x, y, w, random_state=random_seed[0], shuffle=True)
            w_test = reweight(w_test, len(w_test), len(y))
        else:
            raise ValueError('No dataset {}.'.format(dataset_info["name"]))

        os.makedirs(target_dir + "/stacker_test")
        for i in range(len(parameters["epoch"])):
            for para in parameters:
                try:
                    setattr(model, para, parameters[para][i])
                except:
                    setattr(model, para, parameters[para][0])
            model.fit(x_train, y_train)
            auc = auc_cls(model, x_test, y_test, w_test, cnn_bdt=bdt, use_weights=dataset_info["use_weights"])
            s_0 = s_0_cls(model, x_test, y_test, w_test, cnn_bdt=bdt, use_weights=dataset_info["use_weights"])
            print("AUC {} S_0 {} from {}".format(auc, s_0, model_str))
            cur_parameters = {}
            for para in parameters:
                try:
                    cur_parameters[para] = parameters[para][i]
                except:
                    cur_parameters[para] = parameters[para][0]
            results = {
                "AUC": auc,
                "s_0": s_0,
                "params": parameters,
                "checkpoints": model.checkpoint_path
            }
            write_results(target_dir + "/stacker_test/stacker_test_{}.json".format(i), results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a gridsearch experiment')
    parser.add_argument('-j', '--grid_path', action='store', type=str,
                        help='Path to the .json file containing the gridsearch hyperparameters.')
    parser.add_argument('-n', '--num_jobs', action='store', default=4, type=int,
                        help='How many parallel jobs to run')
    parser.add_argument('-o', '--out_path', action='store', default='{}/{}'.format(GRID_ROOT, TIMESTR), type=str,
                        help='Out dir for results')
    parser.add_argument('-np', '--noparallel', action='store_true', help='Dont use parallelism. -n will be ignored',
                        default=False)

    args = parser.parse_args()
    parameters, dataset, preprocess_fn, models = load_params(args.grid_path)
    test_data = parameters["test_data"]
    parameters.pop('test_data')
    gridsearch = parameters["gridsearch"]
    parameters.pop('gridsearch')
    use_weights = parameters["use_weights"]
    parameters.pop('use_weights')
    for dset in dataset:
        for pre in preprocess_fn:
            for model in models:
                if dset != 'Tree':
                    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                    del os.environ["CUDA_VISIBLE_DEVICES"]
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                data, data_info = prepare_data(
                    dset,
                    pre,
                    test_data=test_data,
                    random_state=parameters['random_seed'][0],
                    use_weights=use_weights
                )
                param_cur = parameters.copy()
                run_experiment(
                    model_str=model,
                    parameters=param_cur,
                    dataset=data,
                    dataset_info=data_info,
                    out_dir=args.out_path,
                    num_jobs=args.num_jobs,
                    gridsearch=gridsearch
                )
