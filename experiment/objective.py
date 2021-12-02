import hashlib
import json
import time
import sys
import os


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multiprocessing import Process, Pipe
from hyperopt import STATUS_OK, STATUS_FAIL
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import metrics

from experiment.algorithm import space as ALGORITHM_SPACE
from experiment.pipeline.prototype import pipeline_conf_to_full_pipeline, get_baseline
from experiment.utils.metrics import my_silhouette_samples
from experiment.pipeline.PrototypeSingleton import PrototypeSingleton


def objective(pipeline_config, algo_config, algorithm, X, y, context, config, step):
    pipeline_hash = hashlib.sha1(json.dumps(pipeline_config, sort_keys=True).encode()).hexdigest()
    algorithm_hash = hashlib.sha1(json.dumps(algo_config, sort_keys=True).encode()).hexdigest()
    item_hash = {
        'pipeline': pipeline_hash,
        'algorithm': algorithm_hash,
        'config': hashlib.sha1(str(pipeline_hash + algorithm_hash).encode()).hexdigest()
    }

    item = {
        'pipeline': pipeline_config,
        'algorithm': algo_config,
        'step': step
    }

    if algorithm == None:
        algorithm = algo_config[0]
        algo_config = algo_config[1]

    pipeline, operators = pipeline_conf_to_full_pipeline(
        pipeline_config, 
        ALGORITHM_SPACE.algorithms.get(algorithm), 
        config['seed'], 
        algo_config
    )
    #print(pipeline)
    history_index = context['history_index'].get(item_hash['config'])
    if history_index is not None:
        return context['history'][history_index]

    start = time.time()
    try:
        Xt, yt = X.copy(), y.copy()
        Xt_to_export, yt_to_export = {}, {}
        labels = PrototypeSingleton.getInstance().getFeaturesFromMask()
        Xt_to_export['original'] = pd.DataFrame(Xt.copy(), columns=[l for l in labels if l != 'None'])
        yt_to_export['original'] = pd.DataFrame(yt.copy(), columns=['target'])
        if len(pipeline.steps) > 1:
            for step, operator in pipeline[:-1].named_steps.items():
                if step == 'outlier':
                    Xt, yt = operator.fit_resample(Xt, yt)
                    yt_to_export[step] = pd.DataFrame(yt.copy(), columns=['target'])
                else:
                    Xt = operator.fit_transform(Xt, None)
                    if step == 'features':
                        mask = pipeline['features'].get_support()
                        labels = PrototypeSingleton.getInstance().getFeaturesFromMask(mask)
                Xt_to_export[step] = pd.DataFrame(Xt.copy(), columns=[l for l in labels if l != 'None'])


        result = pipeline.fit_predict(X, y)
        yt_to_export['pred'] = pd.DataFrame(result.copy(), columns=['target'])
        if config['metric'] == 'SIL':
            score = silhouette_score(Xt, result)
            sil_samples, intra_clust_dists, inter_clust_dists = my_silhouette_samples(Xt, result)
        elif config['metric'] == 'CH':
            score = calinski_harabasz_score(Xt, result)
        elif config['metric'] == 'DBI':
            score = -1 * davies_bouldin_score(Xt, result)
        ami = metrics.adjusted_mutual_info_score(yt, result)
        status = STATUS_OK
    except Exception as e:
        score = float('-inf')
        ami = float('-inf')
        status = STATUS_FAIL
        print(e)
    stop = time.time()

    details_path = os.path.join(config['result_path'], 'details', config['dataset'] + '_' + config['metric'].lower())

    if not os.path.exists(details_path):
        os.makedirs(details_path)

    iteration_number = len(context['history'])
    file_name = config['dataset'] + '_' + config['metric'].lower() + "_" + str(iteration_number)
    try:
        for step, xt_df in Xt_to_export.items():
            xt_df.to_csv(os.path.join(details_path, file_name + "_X_" + step + ".csv"), index=False)
        for step, yt_df in yt_to_export.items():
            yt_df.to_csv(os.path.join(details_path, file_name + "_y_" + step + ".csv"), index=False)
    except Exception as e:
        f= open(os.path.join(details_path, file_name + ".txt"), "a+")
        f.write(str(e))
        f.close()

    item.update({
        'start_time': start,
        'stop_time': stop,
        'duration': stop - start,
        'loss': 1 - score, 
        'status': status, 
        'score': score,
        'ami': ami,
        'iteration': iteration_number,
        'config_hash': item_hash,
        'max_history_score': context['max_history_score'],
        'max_history_step': context['max_history_step'],
        'max_history_score_ami': context['max_history_score_ami'],
        'step': step
    })

    if context['max_history_score'] < score:
        item['max_history_score'] = score
        item['max_history_step'] = step
        item['max_history_score_ami'] = ami
        context['max_history_score'] = score
        context['max_history_step'] = step
        context['max_history_score_ami'] = ami
        context['best_config'] = item

    # Update hash index
    context['history_hash'].append(item_hash['config'])
    context['history_index'][item_hash['config']] = iteration_number
    context['iteration'] = iteration_number

    context['history'].append(item)

    print('{}. Best score: {} ({}) [{}] | Score: {} ({}) [{}]'.format(
        iteration_number,
        item['max_history_score'],
        item['max_history_score_ami'],
        item['max_history_step'][0].upper(),
        item['score'],
        item['ami'],
        item['step'][0].upper(),
        )
    )
    with open(os.path.join(details_path, config['dataset'] + "_context.json"), 'w') as outfile:
        json.dump(context, outfile, indent=4)

    return item

def objective_pipeline(pipeline_config, current_algo_config, algorithm, X, y, context, config):
    return objective(pipeline_config, current_algo_config, algorithm, X, y, context, config, step='pipeline')

def objective_algo(algo_config, current_pipeline_config, algorithm, X, y, context, config):
    return objective(current_pipeline_config, algo_config, algorithm, X, y, context, config, step='algorithm')

def objective_joint(wconfig, algorithm, X, y, context, config):
    return objective(wconfig['pipeline'], wconfig['algorithm'], algorithm, X, y, context, config, step='joint')

def objective_union(wconfig, X, y, context, config):
    return objective(wconfig['pipeline'], wconfig['algorithm'], None, X, y, context, config, step='union')

def get_baseline_score(algorithm, X, y, seed):
    pipeline, _ = pipeline_conf_to_full_pipeline(
        get_baseline(), 
        ALGORITHM_SPACE.algorithms.get(algorithm), 
        seed, 
        {}
    )
    scores = cross_validate(pipeline, 
                    X,
                    y,
                    # scoring=["precision_macro", "recall_macro", "f1_macro", "roc_auc", "accuracy", "balanced_accuracy"],
                    scoring=["balanced_accuracy"],
                    cv=10,
                    n_jobs=-1,
                    return_estimator=False,
                    return_train_score=False,
                    verbose=0)
    score = np.mean(scores['test_balanced_accuracy']) // 0.0001 / 10000
    std = np.std(scores['test_balanced_accuracy']) // 0.0001 / 10000
    return score, std


def wrap_cost(cost_fn, timeout=None, iters=1, verbose=0):
    """Wrap cost function to execute trials safely on a separate process.
    Parameters
    ----------
    cost_fn : callable
        The cost function (aka. objective function) to wrap. It follows the
        same specifications as normal Hyperopt cost functions.
    timeout : int
        Time to wait for process to complete, in seconds. If this time is
        reached, the process is re-tried if there are remaining iterations,
        otherwise marked as a failure. If ``None``, wait indefinitely.
    iters : int
        Number of times to allow the trial to timeout before marking it as
        a failure due to timeout.
    verbose : int
        How verbose this function should be. 0 is not verbose, 1 is verbose.
    Example
    -------
    def objective(args):
        case, val = args
        return val**2 if case else val
    space = [hp.choice('case', [False, True]), hp.uniform('val', -1, 1)]
    safe_objective = wrap_cost(objective, timeout=2, iters=2, verbose=1)
    best = hyperopt.fmin(safe_objective, space, max_evals=100)
    Notes
    -----
    Based on code from https://github.com/hyperopt/hyperopt-sklearn
    """
    def _cost_fn(*args, **kwargs):
        _conn = kwargs.pop('_conn')
        try:
            t_start = time.time()
            rval = cost_fn(*args, **kwargs)
            t_done = time.time()

            if not isinstance(rval, dict):
                rval = dict(loss=rval)
            assert 'loss' in rval, "Returned dictionary must include loss"
            loss = rval['loss']
            assert is_number(loss), "Returned loss must be a number type"
            rval.setdefault('status', hyperopt.STATUS_OK if np.isfinite(loss)
                            else hyperopt.STATUS_FAIL)
            rval.setdefault('duration', t_done - t_start)
            rtype = 'return'

        except Exception as exc:
            rval = exc
            rtype = 'raise'

        # -- return the result to calling process
        _conn.send((rtype, rval))

    def wrapper(*args, **kwargs):
        for k in range(iters):
            conn1, conn2 = Pipe()
            kwargs['_conn'] = conn2
            th = Process(target=_cost_fn, args=args, kwargs=kwargs)
            th.start()
            if conn1.poll(timeout):
                fn_rval = conn1.recv()
                th.join()
            else:
                if verbose >= 1:
                    print("TRIAL TIMED OUT (%d/%d)" % (k+1, iters))
                th.terminate()
                th.join()
                continue

            assert fn_rval[0] in ('raise', 'return')
            if fn_rval[0] == 'raise':
                raise fn_rval[1]
            else:
                return fn_rval[1]

        return {'status': hyperopt.STATUS_FAIL,
                'failure': 'timeout'}

    return wrapper


def wrap_fmin(cost_fn, timeout=None, iters=1, verbose=0):

    def _cost_fn(*args, **kwargs):
        _conn = kwargs.pop('_conn')
        try:
            t_start = time.time()
            rval = cost_fn(*args, **kwargs)
            t_done = time.time()

            if not isinstance(rval, dict):
                rval = dict(loss=rval)
            assert 'loss' in rval, "Returned dictionary must include loss"
            loss = rval['loss']
            assert is_number(loss), "Returned loss must be a number type"
            rval.setdefault('status', hyperopt.STATUS_OK if np.isfinite(loss)
                            else hyperopt.STATUS_FAIL)
            rval.setdefault('duration', t_done - t_start)
            rtype = 'return'

        except Exception as exc:
            rval = exc
            rtype = 'raise'

        # -- return the result to calling process
        _conn.send((rtype, rval))

    def wrapper(*args, **kwargs):
        for k in range(iters):
            conn1, conn2 = Pipe()
            kwargs['_conn'] = conn2
            th = Process(target=_cost_fn, args=args, kwargs=kwargs)
            th.start()
            if conn1.poll(timeout):
                fn_rval = conn1.recv()
                th.join()
            else:
                if verbose >= 1:
                    print("TRIAL TIMED OUT (%d/%d)" % (k+1, iters))
                th.terminate()
                th.join()
                continue

            assert fn_rval[0] in ('raise', 'return')
            if fn_rval[0] == 'raise':
                raise fn_rval[1]
            else:
                return fn_rval[1]

        return {'status': hyperopt.STATUS_FAIL,
                'failure': 'timeout'}

    return wrapper