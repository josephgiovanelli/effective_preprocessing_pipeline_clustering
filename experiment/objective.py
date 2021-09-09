import hashlib
import json
import time
from multiprocessing import Process, Pipe
import sys

from hyperopt import STATUS_OK, STATUS_FAIL
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import metrics

from experiment.algorithm import space as ALGORITHM_SPACE
from experiment.pipeline.prototype import pipeline_conf_to_full_pipeline, get_baseline


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
        #result = pipeline.fit_predict(X, None)
        
        #trained_pipeline = pipeline.fit(X, y)
        #result = trained_pipeline.predict(X)
        #if config['metric'] == 'SIL':
        #    score = silhouette_score(pipeline[0:len(pipeline.steps) - 1].fit_transform(X, None), result)
        #elif config['metric'] == 'CH':
        #    score = calinski_harabasz_score(pipeline[0:len(pipeline.steps) - 1].fit_transform(X, None), result)
        #elif config['metric'] == 'DBI':
        #    score = -1 * davies_bouldin_score(pipeline[0:len(pipeline.steps) - 1].fit_transform(X, None), result)
        #ami = metrics.adjusted_mutual_info_score(y, result)
        scores = cross_validate(pipeline, 
                        X,
                        y,
                        scoring=["precision_weighted", "recall_weighted", "f1_weighted", "accuracy", "balanced_accuracy", "adjusted_mutual_info_score"],
                        cv=10,
                        n_jobs=-1,
                        return_estimator=False,
                        return_train_score=True,
                        verbose=0)
        #print(scores)
        train_precision_weighted = np.mean(scores['train_precision_weighted']) // 0.0001 / 10000
        test_precision_weighted = np.mean(scores['test_precision_weighted']) // 0.0001 / 10000
        train_recall_weighted = np.mean(scores['train_recall_weighted']) // 0.0001 / 10000
        test_recall_weighted = np.mean(scores['test_recall_weighted']) // 0.0001 / 10000
        train_f1_weighted = np.mean(scores['train_f1_weighted']) // 0.0001 / 10000
        test_f1_weighted = np.mean(scores['test_f1_weighted']) // 0.0001 / 10000
        train_accuracy = np.mean(scores['train_accuracy']) // 0.0001 / 10000
        test_accuracy = np.mean(scores['test_accuracy']) // 0.0001 / 10000
        train_adjusted_mutual_info_score = np.mean(scores['train_adjusted_mutual_info_score']) // 0.0001 / 10000
        test_adjusted_mutual_info_score = np.mean(scores['test_adjusted_mutual_info_score']) // 0.0001 / 10000
        train_balanced_accuracy = np.mean(scores['train_balanced_accuracy']) // 0.0001 / 10000
        score = np.mean(scores['test_balanced_accuracy']) // 0.0001 / 10000
        status = STATUS_OK
    except Exception as e:
        train_precision_weighted = float('-inf')
        test_precision_weighted = float('-inf')
        train_recall_weighted = float('-inf')
        test_recall_weighted = float('-inf')
        train_f1_weighted = float('-inf')
        test_f1_weighted = float('-inf')
        train_accuracy = float('-inf')
        test_accuracy = float('-inf')
        train_adjusted_mutual_info_score = float('-inf')
        test_adjusted_mutual_info_score = float('-inf')
        train_balanced_accuracy = float('-inf')
        score = float('-inf')
        status = STATUS_FAIL
        print(e)
    stop = time.time()

    iteration_number = len(context['history'])
    item.update({
        'start_time': start,
        'stop_time': stop,
        'duration': stop - start,
        'loss': 1 - score, 
        'status': status, 
        'train_precision_weighted': train_precision_weighted, 
        'test_precision_weighted': test_precision_weighted, 
        'train_recall_weighted': train_recall_weighted, 
        'test_recall_weighted': test_recall_weighted, 
        'train_f1_weighted': train_f1_weighted, 
        'test_f1_weighted': test_f1_weighted, 
        'train_accuracy': train_accuracy, 
        'test_accuracy': test_accuracy, 
        'train_adjusted_mutual_info_score': train_adjusted_mutual_info_score, 
        'test_adjusted_mutual_info_score': test_adjusted_mutual_info_score, 
        'train_balanced_accuracy': train_balanced_accuracy, 
        'score': score,
        'iteration': iteration_number,
        'config_hash': item_hash,
        'max_history_train_precision_weighted' : context['max_history_train_precision_weighted'],
        'max_history_test_precision_weighted' : context['max_history_test_precision_weighted'],
        'max_history_train_recall_weighted' : context['max_history_train_recall_weighted'],
        'max_history_test_recall_weighted' : context['max_history_test_recall_weighted'],
        'max_history_train_f1_weighted' : context['max_history_train_f1_weighted'],
        'max_history_test_f1_weighted' : context['max_history_test_f1_weighted'],
        'max_history_train_accuracy' : context['max_history_train_accuracy'],
        'max_history_test_accuracy' : context['max_history_test_accuracy'],
        'max_history_train_adjusted_mutual_info_score' : context['max_history_train_adjusted_mutual_info_score'],
        'max_history_test_adjusted_mutual_info_score' : context['max_history_test_adjusted_mutual_info_score'],
        'max_history_train_balanced_accuracy' : context['max_history_train_balanced_accuracy'],
        'max_history_score': context['max_history_score'],
        'max_history_step': context['max_history_step'],
        'step': step
    })

    if context['max_history_score'] < score:
        item['max_history_train_precision_weighted'] = train_precision_weighted
        item['max_history_test_precision_weighted'] = test_precision_weighted
        item['max_history_train_recall_weighted'] = train_recall_weighted
        item['max_history_test_recall_weighted'] = test_recall_weighted
        item['max_history_train_f1_weighted'] = train_f1_weighted
        item['max_history_test_f1_weighted'] = test_f1_weighted
        item['max_history_train_accuracy'] = train_accuracy
        item['max_history_test_accuracy'] = test_accuracy
        item['max_history_train_adjusted_mutual_info_score'] = train_adjusted_mutual_info_score
        item['max_history_test_adjusted_mutual_info_score'] = test_adjusted_mutual_info_score
        item['max_history_train_balanced_accuracy'] = train_balanced_accuracy
        item['max_history_score'] = score
        item['max_history_step'] = step
        context['max_history_train_precision_weighted'] = train_precision_weighted
        context['max_history_test_precision_weighted'] = test_precision_weighted
        context['max_history_train_recall_weighted'] = train_recall_weighted
        context['max_history_test_recall_weighted'] = test_recall_weighted
        context['max_history_train_f1_weighted'] = train_f1_weighted
        context['max_history_test_f1_weighted'] = test_f1_weighted
        context['max_history_train_accuracy'] = train_accuracy
        context['max_history_test_accuracy'] = test_accuracy
        context['max_history_train_adjusted_mutual_info_score'] = train_adjusted_mutual_info_score
        context['max_history_test_adjusted_mutual_info_score'] = test_adjusted_mutual_info_score
        context['max_history_train_balanced_accuracy'] = train_balanced_accuracy
        context['max_history_score'] = score
        context['max_history_step'] = step
        context['best_config'] = item


    # Update hash index
    context['history_hash'].append(item_hash['config'])
    context['history_index'][item_hash['config']] = iteration_number
    context['iteration'] = iteration_number

    context['history'].append(item)

    print('Best score: {} ({}) [{}] | Score: {} ({}) [{}]'.format(
        item['max_history_score'],
        item['max_history_test_adjusted_mutual_info_score'],
        item['max_history_step'][0].upper(),
        item['score'],
        item['test_adjusted_mutual_info_score'],
        item['step'][0].upper(),
        )
    )
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