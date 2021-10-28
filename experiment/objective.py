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
        result = pipeline.fit_predict(X, y)
        if len(pipeline.steps) > 1:
            if pipeline.steps[-2][0] == 'outlier':
                Xt, y = pipeline[:-1].fit_resample(X, y)
            else:
                Xt = pipeline[:-1].fit_transform(X, None)
        else:
            Xt = X.copy()
        #print(result)
        if config['metric'] == 'SIL':
            score = silhouette_score(Xt, result)
            sil_samples, intra_clust_dists, inter_clust_dists = my_silhouette_samples(Xt, result)
        elif config['metric'] == 'CH':
            score = calinski_harabasz_score(Xt, result)
        elif config['metric'] == 'DBI':
            score = -1 * davies_bouldin_score(Xt, result)
        ami = metrics.adjusted_mutual_info_score(y, result)
        status = STATUS_OK
    except Exception as e:
        score = float('-inf')
        ami = float('-inf')
        status = STATUS_FAIL
        print(e)

    plots_path = os.path.join("plots", config['dataset'] + '_' + config['metric'].lower())

    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    iteration_number = len(context['history'])
    file_name = config['dataset'] + '_' + config['metric'].lower() + "_" + str(iteration_number)
    try:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = np.array(['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'grey', 'olive', 'cyan', 'indigo', 'black'])
        Xt = pd.DataFrame(Xt)
        y_pred = pd.DataFrame(result)
        y = pd.DataFrame(y)
        min, max = Xt.min().min(), Xt.max().max()
        xs = Xt.iloc[:, 0]
        ys = [min] * Xt.shape[0] if Xt.shape[1] < 2 else Xt.iloc[:, 1]
        zs = [min] * Xt.shape[0] if Xt.shape[1] < 3 else Xt.iloc[:, 2]
        ax.scatter(xs, ys,  zs, c=[colors[int(i)] for i in result])
        min, max = Xt.min().min(), Xt.max().max()
        ax.set_xlim([min, max])
        ax.set_ylim([min, max])
        ax.set_zlim([min, max])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        fig.savefig(os.path.join(plots_path, file_name + "_scatter.png"))
        
        if config['metric'] == 'SIL':
            fig, ax = plt.subplots(1, 3)
            fig.set_size_inches(18, 7)
            n_clusters = y_pred.iloc[:, 0].unique().size
            my_silhouette_values = pd.DataFrame({
                'silhouette': sil_samples, 
                'inter_dists': inter_clust_dists, 
                'intra_dists': intra_clust_dists, 
                'y_pred': result,})
            for j in range(len(ax)):
                y_lower = 10
                ax[j].set_xlim([-0.1, 1 if j == 0 else my_silhouette_values.max().max()])
                ax[j].set_yticks([])
                ax[j].set_ylim([0, Xt.shape[0] + (n_clusters + 1) * 10])
                ax[j].set_title(my_silhouette_values.columns[j])
                ax[j].set_xlabel("values")
                for i in range(n_clusters):
                    ith_cluster_silhouette_values = my_silhouette_values[my_silhouette_values['y_pred'] == i]
                    ith_cluster_silhouette_values = ith_cluster_silhouette_values.sort_values(by=['silhouette'])
                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i
                        
                    ax[j].fill_betweenx(
                        np.arange(y_lower, y_upper),
                        0,
                        ith_cluster_silhouette_values.iloc[:, j],
                        facecolor=colors[i],
                        edgecolor=colors[i],
                        alpha=0.7,
                    )

                    if j == 0:
                        ax[j].axvline(x=score, color="red", linestyle="--")  
                        ax[j].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i)) 
                        ax[j].set_ylabel("Cluster label")
                    y_lower = y_upper + 10  
            plt.tight_layout()
            fig.savefig(os.path.join(plots_path, file_name + "_silhouette.png"))
        plt.close('all')
        Xt.to_csv(os.path.join(plots_path, file_name + "_Xt.csv"), index=False, header=False)
        y_pred.to_csv(os.path.join(plots_path, file_name + "_y_pred.csv"), index=False, header=False)
        y.to_csv(os.path.join(plots_path, file_name + "_y.csv"), index=False, header=False)
    except:
        f= open(os.path.join(plots_path, file_name + ".txt"), "a+")
        f.write("An error occured.")
        f.close()

    stop = time.time()

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
    with open(os.path.join(plots_path, config['dataset'] + "_context.json"), 'w') as outfile:
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