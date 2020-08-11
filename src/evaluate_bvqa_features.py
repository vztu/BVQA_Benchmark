# -*- coding: utf-8 -*-
"""
This script shows how to apply 80-20 holdout train and validate regression model to predict
MOS from the features
"""
import pandas
import scipy.io
import numpy as np
import argparse
import time
import math
import os
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from scipy.optimize import curve_fit
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats
from concurrent import futures
import functools
import warnings
warnings.filterwarnings("ignore")

def arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, default='BRISQUE',
                      help='Evaluated BVQA model name.')
  parser.add_argument('--dataset_name', type=str, default='KONVID_1K',
                      help='Evaluation dataset.') 
  parser.add_argument('--feature_file', type=str,
                      default='data/KONVID_1K_BRISQUE_feats.mat',
                      help='Pre-computed feature matrix.')
  parser.add_argument('--mos_file', type=str,
                      default='data/KONVID_1K_metadata.csv',
                      help='Dataset MOS scores.')
  parser.add_argument('--out_file', type=str,
                      default='result/KONVID_1K_BRISQUE_SVR_corr.mat',
                      help='Output correlation results')
  parser.add_argument('--color_only', action='store_true',
                      help='Evaluate color values only. (Only for YouTube UGC)')
  parser.add_argument('--log_short', action='store_true',
                      help='Whether log short')
  parser.add_argument('--use_parallel', action='store_true',
                      help='Use parallel for iterations.')
  parser.add_argument('--num_iterations', type=int, default=100,
                      help='Number of iterations of train-test splits')
  parser.add_argument('--max_thread_count', type=int, default=10,
                      help='Number of threads.')
  args = parser.parse_args()
  return args

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
  # 4-parameter logistic function
  logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
  yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
  return yhat

def compute_metrics(y_pred, y):
  '''
  compute metrics btw predictions & labels
  '''
  # compute SRCC & KRCC
  SRCC = scipy.stats.spearmanr(y, y_pred)[0]
  try:
    KRCC = scipy.stats.kendalltau(y, y_pred)[0]
  except:
    KRCC = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]

  # logistic regression btw y_pred & y
  beta_init = [np.max(y), np.min(y), np.mean(y_pred), 0.5]
  popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
  y_pred_logistic = logistic_func(y_pred, *popt)
  
  # compute  PLCC RMSE
  PLCC = scipy.stats.pearsonr(y, y_pred_logistic)[0]
  RMSE = np.sqrt(mean_squared_error(y, y_pred_logistic))
  return [SRCC, KRCC, PLCC, RMSE]

def formatted_print(snapshot, params, duration):
  print('======================================================')
  print('params: ', params)
  print('SRCC_train: ', snapshot[0])
  print('KRCC_train: ', snapshot[1])
  print('PLCC_train: ', snapshot[2])
  print('RMSE_train: ', snapshot[3])
  print('======================================================')
  print('SRCC_test: ', snapshot[4])
  print('KRCC_test: ', snapshot[5])
  print('PLCC_test: ', snapshot[6])
  print('RMSE_test: ', snapshot[7])
  print('======================================================')
  print(' -- ' + str(duration) + ' seconds elapsed...\n\n')

def final_avg(snapshot):
  def formatted(args, pos):
    median = np.median(list(map(lambda x: x[pos], snapshot)))
    stdev = np.std(list(map(lambda x: x[pos], snapshot)))
    print('{}: {} (std: {})'.format(args, median, stdev))

  print('======================================================')
  print('Average training results among all repeated 80-20 holdouts:')
  formatted("SRCC Train", 0)
  formatted("KRCC Train", 1)
  formatted("PLCC Train", 2)
  formatted("RMSE Train", 3)
  print('======================================================')
  print('Average testing results among all repeated 80-20 holdouts:')
  formatted("SRCC Test", 4)
  formatted("KRCC Test", 5)
  formatted("PLCC Test", 6)
  formatted("RMSE Test", 7)
  print('\n\n')

def evaluate_bvqa_one_split(i, X, y, log_short):
  if not log_short:
    print('{} th repeated holdout test'.format(i))
    t_start = time.time()

  # train test split
  X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=math.ceil(8.8*i))

  # grid search CV on the training set
  param_grid = {'C': np.logspace(1, 10, 10, base=2),
                'gamma': np.logspace(-8, 1, 10, base=2)}

  grid = RandomizedSearchCV(SVR(), param_grid, cv=3, n_jobs=-1)
  scaler = preprocessing.MinMaxScaler().fit(X_train)
  X_train = scaler.transform(X_train)
  # grid search
  grid.fit(X_train, y_train)
  best_params = grid.best_params_
  # init model
  regressor = SVR(C=best_params['C'], gamma=best_params['gamma'])
  # re-train the model using the best alpha
  regressor.fit(X_train, y_train)
  # predictions
  y_train_pred = regressor.predict(X_train)
  X_test = scaler.transform(X_test)
  y_test_pred = regressor.predict(X_test)
  # compute metrics
  metrics_train = compute_metrics(y_train_pred, y_train)
  metrics_test = compute_metrics(y_test_pred, y_test)
  # print values
  if not log_short:
    t_end = time.time()
    formatted_print(metrics_train + metrics_test, best_params, (t_end - t_start))
  return best_params, metrics_train, metrics_test
  
def main(args):
  df = pandas.read_csv(args.mos_file, skiprows=[], header=None)
  array = df.values
  if args.dataset_name == 'YOUTUBE_UGC':
      y = array[1:,4]
  else: # for LIVE-VQC & KONVID_1k
      y = array[1:,1] 
  y = np.array(list(y), dtype=np.float)
  X_mat = scipy.io.loadmat(args.feature_file)
  X = np.asarray(X_mat['feats_mat'], dtype=np.float)

  '''57 grayscale videos in YOUTUBE_UGC dataset, we do not consider them for fair comparison'''
  if args.color_only and args.dataset_name == 'YOUTUBE_UGC':
      gray_indices = [3,6,10,22,23,46,51,52,68,74,77,99,103,122,136,141,158,173,368,426,467,477,506,563,594,\
      639,654,657,666,670,671,681,690,697,702,703,710,726,736,764,768,777,786,796,977,990,1012,\
      1015,1023,1091,1118,1205,1282,1312,1336,1344,1380]
      gray_indices = [idx - 1 for idx in gray_indices]
      X = np.delete(X, gray_indices, axis=0)
      y = np.delete(y, gray_indices, axis=0)
  ## preprocessing
  X[np.isinf(X)] = np.nan
  imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X)
  X = imp.transform(X)

  all_iterations = []
  t_overall_start = time.time()
  # 100 times random train-test splits
  if args.use_parallel is True:
    evaluate_bvqa_one_split_partial = functools.partial(
       evaluate_bvqa_one_split, X=X, y=y, log_short=args.log_short)
    with futures.ThreadPoolExecutor(max_workers=args.max_thread_count) as executor:
      iters_future = [
          executor.submit(evaluate_bvqa_one_split_partial, i)
          for i in range(1, args.num_iterations)]
      for future in futures.as_completed(iters_future):
        best_params, metrics_train, metrics_test = future.result()
        all_iterations.append(metrics_train + metrics_test)
  else:
    for i in range(1, args.num_iterations):
      best_params, metrics_train, metrics_test = evaluate_bvqa_one_split(
          i, X, y, args.log_short)
      all_iterations.append(metrics_train + metrics_test)

  # formatted print overall iterations
  final_avg(all_iterations)
  print('Overall {} secs lapsed..'.format(time.time() - t_overall_start))
  # save figures
  dir_path = os.path.dirname(args.out_file)
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
  scipy.io.savemat(args.out_file, 
      mdict={'all_iterations': np.asarray(all_iterations,dtype=np.float)})

if __name__ == '__main__':
  args = arg_parser()
  main(args)