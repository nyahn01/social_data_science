#!/usr/bin/env python3
# ------------------------- CODE CELL injected: setup --------------------------
__IB_FLAG__ = True
__IMPORT_FILTER__ = globals().get('IMPORT_FILTER', None)
__PLOT_REGISTRY__ = []
__LABEL__ = None


# If matplotlib is available on the test system, it's set to headless mode and all plots are stored
# on disk rather than displayed.
try:
    print("use 'Agg' backend for matplotlib")
    import matplotlib as mpl
    mpl.use('Agg')

    import matplotlib.pyplot as plt

    # store current matplotlib figure to disk
    def _dump_figure():
        from pathlib import Path
        from autograde.util import snake_case

        # check if something was plotted
        if plt.gcf().get_axes():
            global __LABEL__
            global __PLOT_REGISTRY__

            # ensure plots folder exists
            root = Path('figures')
            if not root.is_dir():
                root.mkdir()

            # infer name
            name = snake_case(f'fig_{__LABEL__}_{len(__PLOT_REGISTRY__) + 1}')
            path = root.joinpath(name)
            __PLOT_REGISTRY__.append(path)

            # store current figure
            print(f'save current figure: {path}')
            plt.savefig(path)
            plt.close()

    plt.show = lambda *args, **kwargs: _dump_figure()

except ImportError:
    pass


dump_figure = globals().get('_dump_figure', lambda *args, **kwargs: None)


# inform user about import filters used
if __IMPORT_FILTER__ is not None:
    regex, blacklist = __IMPORT_FILTER__
    print(f'set import filter: regex=r"{regex}", blacklist={blacklist}')

# EXECUTED IN 0.325s
# STDOUT
#     use 'Agg' backend for matplotlib


# ------------------------------- CODE CELL nb-1 -------------------------------
# credentials of all team members (you may add or remove items from the dictionary, but keep this dictionary structure)
team_members = [
    {
        'first_name': 'Na Young',
        'last_name': 'Ahn',
        'student_id': 392326
    },
    {
        'first_name': 'Esther',
        'last_name': 'Tala',
        'student_id': 368095
    },
    {
        'first_name': 'Mika',
        'last_name': 'Rosin',
        'student_id': 395049
    },
    {
        'first_name': 'Laurin',
        'last_name': 'Ellenbeck',
        'student_id': 372280
    }
]

# injected by test
dump_figure()

# EXECUTED IN 0.00192s


# ------------------------------- CODE CELL nb-2 -------------------------------
# general imports may go here
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from IPython.display import display, HTML
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# injected by test
dump_figure()

# EXECUTED IN 0.609s


# ------------------------------- CODE CELL nb-3 -------------------------------
# store the data set in this variable
df_t1 = pd.read_csv("task1.csv")

display(df_t1.corr())

x = df_t1.x
y = df_t1.y
z = df_t1.z

plt.title("plot 1")
display(plt.scatter(x,y))

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

plt.title("plot 2")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
display(ax.scatter3D(x,y,z))

# injected by test
dump_figure()

# EXECUTED IN 0.574s
# STDOUT
#     x         y         z
#     x  1.000000  0.998571  0.006157
#     y  0.998571  1.000000  0.009444
#     z  0.006157  0.009444  1.000000
#     <matplotlib.collections.PathCollection at 0x7f02f8783c40>
#     <mpl_toolkits.mplot3d.art3d.Path3DCollection at 0x7f02e0c7c610>
#     save current figure: figures/fig_nb_3_1


# ------------------------------- CODE CELL nb-4 -------------------------------
# load data
df_nsw = pd.read_csv('NSW.csv', index_col=0)
covariates = ['age', 'black', 'hisp', 'married', 'educ', 'nodegree', 're74', 're75']

display(df_nsw.shape)
display(df_nsw.head())

# injected by test
dump_figure()

# EXECUTED IN 0.373s
# STDOUT
#     (18667, 12)
#        age  educ  black  married  ...  hisp  sample  treat  educ_cat4
#     1   42    16      0        1  ...     0       2      0          4
#     2   20    13      0        0  ...     0       2      0          3
#     3   37    12      0        1  ...     0       2      0          2
#     4   48    12      0        1  ...     0       2      0          2
#     5   51    12      0        1  ...     0       2      0          2
#     
#     [5 rows x 12 columns]
#     save current figure: figures/fig_nb_4_1


# ------------------------------- CODE CELL nb-5 -------------------------------
from sklearn.linear_model import LogisticRegression

def add_propensity_scores(df: pd.DataFrame, x_cols: List[str], treatment_col: str, propensity_col: str, clf=LogisticRegression()) -> pd.DataFrame:
    """
    :param df: pandas dataframe
    :param x_cols: list of strings that represent the names of the column of df that are used as the covariates in our analysis
    :param treatment_col: name of column in df that represents the treatment variable
    :param propensity_col: string, name of the column of propensity scores that is to be added
    :param clf: sklearn classifier that should be used to produce the propensity scores. You may assume that it contains a fit() and
        predict_proba() function. Defaults to logistic regression classifier with standard parameters
    :return: input dataframe to which a column of the resulting propensity scores has been added
    """
    #step one: filter x_cols out of df
    prop_df = df[x_cols]
    #use clf to estimate prop-score
    clf.fit(prop_df, df[treatment_col])
    predict_score = clf.predict_proba(prop_df)

    df[propensity_col]=predict_score[:,1]
    return df

# injected by test
dump_figure()

# EXECUTED IN 0.00178s


# ------------------------------- CODE CELL nb-6 -------------------------------
df_nsw = add_propensity_scores(df_nsw, covariates, 'treat', 'ps')
display(df_nsw['ps'])

# injected by test
dump_figure()

# EXECUTED IN 0.153s
# STDOUT
#     1        0.000779
#     2        0.005218
#     3        0.000004
#     4        0.000004
#     5        0.000002
#                ...   
#     18663    0.400349
#     18664    0.143756
#     18665    0.149426
#     18666    0.218331
#     18667    0.670564
#     Name: ps, Length: 18667, dtype: float64


# ------------------------------- CODE CELL nb-7 -------------------------------
def greedy_matching(df: pd.DataFrame, propensity_col: str, treatment_col: str, calliper: Optional[float] = None) -> Tuple[List[int], List[int]]:
   """
   :param df: pandas dataframe
   :param propensity_col: string, name of the column in df that contains the propensity scores
   :param treatment_col: string, name of column in df that represents the treatment variable
   :param calliper: float specifying the maximum difference in propensity scores up to which two instances can be
       matched. If not specified, no calliper is used, i.e., no restriction on the difference between propensity scores
       is enforced during matching.
   :return: two lists of integers
       - list of integers that correspond to the indices of the rows in df that are matched into the control group
       - list of integers that correspond to the indices of the rows in df that are matched into the treatment group
   """
   #get groups
   treatment_group = df[df[treatment_col]==1].sort_values(propensity_col)
   control_group = df[df[treatment_col]==0].sort_values(propensity_col)

   output_list=[[],[]]
   for ind_t, pat_t in treatment_group.iterrows():
       df_diff = abs(control_group[propensity_col]-pat_t[propensity_col])
       df_min = df_diff.min()
       ind_c = df_diff[df_diff == df_min].index[0]

       #set output list
       if abs(df_min)<calliper:
           output_list[0].append(ind_t)
           output_list[1].append(ind_c)
           control_group.drop(ind_c)
           
   return output_list

# injected by test
dump_figure()

# EXECUTED IN 0.000811s


# ------------------------------- CODE CELL nb-8 -------------------------------
pat_index = greedy_matching(df_nsw, 'ps', 'treat', calliper=0.2*df_nsw['ps'].std())
#print(df_nsw.loc[pat_index[0][:5]]['ps'],df_nsw.loc[pat_index[1][:5]]['ps'])
# Add mean with and without greedy matching
ate = np.mean(df_nsw.loc[pat_index[0]]['re78'])
mean_gcg = np.mean(df_nsw.loc[pat_index[1]]['re78'])
mean_cg = np.mean(df_nsw[df_nsw['treat']==0]['re78'])

print(ate)
print(mean_gcg)
print(mean_cg)

# injected by test
dump_figure()

# EXECUTED IN 0.612s
# STDOUT
#     6383.649716750435
#     5851.002153624659
#     15750.299950189565


# ------------------------------- CODE CELL nb-9 -------------------------------
# load the data
df_bypass = pd.read_csv('bypass.csv')
display(df_bypass.head())

display(df_bypass.corr())

df_bypass_ge_80 = df_bypass[df_bypass.age >=  80]
display(df_bypass_ge_80.corr())

# injected by test
dump_figure()

# EXECUTED IN 0.0841s
# STDOUT
#     age   severity  new       stay
#     0  65.095556  42.630644    1  22.302528
#     1  53.825666  52.046443    1  22.494925
#     2  66.458806  62.178975    1  27.901718
#     3  64.397015  51.594144    1  23.670809
#     4  85.503686  43.104949    0  31.140778
#                    age  severity       new      stay
#     age       1.000000  0.165819 -0.644676  0.748344
#     severity  0.165819  1.000000  0.035115  0.707914
#     new      -0.644676  0.035115  1.000000 -0.616616
#     stay      0.748344  0.707914 -0.616616  1.000000
#                    age  severity       new      stay
#     age       1.000000  0.222965 -0.554539  0.635884
#     severity  0.222965  1.000000  0.210684  0.802713
#     new      -0.554539  0.210684  1.000000 -0.396230
#     stay      0.635884  0.802713 -0.396230  1.000000


# ------------------------------ CODE CELL nb-10 -------------------------------
bypass_fit = sm.OLS(df_bypass.stay, df_bypass[['new','age','severity']]).fit()

display(bypass_fit.summary())

stay_mean_new = np.mean(df_bypass.loc[df_bypass.new == 1, "stay"])
stay_mean_control = np.mean(df_bypass.loc[df_bypass.new == 0, "stay"])
print(f"Mean stay time after surgery with new treatment: {stay_mean_new}")
print(f"Mean stay time after surgery with control treatment: {stay_mean_control}")

# injected by test
dump_figure()

# EXECUTED IN 0.0384s
# STDOUT
#     <class 'statsmodels.iolib.summary.Summary'>
#     """
#                                      OLS Regression Results                                
#     =======================================================================================
#     Dep. Variable:                   stay   R-squared (uncentered):                   1.000
#     Model:                            OLS   Adj. R-squared (uncentered):              1.000
#     Method:                 Least Squares   F-statistic:                          5.051e+05
#     Date:                Sun, 17 Jan 2021   Prob (F-statistic):                        0.00
#     Time:                        22:42:16   Log-Likelihood:                         -42.979
#     No. Observations:                 200   AIC:                                      91.96
#     Df Residuals:                     197   BIC:                                      101.9
#     Df Model:                           3                                                  
#     Covariance Type:            nonrobust                                                  
#     ==============================================================================
#                      coef    std err          t      P>|t|      [0.025      0.975]
#     ------------------------------------------------------------------------------
#     new           -4.7542      0.054    -88.276      0.000      -4.860      -4.648
#     age            0.2089      0.001    149.298      0.000       0.206       0.212
#     severity       0.3032      0.002    139.508      0.000       0.299       0.307
#     ==============================================================================
#     Omnibus:                        2.920   Durbin-Watson:                   2.231
#     Prob(Omnibus):                  0.232   Jarque-Bera (JB):                2.980
#     Skew:                          -0.273   Prob(JB):                        0.225
#     Kurtosis:                       2.755   Cond. No.                         220.
#     ==============================================================================
#     
#     Notes:
#     [1] R² is computed without centering (uncentered) since the model does not contain a constant.
#     [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
#     """
#     Mean stay time after surgery with new treatment: 24.660080097356847
#     Mean stay time after surgery with control treatment: 32.42658761780632


# ------------------------------ CODE CELL nb-11 -------------------------------
# Initial loading and splitting of the training data into feature data and target vector

## load the data
df_train = pd.read_csv('housing_train.csv')

## split features and target data from data frame

# dfX_train is to be used as input to preprocess -> do not alter it before passing it to preprocess()! 
dfx_train = df_train.iloc[:,1:]
y_train = df_train.iloc[:,0].to_numpy()

# injected by test
dump_figure()

# EXECUTED IN 0.0137s


# ------------------------------ CODE CELL nb-12 -------------------------------
# Display values
display(df_train.head())
print(y_train)

# Check for nan values
print(f"Has NaN vlaue in a column: {dfx_train.isnull().values.any()}")

# injected by test
dump_figure()

# EXECUTED IN 0.0898s
# STDOUT
#     Price               Type  ...  Garages           Garagetype
#     0   219000.0  Mid-terrace house  ...      3.0               Garage
#     1   185000.0       Corner house  ...      8.0  Outside parking lot
#     2   999000.0             Duplex  ...      3.0               Garage
#     3   465000.0  Mid-terrace house  ...      5.0          Parking lot
#     4  5950000.0              Villa  ...      7.0               Garage
#     
#     [5 rows x 17 columns]
#     [219000. 185000. 999000. ... 196800. 249000. 269000.]
#     Has NaN vlaue in a column: False


# ------------------------------ CODE CELL nb-13 -------------------------------
from pandas.api.types import is_numeric_dtype

def preprocess(dfx: pd.DataFrame) -> np.ndarray:
    """
    Preprocessing function
        - should be used for feature engineering

    :param dfx: pandas dataframe consisting of all the predictor colums in the data
    :return: two-dimensional numpy array
    """
    df_preprocess = dfx.copy()
    # Replace each non numerical column with one-hot encoding.
    for column in list(dfx):
        if not is_numeric_dtype(dfx[column]):
            factorized = pd.factorize(dfx[column])
            # room for showning the mapping
            df_preprocess[column] = factorized[0]
    # Convert toa two-dimensional numpy array.
    return df_preprocess.to_numpy()

# injected by test
dump_figure()

# EXECUTED IN 0.000515s


# ------------------------------ CODE CELL nb-14 -------------------------------
display(df_train.head())
display(preprocess(df_train))

# injected by test
dump_figure()

# EXECUTED IN 0.103s
# STDOUT
#     Price               Type  ...  Garages           Garagetype
#     0   219000.0  Mid-terrace house  ...      3.0               Garage
#     1   185000.0       Corner house  ...      8.0  Outside parking lot
#     2   999000.0             Duplex  ...      3.0               Garage
#     3   465000.0  Mid-terrace house  ...      5.0          Parking lot
#     4  5950000.0              Villa  ...      7.0               Garage
#     
#     [5 rows x 17 columns]
#     array([[2.1900e+05, 0.0000e+00, 1.0500e+02, ..., 0.0000e+00, 3.0000e+00,
#             0.0000e+00],
#            [1.8500e+05, 1.0000e+00, 1.5600e+02, ..., 1.0000e+00, 8.0000e+00,
#             1.0000e+00],
#            [9.9900e+05, 2.0000e+00, 2.0000e+02, ..., 2.0000e+00, 3.0000e+00,
#             0.0000e+00],
#            ...,
#            [1.9680e+05, 0.0000e+00, 1.6181e+02, ..., 5.0000e+00, 2.0000e+00,
#             0.0000e+00],
#            [2.4900e+05, 0.0000e+00, 1.2932e+02, ..., 5.0000e+00, 2.0000e+00,
#             0.0000e+00],
#            [2.6900e+05, 1.0000e+00, 2.2000e+02, ..., 2.0000e+00, 7.0000e+00,
#             2.0000e+00]])


# ------------------------------ CODE CELL nb-15 -------------------------------
from sklearn.preprocessing import StandardScaler
def scale(x: np.ndarray, x_train: np.ndarray) -> np.ndarray:
    """
    Scaling function
        - should be used to scale input data according to a scaling model that is calibrated on training data
        - Note that the scaling function may also perform other things than scaling, e .g. adding a constant column or
          adding interactions

    :param x: 2D numpy array that is to be scaled
    :param x_train: 2D numpy array of training data that is used to calibrate the scaling
    :return: 2D numpy array
    """    
    s = StandardScaler(with_mean = False)
    s.fit(x_train)
    return s.transform(x)

# injected by test
dump_figure()

# EXECUTED IN 0.000524s


# ------------------------------ CODE CELL nb-16 -------------------------------
scale_temp = scale(preprocess(dfx_train), preprocess(dfx_train))
display(scale_temp)

# injected by test
dump_figure()

# EXECUTED IN 0.028s
# STDOUT
#     array([[0.        , 0.83350338, 0.32773264, ..., 0.        , 1.03340775,
#             0.        ],
#            [0.32296153, 1.23834789, 3.22571501, ..., 0.26079493, 2.755754  ,
#             0.89361844],
#            [0.64592305, 1.58762549, 0.13010384, ..., 0.52158986, 1.03340775,
#             0.        ],
#            ...,
#            [0.        , 1.28446841, 0.32235645, ..., 1.30397465, 0.6889385 ,
#             0.        ],
#            [0.        , 1.02655864, 0.04817068, ..., 1.30397465, 0.6889385 ,
#             0.        ],
#            [0.32296153, 1.74638804, 0.55073708, ..., 0.52158986, 2.41128475,
#             1.78723689]])


# ------------------------------ CODE CELL nb-17 -------------------------------
# Function used to evaluate your model, for illustration. Test data will have the same format as the training data.
# Feel free to use it for your own tests  

from sklearn.metrics import r2_score

def eval_model(df_train, df_test, beta):

    # split predictor dataframe from complete data
    dfx_train_ = df_train.iloc[:,1:]
    dfx_test_ = df_test.iloc[:,1:]

    # preprocess training and test data - preprocessed training data is always needed for scaling
    x_train = preprocess(dfx_train_)
    x_test = preprocess(dfx_test_)

    # finally, scale your data into a proper format. Note that for scaling the training data,
    # you should call 'x_train_scaled = scale(x_train, x_train)'
    x_test_scaled = scale(x_test, x_train)

    # apply your vector to predict on the test data
    y_pred = np.dot(x_test_scaled, beta)

    # get target column from test data and compute MSE
    y_test = df_test.iloc[:,0].to_numpy()
    
    return r2_score(y_test,y_pred)

# injected by test
dump_figure()

# EXECUTED IN 0.000552s


# ------------------------------ CODE CELL nb-18 -------------------------------
# General featrure check
display(dfx_train.corrwith(df_train.iloc[:,0], axis=0).abs().sort_values(kind="quicksort", ascending=False))

# injected by test
dump_figure()

# EXECUTED IN 0.0144s
# STDOUT
#     Living_space      0.478651
#     Bathrooms         0.280130
#     Floors            0.197549
#     Bedrooms          0.190110
#     Rooms             0.189604
#     Year_built        0.156102
#     Garages           0.139410
#     Lot               0.139173
#     Year_renovated    0.054846
#     dtype: float64


# ------------------------------ CODE CELL nb-19 -------------------------------
from sklearn.model_selection import train_test_split

x_train_a, x_test_a = train_test_split(df_train, test_size=0.2)

# injected by test
dump_figure()

# EXECUTED IN 0.00229s


# ------------------------------ CODE CELL nb-20 -------------------------------
# Reference from https://machinelearningmastery.com/ridge-regression-with-python/

from numpy import arange
from pandas import read_csv
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold

X = preprocess(x_train_a.iloc[:,1:])
y = x_train_a.iloc[:,0].to_numpy()

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define model
model = RidgeCV(alphas=np.linspace(1000, 1200, num=100), cv=cv, scoring='r2')
# fit model
model.fit(X, y)
# summarize chosen configuration
print(f"alpha: {model.alpha_}")
print(f"coef: {model.coef_}")
print(f"score: {model.score(X, y)}")

beta_houses = model.coef_

# injected by test
dump_figure()

# EXECUTED IN 15.9s
# STDOUT
#     alpha: 1000.0
#     coef: [ 1.67212032e+04  3.84032027e+03  1.08901856e+01 -5.25784841e+04
#      -6.34361008e+03  1.43455478e+04  2.59593550e+04  1.68684885e+03
#      -8.05941306e+04 -4.27522523e+02  8.27691353e+03 -2.38092921e+04
#       6.31602488e+03 -2.38427118e+04 -2.43704449e+04  1.33145326e+04]
#     score: 0.39087167321934513


# ------------------------------ CODE CELL nb-21 -------------------------------
display(f"R^2 = {eval_model(x_train_a, x_test_a, model.coef_)}")

# injected by test
dump_figure()

# EXECUTED IN 0.0315s
# STDOUT
#     'R^2 = -1.6338220237963852'


# ------------------------------ CODE CELL nb-22 -------------------------------
# injected by test
dump_figure()

# EXECUTED IN 0.000544s


# ------------------------ CODE CELL injected: teardown ------------------------
__IA_FLAG__ = True

# EXECUTED IN 0.000243s


