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

# EXECUTED IN 0.312s
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

# EXECUTED IN 0.00213s


# ------------------------------- CODE CELL nb-2 -------------------------------
# a list of packages that can be used to solve this assignment
# you may load additional packages (cf instructions above)
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# injected by test
dump_figure()

# EXECUTED IN 1.32s


# ------------------------------- CODE CELL nb-3 -------------------------------
##Nayoung

# qog has basic, standard, oecd data in cross sectional(cs) or timeseries(ts)
# we do not use oecd data since it only has 36 countries excluding many african nations
# we only use jan20 data

df_qog_bas = pd.read_csv("qog_bas_cs_jan20.csv", na_values="NaN")
df_reg = pd.read_csv("all.csv")

df1 = df_qog_bas[['cname', 'wdi_lifexp']]

df2 = df_reg[['name', 'region']]
df2.rename(columns={'name':'cname'}, inplace=True)

df_lifexp = pd.merge(df1, df2, how='left', on='cname')

le_af = df_lifexp[df_lifexp['region'] == 'Africa'].wdi_lifexp.dropna().to_numpy(dtype=float)
le_eu = df_lifexp[df_lifexp['region'] == 'Europe'].wdi_lifexp.dropna().to_numpy(dtype=float)

# injected by test
dump_figure()

# EXECUTED IN 0.11s
# STDERR
#     /usr/local/lib/python3.9/site-packages/pandas/core/frame.py:4300: SettingWithCopyWarning: 
#     A value is trying to be set on a copy of a slice from a DataFrame
#     
#     See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
#       return super().rename(


# ------------------------------- CODE CELL nb-4 -------------------------------
##Nayoung

# INPUT VALUES:
# v1: numpy array representing life expectancies in Europe (could also be any other data array)
# v2: numpy array representing life expectancies in Africa (could also be any other data array)
#
# RETURN:
# resulting value of test statistic, as float

def my_statistic(v1, v2):
    # your code here
    mean_diff = np.mean(v1) - np.mean(v2)
    return mean_diff

# injected by test
dump_figure()

# EXECUTED IN 0.000739s


# ------------------------------- CODE CELL nb-5 -------------------------------
my_statistic(le_eu, le_af)

# injected by test
dump_figure()

# EXECUTED IN 0.00069s


# ------------------------------- CODE CELL nb-6 -------------------------------
##Esther, Laurin

choice_divider = 30

def my_test(v1, v2, stat_func, num_sim):
    count = 0
    # Calculate how many elements to take each run.
    # Determined by the smallest set and ensure that atleast 1 element is taken.
    choice_size = max(int(min(len(v1), min(v2) / choice_divider)), 1)
    for i in range(num_sim):
        r1 = np.random.choice(v1, size=choice_size, replace=False)
        r2 = np.random.choice(v2, size=choice_size, replace=False)        
        
        if stat_func(r1, r2) < 0:
            count += 1
    return count/num_sim

# injected by test
dump_figure()

# EXECUTED IN 0.000602s


# ------------------------------- CODE CELL nb-7 -------------------------------
p_val = my_test(le_eu, le_af, my_statistic, 10000)
print(p_val)

# injected by test
dump_figure()

# EXECUTED IN 1.14s
# STDOUT
#     0.0187


# ------------------------------- CODE CELL nb-8 -------------------------------
sns.boxplot(x="region", y="wdi_lifexp", hue="region", data=df_lifexp, dodge=False)

plt.savefig('life_expectancy_plot.png')
plt.legend
plt.legend([],[], frameon=False)

# injected by test
dump_figure()

# EXECUTED IN 0.855s
# STDOUT
#     save current figure: figures/fig_nb_8_1


# ------------------------------- CODE CELL nb-9 -------------------------------
from pandas.api.types import is_numeric_dtype
import math

#df_vdem_corr = df_qog_bas.corrwith(df_qog_bas.vdem_corr, axis=0, drop=False, method='pearson')
#df_vdem_corr.dropna().abs().sort_values(ascending = False)

df_vdem_corr = df_qog_bas[df_qog_bas["vdem_corr"].notna()]

correlation_threshold = 0.85

# Columns to check
columns = list(df_vdem_corr)
columns.remove("vdem_corr")

correlations = {}

for column in columns:    
    # Drop rows that are unusable
    df_column = df_vdem_corr[df_vdem_corr[column].notna()]    
    
    # Only check numeric
    if is_numeric_dtype(df_vdem_corr[column]):
        correlation = df_column[column].corr(df_column["vdem_corr"], method="pearson")
        if not math.isnan(float(correlation)) and abs(correlation) > correlation_threshold:
            correlations[column] = correlation

correlations = sorted(correlations.items(), key = lambda item: abs(item[1]), reverse=True)
            
for (key, value) in correlations:
    print(f"Column: {key}, Value: {value}")

#print()    
#print(f"Selecting: {correlations[0][0]} with value {correlations[0][1]}")
        
corr_index = "wbgi_rle"
data_without_corr = pd.notnull((df_vdem_corr[corr_index]))
plot_data = df_vdem_corr[data_without_corr]

plt.scatter(plot_data['vdem_corr'], plot_data[corr_index])
plt.xlabel("Political Corruption Index (vdem_corr)")
plt.ylabel("Rule of Law, Estimate (wbgi_rle)")
plt.title("Correlation")

# Save first because of jupyter
plt.savefig("corruption_plot.png")
plt.show()

# injected by test
dump_figure()

# EXECUTED IN 1.27s
# STDOUT
#     Column: wbgi_cce, Value: -0.9082863406986007
#     Column: ti_cpi, Value: -0.8943329614676324
#     Column: icrg_qog, Value: -0.8588773500244783
#     Column: wbgi_rle, Value: -0.8566574008051194
#     save current figure: figures/fig_nb_9_1


# ------------------------------ CODE CELL nb-10 -------------------------------
# injected by test
dump_figure()

# EXECUTED IN 0.000626s


# ------------------------ CODE CELL injected: teardown ------------------------
__IA_FLAG__ = True

# EXECUTED IN 0.000321s


