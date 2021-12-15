# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cvx
from scipy.optimize import minimize

# plot styling
plt.style.use('seaborn-darkgrid')  # style
plt.rcParams['font.family'] = ' DIN Alternate'  # for macintosh use
# plt.rcParams['font.family'] =  'HP Simplified Jpan' # for windows use
GLOBAL_FONT_SIZE = 13
plt.rcParams.update({'font.size': GLOBAL_FONT_SIZE})  # font size
plt_dpi = None  # resolution

# %%
def visualize_stocks(data, task=None, ef_params={}, mv_params={}, sharpe_params={}):
    """Plot method for mean-volatility diagrams"""
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=data, hue='Stock', x='Volatility', y='Mean', s=100, zorder=10)

    plt.xlim([0, 8])
    plt.ylim([99, 102])

    ms = 150
    if mv_params != {}:
        plt.scatter(
            mv_params['vol'],
            mv_params['mean'],
            s=ms,
            marker='*',
            label='Minimum variance\nportfolio',
            color='red',
            zorder=10,
        )

    if sharpe_params != {}:
        plt.scatter(
            sharpe_params['vol'],
            sharpe_params['mean'],
            s=ms,
            marker='*',
            label='Sharpe\nportfolio',
            color='purple',
            zorder=10,
        )

    if ef_params != {}:
        plt.plot(
            ef_params['vol'],
            ef_params['mean'],
            label='Efficient frontier\n(w/ shorting)',
        )

    if 'vol_ns' in ef_params:
        plt.plot(
            ef_params['vol_ns'],
            ef_params['mean_ns'],
            label='Efficient frontier\n(w/o shorting)',
        )

    plt.legend(title='STOCKS', bbox_to_anchor=(1.0, 1.0), ncol=2)

    plt.tight_layout()
    if task != None:
        plt.savefig(f'plots_CA4/{task}.png', dpi=plt_dpi, bbox_inches='tight')
    plt.show()


# %% [markdown]
# ## Computer assignment 4
# # Portfolio selection

# %% [markdown]
# ## Task 1

# %%
# read data
prices = pd.read_csv('data/Dow_SP500_9620_weekly.csv', sep=';', index_col='Date')
prices.head()

# %%
# compute gross returns
pct_change = prices.pct_change().dropna()
gross_returns = 1 + pct_change
gross_returns.head()

# %% [markdown]
# ### Compute moving average estimates

# %%
win_size = 100
rolling_mean_gross = gross_returns.rolling(window=win_size).mean()
rolling_mean_gross.columns = gross_returns.columns

# %%
win_size = 100
rolling_cov_gross = gross_returns.rolling(window=win_size).cov()
rolling_cov_gross.columns = gross_returns.columns

# %% [markdown]
# ### Extract data for 101st week

# %%
week = 99
mean_w100 = rolling_mean_gross.iloc[week]
date = mean_w100.name  # date in week 101

cov_w100 = rolling_cov_gross.loc[date]
std_w100 = np.sqrt(np.diag(cov_w100))

# %% [markdown]
# ### 1a) Visualize stocks in Mean-Volatility diagram

# %%
# construct mean-volatilty table
v0 = 100
res = {
    'Stock': gross_returns.columns,
    'Mean': mean_w100 * v0,
    'Volatility': std_w100 * v0,
}
mean_vol = pd.DataFrame(res).set_index('Stock')

# %%
# visualize stocks
# visualize_stocks(mean_vol.iloc[:-1])

# %% [markdown]
# ### 1b) Visualize the efficient frontiers

# %%
# initial investment
v0 = 100

# number of stocks
n = mean_w100.shape[0] - 1

# compute mean vector
mu = mean_w100.values[:-1].reshape((n, 1))

# compute covariance
cov = cov_w100.values[:-1, :-1]

# compute volatilities
vol = np.diag(cov)

# compute inverse covariance
inv_cov = np.linalg.inv(cov)

# define unit vector
unit_vec = np.ones((n, 1))

# compute minimum variance portfolio
w_MV = (inv_cov @ unit_vec) / (unit_vec.T @ inv_cov @ unit_vec)
e_MV = (w_MV.T @ mu) * v0
vol_MV = np.sqrt((w_MV.T @ cov @ w_MV)) * v0

# %%
# minimum-varaince portfolio
mv_params = {'vol': vol_MV, 'mean': e_MV}

# %%
def get_efficient_frontier(mu, cov, e_0, v0):
    n = cov.shape[0]
    unit = np.ones((n, 1))
    inv_cov = np.linalg.inv(cov)
    A = unit.T @ inv_cov @ unit
    B = unit.T @ inv_cov @ mu
    C = mu.T @ inv_cov @ mu
    D = A * C - np.square(B)
    lbd_1 = (A * e_0 - B * v0) / D
    lbd_2 = (C * v0 - B * e_0) / D
    w = lbd_1 * (inv_cov @ mu) + lbd_2 * (inv_cov @ unit)
    w = w.reshape((e_0.shape[0], n))
    mean = w @ mu
    vol = np.sqrt(np.diag(w @ cov @ w.T))
    return vol, mean


# %%
# compute efficient frontier
e_0 = np.linspace(e_MV - 2, e_MV + 2, 250)
vol_ef, mean_ef = get_efficient_frontier(mu, cov, e_0, v0)
ef_params = {'vol': vol_ef, 'mean': mean_ef}

# %%
# visualize_stocks(mean_vol[:-1], ef_params=ef_params)

# %% [markdown]
# ### 1c) Compute Sharpe portfolio

# %% [markdown]
# #### Shorting allowed

# %%
# compute optimal allocation
w_SR = (inv_cov @ (mu - unit_vec)) / (unit_vec.T @ inv_cov @ (mu - unit_vec))

# %%
# compute expectation and volatility of Shaprpe portfolio
e_SR = (w_SR.T @ mu) * v0
vol_SR = np.sqrt(w_SR.T @ cov @ w_SR) * v0
sharpe_params = {'mean': e_SR, 'vol': vol_SR}

# %%
# visualize_stocks(
#     mean_vol[:-1],
#     ef_params=ef_params,
#     sharpe_params=sharpe_params
# )

# %% [markdown]
# #### Shorting NOT allowed

# %%
# initial investment
v0 = 100

# number of stocks
n = mean_w100.shape[0] - 1

# compute mean vector
mu = mean_w100.values[:-1].reshape((n, 1))

# compute covariance
cov = cov_w100.values[:-1, :-1]

# compute volatilities
vol = np.diag(cov)

# compute inverse covariance
inv_cov = np.linalg.inv(cov)

# define unit vector
unit = np.ones((n, 1))

# %%
# find efficient frontier without shorting
e_0 = np.linspace(99, 102, 500)
N = e_0.shape[0]
w_mat = np.zeros((N, n))
for i in np.arange(N):
    # CONVEX OPTIMIZATION PROBLEM
    w = cvx.Variable(n)
    objective_func = cvx.quad_form(w, cov)
    constraints = [
        w @ unit == v0,  # fully invested
        w @ mu == e_0[i],  # varying returns
        w >= 0,  # no short positions
    ]
    minimize = cvx.Minimize(objective_func)
    problem = cvx.Problem(minimize, constraints)
    problem.solve()
    w_mat[i] = w.value

# %%
# compute mean and volatility without shorting
mean_ns = w_mat @ mu
vol_ns = np.sqrt(np.diag(w_mat @ cov @ w_mat.T))
ef_params['vol_ns'] = vol_ns
ef_params['mean_ns'] = mean_ns

# %%
visualize_stocks(
    data=mean_vol[:-1],
    task='1c',
    ef_params=ef_params,
    mv_params=mv_params,
    sharpe_params=sharpe_params,
)

# %% [markdown]
# ### 1d) Plot the Sharpe allocation

# %%
sharpe_tab = {'Stock': mean_vol.index[:-1], 'Allocation': w_SR.flatten() * v0}
sharpe_allocation = pd.DataFrame(sharpe_tab).set_index('Stock')

# %%
sharpe_allocation.plot.bar(figsize=(10, 6), color='green', alpha=0.7)
plt.legend(labels=['Sharpe\nallocation'], frameon=1, facecolor='white')
plt.ylabel('Monetary Unit')
plt.tight_layout()
plt.savefig('plots_CA4/1d.png', dpi=plt_dpi, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 1e) Plot the gross return correlation as heatmap

# %%
# compute correlation matric
# win_size = 100
# rolling_corr_gross = gross_returns.rolling(window=win_size).corr()
# rolling_corr_gross.columns = gross_returns.columns
# corr_w100 = rolling_corr_gross.loc[date]
# corr = corr_w100.iloc[:-1, :-1]
corr = gross_returns.iloc[:-1, :-1].corr()

# %%
_, ax = plt.subplots(figsize=(12, 8))
plt.rcParams['font.size'] = 10  # decrease font size
ax = sns.heatmap(
    corr,
    vmin=-1,
    vmax=1,
    cmap='Greens',
    annot=True,
    fmt='.2f',
    linewidths=0.5,
    ax=ax,
    annot_kws={'color': 'white'},
)
ax.xaxis.set_tick_params(labelsize=11)  # change tick label sizes
ax.yaxis.set_tick_params(labelsize=11)  # change tick label sizes
plt.savefig('plots_CA4/1e.png', dpi=plt_dpi, bbox_inches='tight')
plt.tight_layout()
plt.rcParams['font.size'] = GLOBAL_FONT_SIZE  # reset font size

# %% [markdown]
# ## Task 2

# %%
def get_allocation(mu, cov, mu_0, v):
    n = cov.shape[0]
    unit = np.ones((n, 1))
    inv_cov = np.linalg.inv(cov)
    A = unit.T @ inv_cov @ unit
    B = unit.T @ inv_cov @ mu
    C = mu.T @ inv_cov @ mu
    D = A * C - np.square(B)
    lbd_1 = (A * mu_0 * v - B * v) / D
    lbd_2 = (C * v - B * mu_0 * v) / D
    w = lbd_1 * (inv_cov @ mu) + lbd_2 * (inv_cov @ unit)
    return w


# %%
def get_allocation_NS(mu, cov, mu_0, v):
    """Target portfolio allocation - no shorting"""
    n = cov.shape[0]
    unit = np.ones((n, 1))
    w = cvx.Variable(shape=(n, 1), name='w')
    objective_func = cvx.quad_form(w, cov)
    constraints = [
        w.T @ unit == v,  # fully invested
        w.T @ mu == v * mu_0,  # varying target return
        w >= 0.0,  # no short positions
    ]
    minimize = cvx.Minimize(objective_func)
    problem = cvx.Problem(minimize, constraints)
    problem.solve()

    # if no solution is found
    if w.value is None:
        mu = mu.flatten()
        if mu_0 <= np.mean(mu):
            idx = mu == np.min(mu)
        else:
            idx = mu == np.max(mu)
        w = np.zeros(n)
        w[idx] = v
        return w
    return w.value


# %%
def get_ENC(w):
    w = w / np.sum(w)
    return 1 / np.sum(np.square(w))


# %% [markdown]
# ### Backtesting

# %%
# backtesting rolling means
week = 99
mu_bt = rolling_mean_gross.iloc[week:]
mu_bt = mu_bt.values.reshape((mu_bt.shape[0], mu_bt.shape[1], 1))
mu_bt = mu_bt[:, :-1]

# backtesting gross returns
returns_bt = gross_returns.iloc[week:]
returns_bt = returns_bt.values.reshape((returns_bt.shape[0], returns_bt.shape[1], 1))
returns_bt = returns_bt[:, :-1]

# backtesting rolling covariance
date = rolling_mean_gross.iloc[week].name
cov_bt = rolling_cov_gross.loc[date:]
cov_bt = cov_bt.values.reshape(
    (int(cov_bt.shape[0] / cov_bt.shape[1]), cov_bt.shape[1], cov_bt.shape[1])
)
cov_bt = cov_bt[:, :-1, :-1]

# %%
# backtesting rolling means
week = 99
mu_bt = rolling_mean_gross.iloc[week:]
mu_bt = mu_bt.values.reshape((mu_bt.shape[0], mu_bt.shape[1], 1))
mu_bt = mu_bt[:, :-1]

# backtesting gross returns
returns_bt = gross_returns.iloc[week:]
returns_bt = returns_bt.values.reshape((returns_bt.shape[0], returns_bt.shape[1], 1))
returns_bt = returns_bt[:, :-1]

# backtesting rolling covariance
date = rolling_mean_gross.iloc[week].name
cov_bt = rolling_cov_gross.loc[date:]
cov_bt = cov_bt.values.reshape(
    (int(cov_bt.shape[0] / cov_bt.shape[1]), cov_bt.shape[1], cov_bt.shape[1])
)
cov_bt = cov_bt[:, :-1, :-1]

# %% [markdown]
# #### Shorting allowed

# %%
# shorting allowed
T = cov_bt.shape[0] - 1
e_short = np.zeros(T)
ENC_short = np.zeros(T)
w_short = np.zeros((T, n, 1))

v = 100.0  # initial investment
mu_0 = 1.00275
for t in np.arange(T):
    mu_bt_t = mu_bt[t]
    cov_bt_t = cov_bt[t]
    w = get_allocation(mu_bt_t, cov_bt_t, mu_0, v)
    v = (w.T @ returns_bt[t + 1]).flatten()
    e_short[t] = v
    ENC_short[t] = get_ENC(w)
    w_short[t] = w / np.sum(w)

# %% [markdown]
# #### Shorting not allowed

# %%
# no shorting allowed
T = cov_bt.shape[0] - 1
e_no_short = np.zeros(T)
ENC_no_short = np.zeros(T)
w_no_short = np.zeros((T, n))

v = 100.0  # initial investment
mu_0 = 1.00275
for t in np.arange(T):
    mu_bt_t = mu_bt[t]
    cov_bt_t = cov_bt[t]
    w = get_allocation_NS(mu_bt_t, cov_bt_t, mu_0, v)
    v = (w.T @ returns_bt[t + 1]).flatten()
    e_no_short[t] = v
    ENC_no_short[t] = get_ENC(w)
    w_no_short[t] = w.flatten() / np.sum(w)

# %% [markdown]
# #### Equal weight portfolio

# %%
# equal weight portfolio
T = cov_bt.shape[0] - 1
n = cov_bt.shape[1]
e_ew = np.zeros(T)
ENC_ew = np.zeros(T)
w_ew = np.zeros((T, n))

v = 100.0  # initial investment
w = np.ones((1, n)) * (v / n)
for t in np.arange(T):
    v = w @ returns_bt[t + 1]
    e_ew[t] = v
    w = np.ones((1, n)) * (v / n)
    w_ew[t] = w.flatten() / np.sum(w)
    ENC_ew[t] = get_ENC(w)

# %% [markdown]
# ### 2a) Plot portfolio evolution

# %%
res = {
    'Week': np.arange(T),
    'Target portfolio\n(w/ shorting)': e_short,
    'Target portfolio\n(w/o shorting)': e_no_short,
    'Equal weight portfolio': e_ew,
}
res_2a = pd.DataFrame(res).set_index('Week')

# %%
res_2a.plot(figsize=(10, 6))
plt.ylabel('Portfolio Value')
plt.legend(frameon=1, facecolor='white', title='Allocation')
plt.tight_layout()
plt.savefig('plots_CA4/2a.png', dpi=plt_dpi, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 2b) Create table of portfolio values beginning at every 100th week (start at week 0)

# %%
idx = res_2a.index % 100 == 0
res_2b = res_2a.iloc[idx].round(2)
res_2b

# %%
# for report use - 2b)
res_2b_latex = res_2b.to_latex(position='H', label='tab:2b', caption='Caption')
print(res_2b_latex)

# %% [markdown]
# ### 2c) Calculate annualized means, volatilities, Sharpe ratios, maximum drawdown and average ENCs for the portfolios’ weekly percentage returns over the backtesting period

# %%
def get_sharpe(e):
    R = (e[1:] / e[:-1]) - 1
    return np.sqrt(52.0) * np.mean(R) / np.std(R)


# %%
def get_annualized_mean(e):
    returns = (e[1:] / e[:-1]) - 1
    return np.mean(returns) * 52.0


# %%
def get_annualized_volatility(e):
    returns = (e[1:] / e[:-1]) - 1
    return np.std(returns) * np.sqrt(52.0)


# %%
def get_annualized_ENC(ENC):
    return np.mean(ENC)


# %%
def get_MD(e):
    CM = np.maximum.accumulate(e)
    DD = CM - e
    return np.max(DD / CM)


# %%
# define portfolio lists
portfolio_labels = [
    'Target Portfolio (w/ shorting)',
    'Target Portfolio (w/o shorting)',
    'Equal Weight Portfolio',
]
portfolios = [e_short, e_no_short, e_ew]
ENCs = [ENC_short, ENC_no_short, ENC_ew]

# create dict and df
res = {
    'Portfolio': portfolio_labels,
    'Mean': [get_annualized_mean(e) for e in portfolios],
    'Volatility': [get_annualized_volatility(e) for e in portfolios],
    'Sharpe Ratio': [get_sharpe(e) for e in portfolios],
    'MDD': [get_MD(e) for e in portfolios],
    'Average ENC': [get_annualized_ENC(ENC) for ENC in ENCs],
}
df_2c = pd.DataFrame(res).set_index('Portfolio').round(4)

# print results
df_2c

# %%
# for report use - 2c)
caption = 'Caption'
res_2c = df_2c.to_latex(position='H', caption=caption, label='tab:2c')
print(res_2c)

# %% [markdown]
# ### 2d) Plot the portfolio percentage weights for the no short target portfolio using the area plot

# %%
w_df = pd.DataFrame(w_no_short.reshape((T, n)))
w_df.columns = gross_returns.columns[:-1]
w_df[w_df < 0.0] = 0.0  # remove zeros
w_df.plot.area(
    figsize=(11, 7), stacked=True, lw=0, color=sns.color_palette('ocean', n_colors=n)
)
plt.legend(title='STOCKS', bbox_to_anchor=(1.0, 1.0), fontsize=11.5)
plt.ylabel('Allocation (%)')
plt.xlabel('Week')
plt.title('Target Portfolio (w/o shorting)')
plt.tight_layout()
plt.savefig('plots_CA4/2d.png', dpi=plt_dpi, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Task 3

# %%
def get_w(mu, inv_cov, mu0, R0, v0, w_h):
    n = inv_cov.shape[0]
    unit = np.ones((n, 1))
    upper = (mu0 - R0) * v0 - w_h.T @ (mu - (R0 * unit))
    lower = (mu - (R0 * unit)).T @ inv_cov @ (mu - (R0 * unit))
    w = upper / lower * (inv_cov @ (mu - (R0 * unit))) + w_h
    return w


# %%
def get_tracking_error(var_index, var_port, sigma_LR, w_port):
    return var_port - 2 * w_port.T @ sigma_LR + var_index


# %%
def get_optimal_portfolio(mu, cov, mu0, R0, sigma_LR, v0):
    inv_cov = np.linalg.inv(cov)
    w_h = inv_cov @ sigma_LR
    w = get_w(mu, inv_cov, mu0, R0, v0, w_h)
    return w, w_h


# %%
def get_efficient_frontier_mean_track(mu, cov, R0, sigma_LR, v0, var_index):
    mu0 = np.linspace(99, 102, 1000) / 100.0
    N = mu0.shape[0]
    n = cov.shape[0]
    W = np.zeros((N, n, 1))
    E, vol, error = np.zeros((N, 1)), np.zeros((N, 1)), np.zeros((N, 1))
    inv_cov = np.linalg.inv(cov)
    w_h = inv_cov @ sigma_LR
    for i in np.arange(N):
        W[i] = get_w(mu, inv_cov, mu0[i], R0, v0, w_h)
        E[i] = W[i].T @ mu
        vol[i] = np.sqrt(np.diag(W[i].T @ cov @ W[i]))
        error[i] = get_tracking_error(var_index, np.square(vol[i]), sigma_LR, W[i])
    return vol, E, W, error


# %%
# fetch data
R0 = 1.0
L0, v0 = 100.0, 100.0

# parameter estimates
last_returns = rolling_mean_gross.iloc[-1]
last_date = last_returns.name
last_cov = rolling_cov_gross.loc[last_date]

mu0 = last_returns[-1]  # index return
mu_vec = last_returns[:-1].values.reshape((26, 1))  # stock returns

sigma_LR = (
    last_cov.iloc[-1, :-1].values.reshape((26, 1)) * L0
)  # covariance between index and each stock
sigma = last_cov.iloc[:-1, :-1].values  # covariance matrix for stocks
var_SP500 = last_cov.iloc[-1, -1] * np.square(L0)

# %%
# backtesting rolling means
week = 99
mu_bt = rolling_mean_gross.iloc[week:]
mu_bt = mu_bt.values.reshape((mu_bt.shape[0], mu_bt.shape[1], 1))
mu_bt = mu_bt[:, :-1]

# backtesting gross returns
returns_bt = gross_returns.iloc[week:]
returns_bt = returns_bt.values.reshape((returns_bt.shape[0], returns_bt.shape[1], 1))
returns_bt = returns_bt[:, :-1]

# backtesting rolling covariance
date = rolling_mean_gross.iloc[week].name
cov_bt = rolling_cov_gross.loc[date:]
cov_bt = cov_bt.values.reshape(
    (int(cov_bt.shape[0] / cov_bt.shape[1]), cov_bt.shape[1], cov_bt.shape[1])
)
cov_bt = cov_bt[:, :-1, :-1]

# %%
last_returns.index

# %% [markdown]
# ### 3a) Plot the optimal hedge portfolio allocation as a bar plot and include the values in a table

# %%
_, w_h = get_optimal_portfolio(mu_vec, sigma, mu0, R0, sigma_LR, v0)

# %%
res_3a = {'Stock': last_returns.index[:-1], 'Allocation': w_h.flatten()}

bm_allocation = pd.DataFrame(res_3a).set_index('Stock')

# %%
bm_allocation.shape

# %%
bm_allocation.plot.bar(figsize=(10, 6), color='green', alpha=0.7)
plt.legend(labels=['Optimal hedge\nportfolio allocation'], frameon=1, facecolor='white')
plt.ylabel('Monetary Unit')
plt.tight_layout()
plt.savefig('plots_CA4/3a.png', dpi=plt_dpi, bbox_inches='tight')
plt.show()

# %%
# for report use - 3a)
latex_3a_1 = (
    bm_allocation.iloc[:13]
    .T.round(2)
    .to_latex(position='H', caption='Caption.', label='tab:3a_1')
)
print(latex_3a_1)

latex_3a_2 = (
    bm_allocation.iloc[13:]
    .T.round(2)
    .to_latex(position='H', caption='Caption.', label='tab:3a_2')
)
print(latex_3a_2)

# %% [markdown]
# ### 3b) Plot the “mean – tracking error” efficient frontier

# %%
vol, mean, W, error = get_efficient_frontier_mean_track(
    mu_vec, sigma, R0, sigma_LR, v0, var_SP500
)

# %%
hedge_e = w_h.T @ mu_vec
hedge_vol = np.sqrt(w_h.T @ sigma @ w_h)
hedge_error = get_tracking_error(var_SP500, np.square(hedge_vol), sigma_LR, w_h)

# %%
plt.figure(figsize=(10, 6))
plt.plot(
    np.sqrt(error), mean, label='Benchmark\nallocation', color=sns.color_palette()[2]
)
plt.scatter(
    np.sqrt(hedge_error),
    hedge_e,
    marker='*',
    zorder=10,
    color='red',
    s=150,
    label='Optimal hedge\nportfolio allocation',
)
plt.ylim([99, 102])
plt.xlim([0.4, 0.5])
plt.xlabel('Tracking Error')
plt.ylabel('Mean')
plt.legend(frameon=1, facecolor='white')
plt.tight_layout()
plt.savefig('plots_CA4/3b.png', dpi=plt_dpi, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 3c) What is the smallest tracking error that can be achieved?

# %%
print('Minimum tracking error:', np.sqrt(np.min(error)).round(3))

# %% [markdown]
# ### 3d) What is the expected excess return (in excess of S&P500) of the minimum tracking error portfolio?

# %%
excess_return = (hedge_e - (mu0 * 100)) / 100
print('Expected excess return:', excess_return.flatten()[0].round(4), '%')

# %% [markdown]
# ## Task 4

# %%
def cov2cor(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


# %%
def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:, None] < r
    return A[mask]


# %%
def get_allocation_sharpe(mu, cov):
    """Sharpe portfolio allocation - no shorting"""
    n = cov.shape[0]
    w = cvx.Variable(shape=(n, 1), name='w')
    objective_func = cvx.quad_form(w, cov)
    constraints = [mu.T @ w == 1.0, w >= 0.0]
    minimize = cvx.Minimize(objective_func)
    problem = cvx.Problem(minimize, constraints)
    problem.solve()
    return w.value


# %% [markdown]
# #### Shrinkage (F = average correlation)

# %%
def shrink_cov(cov, alpha=0.3):
    vol = np.diag(np.sqrt(np.diag(cov)))
    cor = cov2cor(cov)
    mean_cor = np.mean(upper_tri_masking(cor))
    F_cor = np.eye(n)
    F_cor[F_cor == 0.0] = mean_cor
    F_cov = vol @ F_cor @ vol
    return (1.0 - alpha) * cov + alpha * F_cov


# %%
def shrink_mu(mu):
    # e = np.linalg.eigvals(cov)
    # e = np.flip(np.sort(e)) # sort descending
    # nom = np.sum(e) - 2 * e[0]
    # denom = (mu - grand_mean).T @ (mu - grand_mean)
    # alpha = (nom / denom) / T

    n = mu.shape[0]
    grand_mean = np.mean(mu) * np.ones((n, 1))
    alpha = 0.3
    return (1.0 - alpha) * mu + alpha * grand_mean


# %%
def denoise(cov, T=100.0):
    n = cov.shape[0]
    q = n / T
    e_lim = 1 + q + 2 * np.sqrt(q)
    vol = np.diag(np.sqrt(np.diag(cov)))
    cor = cov2cor(cov)
    e, v = np.linalg.eig(cor)

    # sort descending
    idx = np.flip(e.argsort())
    e = e[idx]
    v = v[:, idx]

    if np.sum(e > e_lim) == 0:
        e[1:] = np.mean(e[1:])
    else:
        e[e < e_lim] = np.mean(e[e < e_lim])

    cor_d = v @ np.diag(e) @ v.T
    cov_d = vol @ cor_d @ vol
    return cov_d


# %% [markdown]
# #### Fetch backtesting data

# %%
# backtesting rolling gross means
week = 99
mu_bt = rolling_mean_gross.iloc[week:]
mu_bt = mu_bt.values.reshape((mu_bt.shape[0], mu_bt.shape[1], 1))
mu_bt = mu_bt[:, :-1]

# backtesting gross returns
returns_bt = gross_returns.iloc[week:]
returns_bt = returns_bt.values.reshape((returns_bt.shape[0], returns_bt.shape[1], 1))
returns_bt = returns_bt[:, :-1]

# backtesting rolling gross return covariance
date = rolling_mean_gross.iloc[week].name
cov_bt = rolling_cov_gross.loc[date:]
cov_bt = cov_bt.values.reshape(
    (int(cov_bt.shape[0] / cov_bt.shape[1]), cov_bt.shape[1], cov_bt.shape[1])
)
cov_bt = cov_bt[:, :-1, :-1]

# %% [markdown]
# ### 4a)

# %%
# Sharpe - no shorting allowed
T = cov_bt.shape[0] - 1
e_sharpe = np.zeros(T)
ENC_sharpe = np.zeros(T)
w_sharpe = np.zeros((T, n))

v = 100.0  # initial investment
for t in np.arange(T):
    w = get_allocation_sharpe(mu_bt[t] - 1.0, cov_bt[t])
    w = w / np.sum(w)  # normalize
    v = v * (w.T @ returns_bt[t + 1]).flatten()
    e_sharpe[t] = v
    ENC_sharpe[t] = get_ENC(w)
    w_sharpe[t] = w.flatten()

# %%
# Sharpe - no shorting allowed + shrinkage
T = cov_bt.shape[0] - 1
e_sharpe_s = np.zeros(T)
ENC_sharpe_s = np.zeros(T)
w_sharpe_s = np.zeros((T, n))

v = 100.0  # initial investment
for t in np.arange(T):
    cov_bt_t = cov_bt[t]
    w = get_allocation_sharpe(shrink_mu(mu_bt[t]) - 1.0, shrink_cov(cov_bt_t))
    w = w / np.sum(w)  # normalize
    v = v * (w.T @ returns_bt[t + 1]).flatten()
    e_sharpe_s[t] = v
    ENC_sharpe_s[t] = get_ENC(w)
    w_sharpe_s[t] = w.flatten()

# %%
# Sharpe - no shorting allowed + shrinkage (mu) + denoising (cov)
T = cov_bt.shape[0] - 1
e_sharpe_d = np.zeros(T)
ENC_sharpe_d = np.zeros(T)
w_sharpe_d = np.zeros((T, n))

v = 100.0  # initial investment
for t in np.arange(T):
    cov_bt_t = cov_bt[t]
    w = get_allocation_sharpe(shrink_mu(mu_bt[t]) - 1.0, denoise(cov_bt_t))
    w = w / np.sum(w)  # normalize
    v = v * (w.T @ returns_bt[t + 1]).flatten()
    e_sharpe_d[t] = v
    ENC_sharpe_d[t] = get_ENC(w)
    w_sharpe_d[t] = w.flatten()

# %%
res = {
    'Week': np.arange(T),
    'Target portfolio\n(w/ shorting)': e_short,
    'Target portfolio\n(w/o shorting)': e_no_short,
    'Equal weight portfolio': e_ew,
    'Sharpe portfolio': e_sharpe,
    'Sharpe portfolio (shrinkage)': e_sharpe_s,
    'Sharpe portfolio (denoising)': e_sharpe_d,
}
res_4a = pd.DataFrame(res).set_index('Week')

# %%
res_4a.plot(figsize=(10, 6))
plt.ylabel('Portfolio Value')
plt.legend(frameon=1, facecolor='white', title='Allocation')
plt.tight_layout()
plt.savefig('plots_CA4/4a.png', dpi=plt_dpi, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 4b)

# %%
idx = res_4a.index % 100 == 0
res_4b = res_4a.iloc[idx].round(2)
res_4b

# %%
# for report use - 5b)
res_4b_latex = res_4b.to_latex(position='H', label='tab:4b', caption='Caption')
print(res_4b_latex)

# %% [markdown]
# ### 4c)

# %%
# define portfolio lists
portfolio_labels = [
    'Sharpe portfolio',
    'Sharpe portfolio (shrinkage)',
    'Sharpe portfolio (denoising)',
]
portfolios = [e_sharpe, e_sharpe_s, e_sharpe_d]
ENCs = [ENC_sharpe, ENC_sharpe_s, ENC_sharpe_d]

# create dict and df
res = {
    'Portfolio': portfolio_labels,
    'Mean': [get_annualized_mean(e) for e in portfolios],
    'Volatility': [get_annualized_volatility(e) for e in portfolios],
    'Sharpe Ratio': [get_sharpe(e) for e in portfolios],
    'MDD': [get_MD(e) for e in portfolios],
    'Average ENC': [get_annualized_ENC(ENC) for ENC in ENCs],
}
df_4c = pd.DataFrame(res).set_index('Portfolio').round(4)

# print results
df_4c

# %%
# for report use - 4c)
caption = 'Caption'
res_4c = df_4c.to_latex(position='H', caption=caption, label='tab:4c')
print(res_4c)

# %% [markdown]
# ### 4d)

# %%
w_df = pd.DataFrame(w_sharpe)
w_df.columns = gross_returns.columns[:-1]
w_df[w_df < 0.0] = 0.0  # remove zeros
w_df.plot.area(
    figsize=(11, 7), stacked=True, lw=0, color=sns.color_palette('ocean', n_colors=n)
)
plt.legend(title='STOCKS', bbox_to_anchor=(1.0, 1.0), fontsize=11.5)
plt.title('Sharpe Portfolio')
plt.ylabel('Allocation (%)')
plt.xlabel('Week')
plt.tight_layout()
plt.savefig('plots_CA4/4d_1.png', dpi=plt_dpi, bbox_inches='tight')
plt.show()

# %%
w_df = pd.DataFrame(w_sharpe_s)
w_df.columns = gross_returns.columns[:-1]
w_df[w_df < 0.0] = 0.0  # remove zeros
w_df.plot.area(
    figsize=(11, 7), stacked=True, lw=0, color=sns.color_palette('ocean', n_colors=n)
)
plt.legend(title='STOCKS', bbox_to_anchor=(1.0, 1.0), fontsize=11.5)
plt.title('Sharpe Portfolio (Shrinkage)')
plt.ylabel('Allocation (%)')
plt.xlabel('Week')
plt.tight_layout()
plt.savefig('plots_CA4/4d_2.png', dpi=plt_dpi, bbox_inches='tight')
plt.show()

# %%
w_df = pd.DataFrame(w_sharpe_d)
w_df.columns = gross_returns.columns[:-1]
w_df[w_df < 0.0] = 0.0  # remove zeros
w_df.plot.area(
    figsize=(11, 7), stacked=True, lw=0, color=sns.color_palette('ocean', n_colors=n)
)
plt.legend(title='STOCKS', bbox_to_anchor=(1.0, 1.0), fontsize=11.5)
plt.title('Sharpe Portfolio (Denoising)')
plt.ylabel('Allocation (%)')
plt.xlabel('Week')
plt.tight_layout()
plt.savefig('plots_CA4/4d_3.png', dpi=plt_dpi, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Task 5

# %%
def risk_parity(w, cov):
    cov_w = cov @ w
    var = w.T @ cov @ w
    func = w * (cov_w / var) - (1.0 / w.shape[0])
    return np.sum(np.square(func))


# %%
def get_allocation_risk_parity(cov):
    init_w = 1.0 / np.sqrt(np.diag(cov))
    res = minimize(risk_parity, x0=init_w, args=(cov), method='SLSQP')
    return res.x


# %%
def get_allocation_MV_NS(cov, v):
    """Target portfolio allocation - no shorting"""
    n = cov.shape[0]
    w = cvx.Variable(shape=(n, 1), name='w')
    objective_func = cvx.quad_form(w, cov)
    constraints = [cvx.sum(w) == v, w >= 0.0]  # fully invested  # no short positions
    minimize = cvx.Minimize(objective_func)
    problem = cvx.Problem(minimize, constraints)
    problem.solve()
    return w.value


# %% [markdown]
# ### 5a)

# %%
# MV allocation
T = cov_bt.shape[0] - 1
e_MV = np.zeros(T)
ENC_MV = np.zeros(T)
w_MV = np.zeros((T, n))

v = 100.0  # initial investment
for t in np.arange(T):
    w = get_allocation_MV_NS(cov_bt[t], v)
    v = (w.T @ returns_bt[t + 1]).flatten()
    e_MV[t] = v
    ENC_MV[t] = get_ENC(w)
    w_MV[t] = w.flatten() / np.sum(w)

# %%
# risk parity allocation
T = cov_bt.shape[0] - 1
e_rp = np.zeros(T)
ENC_rp = np.zeros(T)
w_rp = np.zeros((T, n))

v = 100.0  # initial investment
for t in np.arange(T):
    w = get_allocation_risk_parity(cov_bt[t])
    w = w / np.sum(w)  # normalize
    v = v * (w.T @ returns_bt[t + 1]).flatten()
    e_rp[t] = v
    ENC_rp[t] = get_ENC(w)
    w_rp[t] = w.flatten()

# %%
# create df for plot
res_5a = {
    'Week': np.arange(T),
    'Minimum variance portfolio': e_MV,
    'Risk parity portfolio': e_rp,
}

df_5a = pd.DataFrame(res_5a).set_index('Week')

# %%
df_5a.plot(figsize=(10, 6), color=sns.color_palette()[8:10])
plt.ylabel('Portfolio Value')
plt.legend(frameon=1, facecolor='white', title='Allocation')
plt.tight_layout()
plt.savefig('plots_CA4/5a.png', dpi=plt_dpi, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### 5b)

# %%
idx = df_5a.index % 100 == 0
tab_5b = df_5a.loc[idx].round(2)
tab_5b

# %%
# for report use - 5b)
res_5b_latex = tab_5b.to_latex(position='H', label='tab:5b', caption='Caption')
print(res_5b_latex)

# %% [markdown]
# ### 5c)

# %%
# define portfolio lists
portfolio_labels = ['Minimum variance portfolio', 'Risk parity portfolio']
portfolios = [e_MV, e_rp]
ENCs = [ENC_MV, ENC_rp]

# create dict and df
res = {
    'Portfolio': portfolio_labels,
    'Mean': [get_annualized_mean(e) for e in portfolios],
    'Volatility': [get_annualized_volatility(e) for e in portfolios],
    'Sharpe Ratio': [get_sharpe(e) for e in portfolios],
    'MDD': [get_MD(e) for e in portfolios],
    'Average ENC': [get_annualized_ENC(ENC) for ENC in ENCs],
}
df_5c = pd.DataFrame(res).set_index('Portfolio').round(4)

# print results
df_5c

# %%
# for report use - 5c)
caption = 'Caption'
res_5c = df_5c.to_latex(position='H', caption=caption, label='tab:5c')
print(res_5c)

# %% [markdown]
# ### 5d)

# %%
w_df = pd.DataFrame(w_MV)
w_df.columns = gross_returns.columns[:-1]
w_df[w_df < 0.0] = 0.0  # remove zeros
w_df.plot.area(
    figsize=(11, 7), stacked=True, lw=0, color=sns.color_palette('ocean', n_colors=n)
)
plt.legend(title='STOCKS', bbox_to_anchor=(1.0, 1.0), fontsize=11.5)
plt.title('Minimum Variance Portfolio')
plt.ylabel('Allocation (%)')
plt.xlabel('Week')
plt.tight_layout()
plt.savefig('plots_CA4/5d_1.png', dpi=plt_dpi, bbox_inches='tight')
plt.show()

# %%
w_df = pd.DataFrame(w_rp.reshape((T, n)))
w_df.columns = gross_returns.columns[:-1]
w_df[w_df < 0.0] = 0.0  # remove zeros
w_df.plot.area(
    figsize=(11, 7), stacked=True, lw=0, color=sns.color_palette('ocean', n_colors=n)
)
plt.legend(title='STOCKS', bbox_to_anchor=(1.0, 1.0), fontsize=11.5)
plt.title('Risk Parity Portfolio')
plt.ylabel('Allocation (%)')
plt.xlabel('Week')
plt.tight_layout()
plt.savefig('plots_CA4/5d_2.png', dpi=plt_dpi, bbox_inches='tight')
plt.show()
