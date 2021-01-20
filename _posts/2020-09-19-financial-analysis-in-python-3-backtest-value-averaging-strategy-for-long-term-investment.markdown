---
layout: post
title: "Financial Analysis in Python #3: Backtest Value Averaging Strategy for Long-term Investment"
date: 2020-09-19T17:16:40+02:00
tags: [Finance, Python, visualization, long-term investment, retirement]
categories: [Finance, Data Science]
---

# Overview

This is the third blog post of my **Finaical Analysis in Python** series.

This blog series covers topics like the following:

- how to **visualize** the **long-term** investment plan
- backtest **dollar-cost averaging** strategy for **long-term** investment
- backtest **value averaging** strategy for **long-term** investment
- compare different investment strategies for **short-term** investment

<!--more-->

The jupyter notebooks can be downloaded [here](https://github.com/RuoyunLin/code_snippets/tree/master/finance).

# Disclaimer

Investing money into anything is often involved with risk. Please do your own research before investing and be responsible for your own investment decisions.

I am just learning investment on my own and want to share some codes that I have written that might be useful for others.

The content here is only for informational purpose (instead of taking them as professional investment advice).

# Introduction

According to Michael E. Edleson's book, [value averaging](https://en.wikipedia.org/wiki/Value_averaging) method is an enhanced version of the dollar cost averaging method.

It performs better in a mixed market with high volatlity because it tends to secure the gains and buy more dips.

Here we are going to backtest the value averaging strategy using the SP500 index historical data.

# Prepare data and define functions


```python
# Import libraries
import datetime
import warnings

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr

plt.style.use('bmh')
warnings.filterwarnings("ignore")
```


```python
# Get historical data
startdate = datetime.datetime(1988, 1, 1)
enddate = datetime.datetime(2021, 1, 1)

# Pick SP 500 total return index
SP = pdr.get_data_yahoo('^SP500TR', start=startdate, end=enddate)
SP['date'] = SP.index
```


```python
# Group by year
SP['year'] = SP['date'].apply(lambda d: f'{d.year}')
df = SP[['year', 'Adj Close']].groupby('year').first()
df['return'] = df['Adj Close'].pct_change() # return rate for last year's investment
df['return_next_y'] = df['return'].shift(-1) # return rate by the begining of next year
df = df.reset_index()
df = df.drop(columns='return')

# You can use the following group by dataframe if you want to do value averging monthly or quarterly
# # Group by month
# SP['month'] = SP['date'].apply(lambda d: f'{d.month:02d}')
# df_m = SP[['year', 'month', 'Adj Close']].groupby(['year','month']).first()
# df_m['return'] = df_m['Adj Close'].pct_change()
# df_m['return_next'] = df_m['return'].shift(-1)
# df_m = df_m.reset_index()
# df_m = df_m.drop(columns='return')

# # Group by quarter
# SP['quarter'] = SP['month'].apply(lambda m: int(m)//4 + 1)
# df_q = SP[['year', 'quarter', 'Adj Close']].groupby(['year','quarter']).first()
# df_q['return'] = df_q['Adj Close'].pct_change()
# df_q['return_next'] = df_q['return'].shift(-1)
# df_q = df_q.reset_index()
# df_q = df_q.drop(columns='return')
```


```python
df.round(2).head(5)
```




<div>
<style>
table {
  border-collapse: collapse;
  width: 100%;
}

th, td {
  text-align: left;
  padding: 5px;
}

tr:nth-child(even) {background-color: #f2f2f2;}
</style>
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>Adj Close</th>
      <th>return_next_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1988</td>
      <td>256.02</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1989</td>
      <td>285.65</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1990</td>
      <td>386.16</td>
      <td>-0.06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1991</td>
      <td>363.44</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1992</td>
      <td>479.84</td>
      <td>0.08</td>
    </tr>
  </tbody>
</table>
</div>




```python
def calc_value_averaging_return(df: pd.DataFrame, value_per_period: float = 1.0, 
                                increase_investment_per_period: float = 0.0,
                                display_irr: bool = True) -> pd.DataFrame:
    """
    This function calculates the internal rate of return of the value averaging strategy.
    It also returns a dataframe that contains the amount of money should invest or withdraw per period.
    :param df: Original dataframe that contains the price at the beginning of each period
    :param value_per_period: Investment value per period
    :param increase_investment_per_period: Increase the investment by x each period
    :param display_irr: Whether to display IRR results or not
    :return: A dataframe that contains all the relevant info
    """
    df_va = pd.DataFrame.copy(df)
    value = value_per_period
    df_va.loc[0,'target_value'] = value
    df_va.loc[0,'shares_should_have_in_total'] = value_per_period/df_va.loc[0,'Adj Close']
    df_va.loc[0,'shares_to_buy_or_sell'] = value_per_period/df_va.loc[0,'Adj Close']
    df_va.loc[0,'should_invest_or_withdraw'] = value
    df_va.loc[0,'total_invest_or_withdraw'] = value

    for i in range(1,len(df_va.index)):
        value += value_per_period*((1 + increase_investment_per_period)**i)
        df_va.loc[i,'target_value'] = value
        df_va.loc[i,'shares_should_have_in_total'] = df_va.loc[i,'target_value']/df_va.loc[i,'Adj Close']
        df_va.loc[i,'shares_to_buy_or_sell'] = df_va.loc[i,'shares_should_have_in_total'] - df_va.loc[i-1,'shares_should_have_in_total']
        df_va.loc[i,'should_invest_or_withdraw'] = df_va.loc[i,'shares_to_buy_or_sell']*df_va.loc[i,'Adj Close']
        df_va.loc[i,'total_invest_or_withdraw'] = df_va.loc[i-1,'total_invest_or_withdraw'] + df_va.loc[i,'should_invest_or_withdraw']

    if display_irr:
        cf = (-df_va['should_invest_or_withdraw']).tolist()
        cf[-1] = (df_va.loc[len(df_va.index)-2,'shares_should_have_in_total'] *
                    df_va.loc[len(df_va.index)-1,'Adj Close'])
        print("The cash flow (the negative sign stands for investment):")
        display(cf)
        irr = round(100*np.irr(cf), 2)
        print(f"The internal return rate is {irr}% per period")
        
    return df_va

```


```python
def calc_buy_only_value_averaging_return(df: pd.DataFrame, value_per_period: float = 1.0, 
                                         increase_investment_per_period: float = 0.0,
                                         min_investment_per_period: float = 0.0,
                                         max_investment_per_period: float = 5.0,
                                         display_returns: bool = True) -> pd.DataFrame:
    """
    This function calculates the overall return rate and the internal return rate of the 
    buy only value averaging strategy, with a minimum and maximum investment amount per period.
    It also returns a dataframe that contains the amount of money should invest per peiod. 
    :param df: Original dataframe that contains the price at the beginning of each period
    :param value_per_period: Investment value per period
    :param increase_investment_per_period: Increase the investment by x each period
    :param min_investment_per_period: minimum investment amount at the beginning of each period
    :param max_investment_per_period: maximum investment amount at the beginning of each period
    :param display_returns: Whether to display return results or not
    :return: A dataframe that contains all the relevant info
    """
    df_va = pd.DataFrame.copy(df)
    value = value_per_period
    shares = value_per_period/df_va.loc[0, 'Adj Close']
    df_va.loc[0, 'target_value'] = value   # target value to reach at the beginning of each period
    df_va.loc[0, 'target_shares'] = shares # target share to have at the beginning of each period
    df_va.loc[0, 'current_shares'] = 0     # current shares before the period
    df_va.loc[0, 'shares_to_buy'] = shares # shares to add based on the target
    df_va.loc[0, 'total_shares'] = shares  # total shares after the investment at the beginning of each period
    df_va.loc[0, 'should_invest'] = value  # investment at the beginning of each period
    df_va.loc[0, 'total_invest'] = value   # total amount of investment including this period
    df_va.loc[0, 'total_value'] = value    # total share values after the investment at the beginning of each period
    df_va.loc[0, 'total_value_next'] = df_va.loc[0, 'total_value'] * (1 + df_va.loc[0, 'return_next_y'])
    df_va.loc[0, 'overall_return_next'] = df_va.loc[0, 'total_value_next']/df_va.loc[0, 'total_invest'] - 1
    
    for i in range(1,len(df_va.index)):
        df_va.loc[i, 'current_shares'] = df_va.loc[i-1, 'total_shares']
        
        value += value_per_period*((1 + increase_investment_per_period)**i)
        df_va.loc[i,'target_value'] = value
        df_va.loc[i,'target_shares'] = df_va.loc[i,'target_value']/df_va.loc[i,'Adj Close']
        df_va.loc[i,'shares_to_buy'] = df_va.loc[i,'target_shares'] - df_va.loc[i,'current_shares']
        df_va.loc[i,'should_invest'] = df_va.loc[i,'shares_to_buy'] * df_va.loc[i, 'Adj Close']
        
        if df_va.loc[i, 'should_invest'] < min_investment_per_period:
            df_va.loc[i, 'should_invest'] = min_investment_per_period
            df_va.loc[i, 'shares_to_buy'] = min_investment_per_period/df_va.loc[i,'Adj Close']
            
        elif df_va.loc[i, 'should_invest'] > max_investment_per_period:
            df_va.loc[i, 'should_invest'] = max_investment_per_period
            df_va.loc[i, 'shares_to_buy'] = max_investment_per_period/df_va.loc[i,'Adj Close']
        
        df_va.loc[i, 'total_shares'] = df_va.loc[i-1, 'total_shares'] + df_va.loc[i, 'shares_to_buy']
        df_va.loc[i, 'total_invest'] = df_va.loc[i-1, 'total_invest'] + df_va.loc[i, 'should_invest']
        
        df_va.loc[i, 'total_value'] = df_va.loc[i, 'total_shares']*df_va.loc[i, 'Adj Close']
        
        df_va.loc[i, 'total_value_next'] = df_va.loc[i, 'total_value'] * (1 + df_va.loc[i, 'return_next_y'])
    
        df_va.loc[i, 'overall_return_next'] = df_va.loc[i, 'total_value_next']/df_va.loc[i, 'total_invest'] - 1
        
    if display_returns:     

        # Calculate IRR
        cf = (-df_va['should_invest']).tolist()
        cf[-1] = df_va.loc[len(df_va.index) - 2, 'total_value_next']
        print("The cash flow (the negative sign stands for investment):")
        display(cf)
        irr = round(100*np.irr(cf), 2)
        print(f"The internal return rate is {irr}% per period")
              
        # Calculate overall return
        overall_return = round(df_va.loc[len(df_va.index) - 2, 'overall_return_next']*100, 2)
        print(f"The overall return rate (IRR) is {overall_return}%")
        
        # Calculate CAGR
        cagr = round(100*((df_va.loc[len(df_va.index) - 2, 'total_value_next']/
                           df_va.loc[len(df_va.index) - 2, 'total_invest'])**(1/(len(df_va.index) - 1)) - 1), 2)
        print(f"The compound annual growth rate (CAGR) is {cagr}% per period")
        
    return df_va

```


```python
def calc_dollar_cost_averaging_return(df: pd.DataFrame, value_per_period: float = 1.0, 
                                      increase_investment_per_period: float = 0.0,
                                      display_returns: bool = True) -> pd.DataFrame:
    """
    This function calculates the overall rate of return and the internal rate of return of dollar cost averaging method. 
    :param df: Original dataframe that contains the price at the beginning of each period
    :param value_per_period: Investment value per period
    :param increase_investment_per_period: Increase the investment by x each period
    :param display_returns: Whether to display return results or not
    :return: A dataframe that contains all the relevant info
    """
    df_dc = pd.DataFrame.copy(df)
    # amount to invest each period
    df_dc.loc[0, 'should_invest'] = value_per_period   
    # total amount of investment after the investment this period
    df_dc.loc[0, 'total_invest'] = value_per_period   
    # the total value by the beginning of next period
    df_dc.loc[0, 'total_value_next'] = value_per_period*(1 + df_dc.loc[0, 'return_next_y'])
    df_dc.loc[0, 'overall_return_next'] = df_dc.loc[0, 'total_value_next'] / df_dc.loc[0, 'total_invest'] - 1

    for i in range(1,len(df_dc.index)):
        df_dc.loc[i, 'should_invest'] = value_per_period * (1 + increase_investment_per_period)**i
        df_dc.loc[i, 'total_invest'] = df_dc.loc[i-1, 'total_invest'] + df_dc.loc[i, 'should_invest']
        df_dc.loc[i, 'total_value_next'] = ((df_dc.loc[i-1, 'total_value_next'] + 
                                            df_dc.loc[i, 'should_invest']) *
                                            (1 + df_dc.loc[i, 'return_next_y']) 
                                           )
        df_dc.loc[i, 'overall_return_next'] = df_dc.loc[i, 'total_value_next'] / df_dc.loc[i, 'total_invest'] - 1
        
    if display_returns:
        
        # Calculate IRR
        cf = (-df_dc['should_invest']).tolist()
        cf[-1] = df_dc.loc[len(df_dc.index) - 2, 'total_value_next']
        print("The cash flow (the negative sign stands for investment):")
        display(cf)
        irr = round(100*np.irr(cf), 2)
        print(f"The internal return rate (IRR) is {irr}% per period")
                
        # Calculate overall return
        overall_return = round(df_dc.loc[len(df_dc.index) - 2, 'overall_return_next']*100, 2)
        print(f"The overall return rate is {overall_return}%")
        
        # Calculate CAGR
        cagr = round(100*((df_dc.loc[len(df_dc.index) - 2, 'total_value_next']/
                           df_dc.loc[len(df_dc.index) - 2, 'total_invest'])**(1/(len(df_dc.index) - 1)) - 1), 2)
        print(f"The compound annual growth rate (CAGR) is {cagr}% per period")
        
    return df_dc


def plot_changes(df: pd.DataFrame,
                 y1: str = 'total_invest', y2: str = 'total_value_next',
                 xlabel: str = 'period', ylabel: str = 'value',
                 title: str = 'Visualize total investment and value each year',
                 target: int = None) -> None:
    """
    This function visualizes the total investment and the total value of the investment plan across time
    :param df: A dataframe contains total investment and total value
    :param y1: Column name
    :param y2: Column name
    :param xlabel: X label
    :param ylabel: Y label
    :param title: Title of the graph
    :return: A plot
    """
    plt.figure(figsize=(15, 5))
    plt.plot(df[y1], label=y1)
    plt.plot(df[y2], label=y2)
    if target:
        plt.axhline(y=target, c='black', label='target value')
    plt.xlim([0, len(df[y1])-1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
```

# Plan 1: Value averaging method


```python
df_va1 = calc_value_averaging_return(df, value_per_period=1200)
```
Output:

    The cash flow (the negative sign stands for investment):



    [-1200.0,
     -1061.1201961364186,
     -355.52589325481887,
     -1411.8085861271627,
     337.31005654283166,
     -747.849233851947,
     -486.97708484738405,
     -1075.2805992143667,
     2514.761361370703,
     1111.3480648011666,
     2960.6528724251534,
     2476.466314117218,
     1679.0158053734597,
     -2884.972366494456,
     -2680.227795866851,
     -4795.998397092098,
     3439.2227337015943,
     904.3614871476314,
     421.54897775958983,
     1948.1048236273016,
     -216.55745519229592,
     -9782.193899045382,
     5278.448809688893,
     2809.6319858651427,
     -471.3483862548249,
     3947.1013032192404,
     7523.195663896054,
     3551.4295421831143,
     -1241.8135883408345,
     3895.1078797634023,
     6652.273617097743,
     -3080.564625962192,
     50855.361827168424]


    The internal return rate is 12.12% per period



```python
df_va1.round(2)
```




<div>
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>Adj Close</th>
      <th>return_next_y</th>
      <th>target_value</th>
      <th>shares_should_have_in_total</th>
      <th>shares_to_buy_or_sell</th>
      <th>should_invest_or_withdraw</th>
      <th>total_invest_or_withdraw</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1988</td>
      <td>256.02</td>
      <td>0.12</td>
      <td>1200.0</td>
      <td>4.69</td>
      <td>4.69</td>
      <td>1200.00</td>
      <td>1200.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1989</td>
      <td>285.65</td>
      <td>0.35</td>
      <td>2400.0</td>
      <td>8.40</td>
      <td>3.71</td>
      <td>1061.12</td>
      <td>2261.12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1990</td>
      <td>386.16</td>
      <td>-0.06</td>
      <td>3600.0</td>
      <td>9.32</td>
      <td>0.92</td>
      <td>355.53</td>
      <td>2616.65</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1991</td>
      <td>363.44</td>
      <td>0.32</td>
      <td>4800.0</td>
      <td>13.21</td>
      <td>3.88</td>
      <td>1411.81</td>
      <td>4028.45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1992</td>
      <td>479.84</td>
      <td>0.08</td>
      <td>6000.0</td>
      <td>12.50</td>
      <td>-0.70</td>
      <td>-337.31</td>
      <td>3691.14</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1993</td>
      <td>516.00</td>
      <td>0.10</td>
      <td>7200.0</td>
      <td>13.95</td>
      <td>1.45</td>
      <td>747.85</td>
      <td>4438.99</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1994</td>
      <td>567.10</td>
      <td>0.01</td>
      <td>8400.0</td>
      <td>14.81</td>
      <td>0.86</td>
      <td>486.98</td>
      <td>4925.97</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1995</td>
      <td>575.52</td>
      <td>0.39</td>
      <td>9600.0</td>
      <td>16.68</td>
      <td>1.87</td>
      <td>1075.28</td>
      <td>6001.25</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1996</td>
      <td>798.22</td>
      <td>0.21</td>
      <td>10800.0</td>
      <td>13.53</td>
      <td>-3.15</td>
      <td>-2514.76</td>
      <td>3486.49</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1997</td>
      <td>969.05</td>
      <td>0.35</td>
      <td>12000.0</td>
      <td>12.38</td>
      <td>-1.15</td>
      <td>-1111.35</td>
      <td>2375.14</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1998</td>
      <td>1305.04</td>
      <td>0.28</td>
      <td>13200.0</td>
      <td>10.11</td>
      <td>-2.27</td>
      <td>-2960.65</td>
      <td>-585.51</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1999</td>
      <td>1668.52</td>
      <td>0.20</td>
      <td>14400.0</td>
      <td>8.63</td>
      <td>-1.48</td>
      <td>-2476.47</td>
      <td>-3061.98</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2000</td>
      <td>2002.11</td>
      <td>-0.11</td>
      <td>15600.0</td>
      <td>7.79</td>
      <td>-0.84</td>
      <td>-1679.02</td>
      <td>-4740.99</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2001</td>
      <td>1785.86</td>
      <td>-0.09</td>
      <td>16800.0</td>
      <td>9.41</td>
      <td>1.62</td>
      <td>2884.97</td>
      <td>-1856.02</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2002</td>
      <td>1628.51</td>
      <td>-0.20</td>
      <td>18000.0</td>
      <td>11.05</td>
      <td>1.65</td>
      <td>2680.23</td>
      <td>824.21</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2003</td>
      <td>1303.17</td>
      <td>0.24</td>
      <td>19200.0</td>
      <td>14.73</td>
      <td>3.68</td>
      <td>4796.00</td>
      <td>5620.21</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2004</td>
      <td>1618.05</td>
      <td>0.10</td>
      <td>20400.0</td>
      <td>12.61</td>
      <td>-2.13</td>
      <td>-3439.22</td>
      <td>2180.98</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2005</td>
      <td>1784.96</td>
      <td>0.08</td>
      <td>21600.0</td>
      <td>12.10</td>
      <td>-0.51</td>
      <td>-904.36</td>
      <td>1276.62</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2006</td>
      <td>1918.96</td>
      <td>0.14</td>
      <td>22800.0</td>
      <td>11.88</td>
      <td>-0.22</td>
      <td>-421.55</td>
      <td>855.07</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2007</td>
      <td>2183.92</td>
      <td>0.04</td>
      <td>24000.0</td>
      <td>10.99</td>
      <td>-0.89</td>
      <td>-1948.10</td>
      <td>-1093.03</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2008</td>
      <td>2273.41</td>
      <td>-0.34</td>
      <td>25200.0</td>
      <td>11.08</td>
      <td>0.10</td>
      <td>216.56</td>
      <td>-876.47</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2009</td>
      <td>1499.17</td>
      <td>0.25</td>
      <td>26400.0</td>
      <td>17.61</td>
      <td>6.53</td>
      <td>9782.19</td>
      <td>8905.72</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2010</td>
      <td>1867.06</td>
      <td>0.15</td>
      <td>27600.0</td>
      <td>14.78</td>
      <td>-2.83</td>
      <td>-5278.45</td>
      <td>3627.27</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2011</td>
      <td>2138.30</td>
      <td>0.03</td>
      <td>28800.0</td>
      <td>13.47</td>
      <td>-1.31</td>
      <td>-2809.63</td>
      <td>817.64</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2012</td>
      <td>2192.40</td>
      <td>0.17</td>
      <td>30000.0</td>
      <td>13.68</td>
      <td>0.21</td>
      <td>471.35</td>
      <td>1288.99</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2013</td>
      <td>2568.55</td>
      <td>0.28</td>
      <td>31200.0</td>
      <td>12.15</td>
      <td>-1.54</td>
      <td>-3947.10</td>
      <td>-2658.11</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2014</td>
      <td>3286.69</td>
      <td>0.15</td>
      <td>32400.0</td>
      <td>9.86</td>
      <td>-2.29</td>
      <td>-7523.20</td>
      <td>-10181.31</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2015</td>
      <td>3768.68</td>
      <td>-0.00</td>
      <td>33600.0</td>
      <td>8.92</td>
      <td>-0.94</td>
      <td>-3551.43</td>
      <td>-13732.74</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2016</td>
      <td>3763.99</td>
      <td>0.15</td>
      <td>34800.0</td>
      <td>9.25</td>
      <td>0.33</td>
      <td>1241.81</td>
      <td>-12490.93</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2017</td>
      <td>4315.08</td>
      <td>0.22</td>
      <td>36000.0</td>
      <td>8.34</td>
      <td>-0.90</td>
      <td>-3895.11</td>
      <td>-16386.03</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2018</td>
      <td>5256.28</td>
      <td>-0.05</td>
      <td>37200.0</td>
      <td>7.08</td>
      <td>-1.27</td>
      <td>-6652.27</td>
      <td>-23038.31</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2019</td>
      <td>4990.56</td>
      <td>0.32</td>
      <td>38400.0</td>
      <td>7.69</td>
      <td>0.62</td>
      <td>3080.56</td>
      <td>-19957.74</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2020</td>
      <td>6609.29</td>
      <td>NaN</td>
      <td>39600.0</td>
      <td>5.99</td>
      <td>-1.70</td>
      <td>-11255.36</td>
      <td>-31213.11</td>
    </tr>
  </tbody>
</table>
</div>



In the last notebook, we already know that the IRR for dollar cost averaging is below 10% per year. Using the same test data, the IRR of the value averaging method is about 12% per year, which supports our hypothesis.

However, we can also observe that, because we are taking money out from the investment plan, the final value of the value averaging is way below that of the dollar cost averaing.

so the value averaging method is not really a good one for long term investment.

In the last notebook we know that, if we invest 1200 dollar per year starting from 1988, we will have about 261192 dollar by 2020. If we want to accumulate the same amount of money by 2020 using the value averaging, we need to set a higher amount of investment each year.

So let's check the IRR if we increase the investment each year by 10% using the value averaging strategy. 

# Plan 2: Increase investment each year by 10%


```python
df_va2 = calc_value_averaging_return(df, value_per_period=1200, 
                                     increase_investment_per_period=0.1)
```
Output:

    The cash flow (the negative sign stands for investment):



    [-1200.0,
     -1181.1201961364188,
     -565.3021879175599,
     -1830.8954733603034,
     26.743993103820433,
     -1380.5268715179054,
     -1208.9720776032202,
     -2169.4270511041677,
     2737.892833854878,
     657.8958460156231,
     3518.5181924009958,
     2769.824630978246,
     1364.3601939671873,
     -7321.194096811861,
     -7514.80857127775,
     -12629.617352887119,
     4909.707669716029,
     -1046.4989781541021,
     -2564.0511882120322,
     1137.4449047712715,
     -5256.666376415341,
     -35036.57899214694,
     11257.990836019511,
     3121.7177152010245,
     -9132.857546202227,
     7246.443979130866,
     22329.4920899403,
     5579.038139518119,
     -17505.61372633736,
     7077.75812974924,
     22115.78642833783,
     -34070.51321757376,
     319654.1856549351]


    The internal return rate is 11.21% per period



```python
df_va2.round(2)
```




<div>
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>Adj Close</th>
      <th>return_next_y</th>
      <th>target_value</th>
      <th>shares_should_have_in_total</th>
      <th>shares_to_buy_or_sell</th>
      <th>should_invest_or_withdraw</th>
      <th>total_invest_or_withdraw</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1988</td>
      <td>256.02</td>
      <td>0.12</td>
      <td>1200.00</td>
      <td>4.69</td>
      <td>4.69</td>
      <td>1200.00</td>
      <td>1200.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1989</td>
      <td>285.65</td>
      <td>0.35</td>
      <td>2520.00</td>
      <td>8.82</td>
      <td>4.13</td>
      <td>1181.12</td>
      <td>2381.12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1990</td>
      <td>386.16</td>
      <td>-0.06</td>
      <td>3972.00</td>
      <td>10.29</td>
      <td>1.46</td>
      <td>565.30</td>
      <td>2946.42</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1991</td>
      <td>363.44</td>
      <td>0.32</td>
      <td>5569.20</td>
      <td>15.32</td>
      <td>5.04</td>
      <td>1830.90</td>
      <td>4777.32</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1992</td>
      <td>479.84</td>
      <td>0.08</td>
      <td>7326.12</td>
      <td>15.27</td>
      <td>-0.06</td>
      <td>-26.74</td>
      <td>4750.57</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1993</td>
      <td>516.00</td>
      <td>0.10</td>
      <td>9258.73</td>
      <td>17.94</td>
      <td>2.68</td>
      <td>1380.53</td>
      <td>6131.10</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1994</td>
      <td>567.10</td>
      <td>0.01</td>
      <td>11384.61</td>
      <td>20.08</td>
      <td>2.13</td>
      <td>1208.97</td>
      <td>7340.07</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1995</td>
      <td>575.52</td>
      <td>0.39</td>
      <td>13723.07</td>
      <td>23.84</td>
      <td>3.77</td>
      <td>2169.43</td>
      <td>9509.50</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1996</td>
      <td>798.22</td>
      <td>0.21</td>
      <td>16295.37</td>
      <td>20.41</td>
      <td>-3.43</td>
      <td>-2737.89</td>
      <td>6771.61</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1997</td>
      <td>969.05</td>
      <td>0.35</td>
      <td>19124.91</td>
      <td>19.74</td>
      <td>-0.68</td>
      <td>-657.90</td>
      <td>6113.71</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1998</td>
      <td>1305.04</td>
      <td>0.28</td>
      <td>22237.40</td>
      <td>17.04</td>
      <td>-2.70</td>
      <td>-3518.52</td>
      <td>2595.19</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1999</td>
      <td>1668.52</td>
      <td>0.20</td>
      <td>25661.14</td>
      <td>15.38</td>
      <td>-1.66</td>
      <td>-2769.82</td>
      <td>-174.63</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2000</td>
      <td>2002.11</td>
      <td>-0.11</td>
      <td>29427.25</td>
      <td>14.70</td>
      <td>-0.68</td>
      <td>-1364.36</td>
      <td>-1538.99</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2001</td>
      <td>1785.86</td>
      <td>-0.09</td>
      <td>33569.98</td>
      <td>18.80</td>
      <td>4.10</td>
      <td>7321.19</td>
      <td>5782.20</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2002</td>
      <td>1628.51</td>
      <td>-0.20</td>
      <td>38126.98</td>
      <td>23.41</td>
      <td>4.61</td>
      <td>7514.81</td>
      <td>13297.01</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2003</td>
      <td>1303.17</td>
      <td>0.24</td>
      <td>43139.68</td>
      <td>33.10</td>
      <td>9.69</td>
      <td>12629.62</td>
      <td>25926.63</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2004</td>
      <td>1618.05</td>
      <td>0.10</td>
      <td>48653.64</td>
      <td>30.07</td>
      <td>-3.03</td>
      <td>-4909.71</td>
      <td>21016.92</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2005</td>
      <td>1784.96</td>
      <td>0.08</td>
      <td>54719.01</td>
      <td>30.66</td>
      <td>0.59</td>
      <td>1046.50</td>
      <td>22063.42</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2006</td>
      <td>1918.96</td>
      <td>0.14</td>
      <td>61390.91</td>
      <td>31.99</td>
      <td>1.34</td>
      <td>2564.05</td>
      <td>24627.47</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2007</td>
      <td>2183.92</td>
      <td>0.04</td>
      <td>68730.00</td>
      <td>31.47</td>
      <td>-0.52</td>
      <td>-1137.44</td>
      <td>23490.03</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2008</td>
      <td>2273.41</td>
      <td>-0.34</td>
      <td>76803.00</td>
      <td>33.78</td>
      <td>2.31</td>
      <td>5256.67</td>
      <td>28746.69</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2009</td>
      <td>1499.17</td>
      <td>0.25</td>
      <td>85683.30</td>
      <td>57.15</td>
      <td>23.37</td>
      <td>35036.58</td>
      <td>63783.27</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2010</td>
      <td>1867.06</td>
      <td>0.15</td>
      <td>95451.63</td>
      <td>51.12</td>
      <td>-6.03</td>
      <td>-11257.99</td>
      <td>52525.28</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2011</td>
      <td>2138.30</td>
      <td>0.03</td>
      <td>106196.79</td>
      <td>49.66</td>
      <td>-1.46</td>
      <td>-3121.72</td>
      <td>49403.56</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2012</td>
      <td>2192.40</td>
      <td>0.17</td>
      <td>118016.47</td>
      <td>53.83</td>
      <td>4.17</td>
      <td>9132.86</td>
      <td>58536.42</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2013</td>
      <td>2568.55</td>
      <td>0.28</td>
      <td>131018.12</td>
      <td>51.01</td>
      <td>-2.82</td>
      <td>-7246.44</td>
      <td>51289.98</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2014</td>
      <td>3286.69</td>
      <td>0.15</td>
      <td>145319.93</td>
      <td>44.21</td>
      <td>-6.79</td>
      <td>-22329.49</td>
      <td>28960.48</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2015</td>
      <td>3768.68</td>
      <td>-0.00</td>
      <td>161051.92</td>
      <td>42.73</td>
      <td>-1.48</td>
      <td>-5579.04</td>
      <td>23381.45</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2016</td>
      <td>3763.99</td>
      <td>0.15</td>
      <td>178357.12</td>
      <td>47.39</td>
      <td>4.65</td>
      <td>17505.61</td>
      <td>40887.06</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2017</td>
      <td>4315.08</td>
      <td>0.22</td>
      <td>197392.83</td>
      <td>45.74</td>
      <td>-1.64</td>
      <td>-7077.76</td>
      <td>33809.30</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2018</td>
      <td>5256.28</td>
      <td>-0.05</td>
      <td>218332.11</td>
      <td>41.54</td>
      <td>-4.21</td>
      <td>-22115.79</td>
      <td>11693.52</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2019</td>
      <td>4990.56</td>
      <td>0.32</td>
      <td>241365.32</td>
      <td>48.36</td>
      <td>6.83</td>
      <td>34070.51</td>
      <td>45764.03</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2020</td>
      <td>6609.29</td>
      <td>NaN</td>
      <td>266701.85</td>
      <td>40.35</td>
      <td>-8.01</td>
      <td>-52952.33</td>
      <td>-7188.30</td>
    </tr>
  </tbody>
</table>
</div>



With the increased amount of investment, the IRR of value averaging Plan 2 is still higher than that of the dollar cost averaging method. 

However, there is a drawback of the value averaging method:

If we check the cashflow, we can see that, whenever there is a market crash (especially at the lates stage of the investment), we need to invest a huge amount of money (e.g., 2019).

What if we adjust the value averaging method into a buy and hold strategy? We can also set a minimum and maximum investment value per period.

# Plan 3: No selling, with minimum and maximum investment pre period


```python
df_va3 = calc_buy_only_value_averaging_return(df, value_per_period=1200, 
                                             increase_investment_per_period=0.10,
                                             min_investment_per_period=600,
                                             max_investment_per_period=6000)
```

Output:

    The cash flow (the negative sign stands for investment):



    [-1200.0,
     -1181.1201961364188,
     -600.0,
     -1798.239131977554,
     -600.0,
     -706.552415637322,
     -1208.9720776032202,
     -2169.4270511041677,
     -600.0,
     -600.0,
     -600.0,
     -600.0,
     -600.0,
     -600.0,
     -600.0,
     -6000.0,
     -600.0,
     -600.0,
     -600.0,
     -600.0,
     -1922.6955641279228,
     -6000.0,
     -6000.0,
     -6000.0,
     -6000.0,
     -6000.0,
     -600.0,
     -600.0,
     -600.0,
     -600.0,
     -600.0,
     -600.0,
     328826.905866578]


    The internal return rate is 10.54% per period
    The overall return rate (IRR) is 471.01%
    The compound annual growth rate (CAGR) is 5.6% per period


We can see here that the IRR is still higher than the dollar averaging method, but the overall return by the next period still performs poorer compared to the dollar cost averaging method.


```python
df_va3.round(2)
```




<div>
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>Adj Close</th>
      <th>return_next_y</th>
      <th>target_value</th>
      <th>target_shares</th>
      <th>current_shares</th>
      <th>shares_to_buy</th>
      <th>total_shares</th>
      <th>should_invest</th>
      <th>total_invest</th>
      <th>total_value</th>
      <th>total_value_next</th>
      <th>overall_return_next</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1988</td>
      <td>256.02</td>
      <td>0.12</td>
      <td>1200.00</td>
      <td>4.69</td>
      <td>0.00</td>
      <td>4.69</td>
      <td>4.69</td>
      <td>1200.00</td>
      <td>1200.00</td>
      <td>1200.00</td>
      <td>1338.88</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1989</td>
      <td>285.65</td>
      <td>0.35</td>
      <td>2520.00</td>
      <td>8.82</td>
      <td>4.69</td>
      <td>4.13</td>
      <td>8.82</td>
      <td>1181.12</td>
      <td>2381.12</td>
      <td>2520.00</td>
      <td>3406.70</td>
      <td>0.43</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1990</td>
      <td>386.16</td>
      <td>-0.06</td>
      <td>3972.00</td>
      <td>10.29</td>
      <td>8.82</td>
      <td>1.55</td>
      <td>10.38</td>
      <td>600.00</td>
      <td>2981.12</td>
      <td>4006.70</td>
      <td>3770.96</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1991</td>
      <td>363.44</td>
      <td>0.32</td>
      <td>5569.20</td>
      <td>15.32</td>
      <td>10.38</td>
      <td>4.95</td>
      <td>15.32</td>
      <td>1798.24</td>
      <td>4779.36</td>
      <td>5569.20</td>
      <td>7352.86</td>
      <td>0.54</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1992</td>
      <td>479.84</td>
      <td>0.08</td>
      <td>7326.12</td>
      <td>15.27</td>
      <td>15.32</td>
      <td>1.25</td>
      <td>16.57</td>
      <td>600.00</td>
      <td>5379.36</td>
      <td>7952.86</td>
      <td>8552.18</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1993</td>
      <td>516.00</td>
      <td>0.10</td>
      <td>9258.73</td>
      <td>17.94</td>
      <td>16.57</td>
      <td>1.37</td>
      <td>17.94</td>
      <td>706.55</td>
      <td>6085.91</td>
      <td>9258.73</td>
      <td>10175.63</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1994</td>
      <td>567.10</td>
      <td>0.01</td>
      <td>11384.61</td>
      <td>20.08</td>
      <td>17.94</td>
      <td>2.13</td>
      <td>20.08</td>
      <td>1208.97</td>
      <td>7294.88</td>
      <td>11384.61</td>
      <td>11553.64</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1995</td>
      <td>575.52</td>
      <td>0.39</td>
      <td>13723.07</td>
      <td>23.84</td>
      <td>20.08</td>
      <td>3.77</td>
      <td>23.84</td>
      <td>2169.43</td>
      <td>9464.31</td>
      <td>13723.07</td>
      <td>19033.27</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1996</td>
      <td>798.22</td>
      <td>0.21</td>
      <td>16295.37</td>
      <td>20.41</td>
      <td>23.84</td>
      <td>0.75</td>
      <td>24.60</td>
      <td>600.00</td>
      <td>10064.31</td>
      <td>19633.27</td>
      <td>23835.05</td>
      <td>1.37</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1997</td>
      <td>969.05</td>
      <td>0.35</td>
      <td>19124.91</td>
      <td>19.74</td>
      <td>24.60</td>
      <td>0.62</td>
      <td>25.22</td>
      <td>600.00</td>
      <td>10664.31</td>
      <td>24435.05</td>
      <td>32907.20</td>
      <td>2.09</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1998</td>
      <td>1305.04</td>
      <td>0.28</td>
      <td>22237.40</td>
      <td>17.04</td>
      <td>25.22</td>
      <td>0.46</td>
      <td>25.68</td>
      <td>600.00</td>
      <td>11264.31</td>
      <td>33507.20</td>
      <td>42839.63</td>
      <td>2.80</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1999</td>
      <td>1668.52</td>
      <td>0.20</td>
      <td>25661.14</td>
      <td>15.38</td>
      <td>25.68</td>
      <td>0.36</td>
      <td>26.03</td>
      <td>600.00</td>
      <td>11864.31</td>
      <td>43439.63</td>
      <td>52124.59</td>
      <td>3.39</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2000</td>
      <td>2002.11</td>
      <td>-0.11</td>
      <td>29427.25</td>
      <td>14.70</td>
      <td>26.03</td>
      <td>0.30</td>
      <td>26.33</td>
      <td>600.00</td>
      <td>12464.31</td>
      <td>52724.59</td>
      <td>47029.75</td>
      <td>2.77</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2001</td>
      <td>1785.86</td>
      <td>-0.09</td>
      <td>33569.98</td>
      <td>18.80</td>
      <td>26.33</td>
      <td>0.34</td>
      <td>26.67</td>
      <td>600.00</td>
      <td>13064.31</td>
      <td>47629.75</td>
      <td>43433.15</td>
      <td>2.32</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2002</td>
      <td>1628.51</td>
      <td>-0.20</td>
      <td>38126.98</td>
      <td>23.41</td>
      <td>26.67</td>
      <td>0.37</td>
      <td>27.04</td>
      <td>600.00</td>
      <td>13664.31</td>
      <td>44033.15</td>
      <td>35236.31</td>
      <td>1.58</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2003</td>
      <td>1303.17</td>
      <td>0.24</td>
      <td>43139.68</td>
      <td>33.10</td>
      <td>27.04</td>
      <td>4.60</td>
      <td>31.64</td>
      <td>6000.00</td>
      <td>19664.31</td>
      <td>41236.31</td>
      <td>51200.08</td>
      <td>1.60</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2004</td>
      <td>1618.05</td>
      <td>0.10</td>
      <td>48653.64</td>
      <td>30.07</td>
      <td>31.64</td>
      <td>0.37</td>
      <td>32.01</td>
      <td>600.00</td>
      <td>20264.31</td>
      <td>51800.08</td>
      <td>57143.52</td>
      <td>1.82</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2005</td>
      <td>1784.96</td>
      <td>0.08</td>
      <td>54719.01</td>
      <td>30.66</td>
      <td>32.01</td>
      <td>0.34</td>
      <td>32.35</td>
      <td>600.00</td>
      <td>20864.31</td>
      <td>57743.52</td>
      <td>62078.42</td>
      <td>1.98</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2006</td>
      <td>1918.96</td>
      <td>0.14</td>
      <td>61390.91</td>
      <td>31.99</td>
      <td>32.35</td>
      <td>0.31</td>
      <td>32.66</td>
      <td>600.00</td>
      <td>21464.31</td>
      <td>62678.42</td>
      <td>71332.73</td>
      <td>2.32</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2007</td>
      <td>2183.92</td>
      <td>0.04</td>
      <td>68730.00</td>
      <td>31.47</td>
      <td>32.66</td>
      <td>0.27</td>
      <td>32.94</td>
      <td>600.00</td>
      <td>22064.31</td>
      <td>71932.73</td>
      <td>74880.30</td>
      <td>2.39</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2008</td>
      <td>2273.41</td>
      <td>-0.34</td>
      <td>76803.00</td>
      <td>33.78</td>
      <td>32.94</td>
      <td>0.85</td>
      <td>33.78</td>
      <td>1922.70</td>
      <td>23987.01</td>
      <td>76803.00</td>
      <td>50646.72</td>
      <td>1.11</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2009</td>
      <td>1499.17</td>
      <td>0.25</td>
      <td>85683.30</td>
      <td>57.15</td>
      <td>33.78</td>
      <td>4.00</td>
      <td>37.79</td>
      <td>6000.00</td>
      <td>29987.01</td>
      <td>56646.72</td>
      <td>70547.59</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2010</td>
      <td>1867.06</td>
      <td>0.15</td>
      <td>95451.63</td>
      <td>51.12</td>
      <td>37.79</td>
      <td>3.21</td>
      <td>41.00</td>
      <td>6000.00</td>
      <td>35987.01</td>
      <td>76547.59</td>
      <td>87668.15</td>
      <td>1.44</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2011</td>
      <td>2138.30</td>
      <td>0.03</td>
      <td>106196.79</td>
      <td>49.66</td>
      <td>41.00</td>
      <td>2.81</td>
      <td>43.80</td>
      <td>6000.00</td>
      <td>41987.01</td>
      <td>93668.15</td>
      <td>96038.00</td>
      <td>1.29</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2012</td>
      <td>2192.40</td>
      <td>0.17</td>
      <td>118016.47</td>
      <td>53.83</td>
      <td>43.80</td>
      <td>2.74</td>
      <td>46.54</td>
      <td>6000.00</td>
      <td>47987.01</td>
      <td>102038.00</td>
      <td>119544.66</td>
      <td>1.49</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2013</td>
      <td>2568.55</td>
      <td>0.28</td>
      <td>131018.12</td>
      <td>51.01</td>
      <td>46.54</td>
      <td>2.34</td>
      <td>48.88</td>
      <td>6000.00</td>
      <td>53987.01</td>
      <td>125544.66</td>
      <td>160645.64</td>
      <td>1.98</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2014</td>
      <td>3286.69</td>
      <td>0.15</td>
      <td>145319.93</td>
      <td>44.21</td>
      <td>48.88</td>
      <td>0.18</td>
      <td>49.06</td>
      <td>600.00</td>
      <td>54587.01</td>
      <td>161245.64</td>
      <td>184892.16</td>
      <td>2.39</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2015</td>
      <td>3768.68</td>
      <td>-0.00</td>
      <td>161051.92</td>
      <td>42.73</td>
      <td>49.06</td>
      <td>0.16</td>
      <td>49.22</td>
      <td>600.00</td>
      <td>55187.01</td>
      <td>185492.16</td>
      <td>185261.33</td>
      <td>2.36</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2016</td>
      <td>3763.99</td>
      <td>0.15</td>
      <td>178357.12</td>
      <td>47.39</td>
      <td>49.22</td>
      <td>0.16</td>
      <td>49.38</td>
      <td>600.00</td>
      <td>55787.01</td>
      <td>185861.33</td>
      <td>213073.50</td>
      <td>2.82</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2017</td>
      <td>4315.08</td>
      <td>0.22</td>
      <td>197392.83</td>
      <td>45.74</td>
      <td>49.38</td>
      <td>0.14</td>
      <td>49.52</td>
      <td>600.00</td>
      <td>56387.01</td>
      <td>213673.50</td>
      <td>260279.68</td>
      <td>3.62</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2018</td>
      <td>5256.28</td>
      <td>-0.05</td>
      <td>218332.11</td>
      <td>41.54</td>
      <td>49.52</td>
      <td>0.11</td>
      <td>49.63</td>
      <td>600.00</td>
      <td>56987.01</td>
      <td>260879.68</td>
      <td>247691.48</td>
      <td>3.35</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2019</td>
      <td>4990.56</td>
      <td>0.32</td>
      <td>241365.32</td>
      <td>48.36</td>
      <td>49.63</td>
      <td>0.12</td>
      <td>49.75</td>
      <td>600.00</td>
      <td>57587.01</td>
      <td>248291.48</td>
      <td>328826.91</td>
      <td>4.71</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2020</td>
      <td>6609.29</td>
      <td>NaN</td>
      <td>266701.85</td>
      <td>40.35</td>
      <td>49.75</td>
      <td>0.09</td>
      <td>49.84</td>
      <td>600.00</td>
      <td>58187.01</td>
      <td>329426.91</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_changes(df_va3, title='Value averaging with adjustment: No selling, with min/max investment amount', xlabel='year')
```


![png](/assets/4_output_22_0.png)


# Compare with dollar cost averaging


```python
# Dollar cost averaging: increase the investment per year by 2% 
df_dc = calc_dollar_cost_averaging_return(df, value_per_period=1200, 
                                          increase_investment_per_period=0.02)
```

Output:

    The cash flow (the negative sign stands for investment):



    [-1200.0,
     -1224.0,
     -1248.48,
     -1273.4496000000001,
     -1298.918592,
     -1324.8969638400001,
     -1351.3949031168002,
     -1378.422801179136,
     -1405.991257202719,
     -1434.1110823467732,
     -1462.7933039937088,
     -1492.049170073583,
     -1521.8901534750546,
     -1552.327956544556,
     -1583.374515675447,
     -1615.042005988956,
     -1647.3428461087349,
     -1680.2897030309098,
     -1713.8954970915281,
     -1748.1734070333584,
     -1783.1368751740258,
     -1818.7996126775063,
     -1855.1756049310566,
     -1892.2791170296778,
     -1930.1246993702714,
     -1968.7271933576767,
     -2008.1017372248302,
     -2048.263771969327,
     -2089.2290474087135,
     -2131.0136283568877,
     -2173.633900924026,
     -2217.106578942506,
     311273.63335198513]


    The internal return rate (IRR) is 9.91% per period
    The overall return rate is 486.51%
    The compound annual growth rate (CAGR) is 5.68% per period



```python
df_dc.round(2)
```




<div>
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>Adj Close</th>
      <th>return_next_y</th>
      <th>should_invest</th>
      <th>total_invest</th>
      <th>total_value_next</th>
      <th>overall_return_next</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1988</td>
      <td>256.02</td>
      <td>0.12</td>
      <td>1200.00</td>
      <td>1200.00</td>
      <td>1338.88</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1989</td>
      <td>285.65</td>
      <td>0.35</td>
      <td>1224.00</td>
      <td>2424.00</td>
      <td>3464.67</td>
      <td>0.43</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1990</td>
      <td>386.16</td>
      <td>-0.06</td>
      <td>1248.48</td>
      <td>3672.48</td>
      <td>4435.84</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1991</td>
      <td>363.44</td>
      <td>0.32</td>
      <td>1273.45</td>
      <td>4945.93</td>
      <td>7537.83</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1992</td>
      <td>479.84</td>
      <td>0.08</td>
      <td>1298.92</td>
      <td>6244.85</td>
      <td>9502.67</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1993</td>
      <td>516.00</td>
      <td>0.10</td>
      <td>1324.90</td>
      <td>7569.75</td>
      <td>11899.83</td>
      <td>0.57</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1994</td>
      <td>567.10</td>
      <td>0.01</td>
      <td>1351.39</td>
      <td>8921.14</td>
      <td>13447.97</td>
      <td>0.51</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1995</td>
      <td>575.52</td>
      <td>0.39</td>
      <td>1378.42</td>
      <td>10299.56</td>
      <td>20563.53</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1996</td>
      <td>798.22</td>
      <td>0.21</td>
      <td>1405.99</td>
      <td>11705.55</td>
      <td>26671.30</td>
      <td>1.28</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1997</td>
      <td>969.05</td>
      <td>0.35</td>
      <td>1434.11</td>
      <td>13139.67</td>
      <td>37850.15</td>
      <td>1.88</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1998</td>
      <td>1305.04</td>
      <td>0.28</td>
      <td>1462.79</td>
      <td>14602.46</td>
      <td>50262.40</td>
      <td>2.44</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1999</td>
      <td>1668.52</td>
      <td>0.20</td>
      <td>1492.05</td>
      <td>16094.51</td>
      <td>62101.80</td>
      <td>2.86</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2000</td>
      <td>2002.11</td>
      <td>-0.11</td>
      <td>1521.89</td>
      <td>17616.40</td>
      <td>56751.63</td>
      <td>2.22</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2001</td>
      <td>1785.86</td>
      <td>-0.09</td>
      <td>1552.33</td>
      <td>19168.73</td>
      <td>53166.86</td>
      <td>1.77</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2002</td>
      <td>1628.51</td>
      <td>-0.20</td>
      <td>1583.37</td>
      <td>20752.10</td>
      <td>43812.36</td>
      <td>1.11</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2003</td>
      <td>1303.17</td>
      <td>0.24</td>
      <td>1615.04</td>
      <td>22367.14</td>
      <td>56403.85</td>
      <td>1.52</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2004</td>
      <td>1618.05</td>
      <td>0.10</td>
      <td>1647.34</td>
      <td>24014.49</td>
      <td>64039.47</td>
      <td>1.67</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2005</td>
      <td>1784.96</td>
      <td>0.08</td>
      <td>1680.29</td>
      <td>25694.77</td>
      <td>70653.45</td>
      <td>1.75</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2006</td>
      <td>1918.96</td>
      <td>0.14</td>
      <td>1713.90</td>
      <td>27408.67</td>
      <td>82359.45</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2007</td>
      <td>2183.92</td>
      <td>0.04</td>
      <td>1748.17</td>
      <td>29156.84</td>
      <td>87554.08</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2008</td>
      <td>2273.41</td>
      <td>-0.34</td>
      <td>1783.14</td>
      <td>30939.98</td>
      <td>58912.25</td>
      <td>0.90</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2009</td>
      <td>1499.17</td>
      <td>0.25</td>
      <td>1818.80</td>
      <td>32758.78</td>
      <td>75634.19</td>
      <td>1.31</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2010</td>
      <td>1867.06</td>
      <td>0.15</td>
      <td>1855.18</td>
      <td>34613.96</td>
      <td>88746.75</td>
      <td>1.56</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2011</td>
      <td>2138.30</td>
      <td>0.03</td>
      <td>1892.28</td>
      <td>36506.23</td>
      <td>92932.23</td>
      <td>1.55</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2012</td>
      <td>2192.40</td>
      <td>0.17</td>
      <td>1930.12</td>
      <td>38436.36</td>
      <td>111137.90</td>
      <td>1.89</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2013</td>
      <td>2568.55</td>
      <td>0.28</td>
      <td>1968.73</td>
      <td>40405.09</td>
      <td>144730.06</td>
      <td>2.58</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2014</td>
      <td>3286.69</td>
      <td>0.15</td>
      <td>2008.10</td>
      <td>42413.19</td>
      <td>168257.17</td>
      <td>2.97</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2015</td>
      <td>3768.68</td>
      <td>-0.00</td>
      <td>2048.26</td>
      <td>44461.45</td>
      <td>170093.50</td>
      <td>2.83</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2016</td>
      <td>3763.99</td>
      <td>0.15</td>
      <td>2089.23</td>
      <td>46550.68</td>
      <td>197392.20</td>
      <td>3.24</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2017</td>
      <td>4315.08</td>
      <td>0.22</td>
      <td>2131.01</td>
      <td>48681.70</td>
      <td>243042.96</td>
      <td>3.99</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2018</td>
      <td>5256.28</td>
      <td>-0.05</td>
      <td>2173.63</td>
      <td>50855.33</td>
      <td>232820.20</td>
      <td>3.58</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2019</td>
      <td>4990.56</td>
      <td>0.32</td>
      <td>2217.11</td>
      <td>53072.44</td>
      <td>311273.63</td>
      <td>4.87</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2020</td>
      <td>6609.29</td>
      <td>NaN</td>
      <td>2261.45</td>
      <td>55333.88</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Dollar cost averaging: increase the investment per year by 2% 
plot_changes(df_dc, title='Yearly dollar cost averaging', xlabel='year')
```


![png](/assets/4_output_26_0.png)



```python
# Compare dollar cost averaging(2% increase by year) with 
# value averaging method (no selling, with min and max value per period)
select_columns = ['should_invest', 'total_value_next', 'overall_return_next']
df_compare = pd.merge(df_dc[select_columns], df_va3[select_columns], 
         left_index=True, right_index=True, 
         suffixes=('_dc', '_va3')).round(2)
df_compare
```




<div>
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>should_invest_dc</th>
      <th>total_value_next_dc</th>
      <th>overall_return_next_dc</th>
      <th>should_invest_va3</th>
      <th>total_value_next_va3</th>
      <th>overall_return_next_va3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1200.00</td>
      <td>1338.88</td>
      <td>0.12</td>
      <td>1200.00</td>
      <td>1338.88</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1224.00</td>
      <td>3464.67</td>
      <td>0.43</td>
      <td>1181.12</td>
      <td>3406.70</td>
      <td>0.43</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1248.48</td>
      <td>4435.84</td>
      <td>0.21</td>
      <td>600.00</td>
      <td>3770.96</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1273.45</td>
      <td>7537.83</td>
      <td>0.52</td>
      <td>1798.24</td>
      <td>7352.86</td>
      <td>0.54</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1298.92</td>
      <td>9502.67</td>
      <td>0.52</td>
      <td>600.00</td>
      <td>8552.18</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1324.90</td>
      <td>11899.83</td>
      <td>0.57</td>
      <td>706.55</td>
      <td>10175.63</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1351.39</td>
      <td>13447.97</td>
      <td>0.51</td>
      <td>1208.97</td>
      <td>11553.64</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1378.42</td>
      <td>20563.53</td>
      <td>1.00</td>
      <td>2169.43</td>
      <td>19033.27</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1405.99</td>
      <td>26671.30</td>
      <td>1.28</td>
      <td>600.00</td>
      <td>23835.05</td>
      <td>1.37</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1434.11</td>
      <td>37850.15</td>
      <td>1.88</td>
      <td>600.00</td>
      <td>32907.20</td>
      <td>2.09</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1462.79</td>
      <td>50262.40</td>
      <td>2.44</td>
      <td>600.00</td>
      <td>42839.63</td>
      <td>2.80</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1492.05</td>
      <td>62101.80</td>
      <td>2.86</td>
      <td>600.00</td>
      <td>52124.59</td>
      <td>3.39</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1521.89</td>
      <td>56751.63</td>
      <td>2.22</td>
      <td>600.00</td>
      <td>47029.75</td>
      <td>2.77</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1552.33</td>
      <td>53166.86</td>
      <td>1.77</td>
      <td>600.00</td>
      <td>43433.15</td>
      <td>2.32</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1583.37</td>
      <td>43812.36</td>
      <td>1.11</td>
      <td>600.00</td>
      <td>35236.31</td>
      <td>1.58</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1615.04</td>
      <td>56403.85</td>
      <td>1.52</td>
      <td>6000.00</td>
      <td>51200.08</td>
      <td>1.60</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1647.34</td>
      <td>64039.47</td>
      <td>1.67</td>
      <td>600.00</td>
      <td>57143.52</td>
      <td>1.82</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1680.29</td>
      <td>70653.45</td>
      <td>1.75</td>
      <td>600.00</td>
      <td>62078.42</td>
      <td>1.98</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1713.90</td>
      <td>82359.45</td>
      <td>2.00</td>
      <td>600.00</td>
      <td>71332.73</td>
      <td>2.32</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1748.17</td>
      <td>87554.08</td>
      <td>2.00</td>
      <td>600.00</td>
      <td>74880.30</td>
      <td>2.39</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1783.14</td>
      <td>58912.25</td>
      <td>0.90</td>
      <td>1922.70</td>
      <td>50646.72</td>
      <td>1.11</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1818.80</td>
      <td>75634.19</td>
      <td>1.31</td>
      <td>6000.00</td>
      <td>70547.59</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1855.18</td>
      <td>88746.75</td>
      <td>1.56</td>
      <td>6000.00</td>
      <td>87668.15</td>
      <td>1.44</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1892.28</td>
      <td>92932.23</td>
      <td>1.55</td>
      <td>6000.00</td>
      <td>96038.00</td>
      <td>1.29</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1930.12</td>
      <td>111137.90</td>
      <td>1.89</td>
      <td>6000.00</td>
      <td>119544.66</td>
      <td>1.49</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1968.73</td>
      <td>144730.06</td>
      <td>2.58</td>
      <td>6000.00</td>
      <td>160645.64</td>
      <td>1.98</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2008.10</td>
      <td>168257.17</td>
      <td>2.97</td>
      <td>600.00</td>
      <td>184892.16</td>
      <td>2.39</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2048.26</td>
      <td>170093.50</td>
      <td>2.83</td>
      <td>600.00</td>
      <td>185261.33</td>
      <td>2.36</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2089.23</td>
      <td>197392.20</td>
      <td>3.24</td>
      <td>600.00</td>
      <td>213073.50</td>
      <td>2.82</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2131.01</td>
      <td>243042.96</td>
      <td>3.99</td>
      <td>600.00</td>
      <td>260279.68</td>
      <td>3.62</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2173.63</td>
      <td>232820.20</td>
      <td>3.58</td>
      <td>600.00</td>
      <td>247691.48</td>
      <td>3.35</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2217.11</td>
      <td>311273.63</td>
      <td>4.87</td>
      <td>600.00</td>
      <td>328826.91</td>
      <td>4.71</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2261.45</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>600.00</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
def compare_strategy(df, str1, str2, str1_name, str2_name):
    print(f'number of times {str1_name} is better than {str2_name}: {df.loc[df[str1] > df[str2],:].shape[0]}')
    print(f'number of times {str1_name} is equal to {str2_name}: {df.loc[df[str1] == df[str2],:].shape[0]}')
    print(f'number of times {str1_name} is worse than {str2_name}: {df.loc[df[str1] < df[str2],:].shape[0]}')
```


```python
compare_strategy(df_compare, 'overall_return_next_dc', 'overall_return_next_va3', 
                 'yearly dollar-cost-averaging', 'adjusted version of value averaging')
```

Output:

    number of times yearly dollar-cost-averaging is better than adjusted version of value averaging: 10
    number of times yearly dollar-cost-averaging is equal to adjusted version of value averaging: 2
    number of times yearly dollar-cost-averaging is worse than adjusted version of value averaging: 20


# Conclusion

Based on the results above, we can see that value averaging is a good method to increase IRR, but it does not fit the purpose of long term investment as it takes out the profit too early.

The **adjusted version of the value averaging** seems to **outperform the dollar-cost averaging in most years**. But the difference is not huge. 

So for **long term investment**, it seems that **dollar cost averaging with an annual increasement of 2% works better (with the least effort)**.

In the next blogpost, we are going to examine how to invest a fixed amount of money in short term. To be more specfic, we are going to compare the dollar-cost avearging method, value averaging with all-in moethod for short term invesment.
