---
layout: post
title: "Financial Analysis in Python #2: Backtest Dollar Cost Averaging Strategy for Long-term Investment"
date: 2020-09-13T16:15:20+02:00
tags: [Finance, Python, visualization, long-term investment, retirement]
categories: [Finance, Data Science]
---


# Overview

This is the second blog post of my **Finaical Analysis in Python** series.

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

In the last notebook, we already talked about how to visualize the return of a long term investment plan. 

We made an assumption that the annual return rate is about 5%. How can we ensure that we can invest our savings properly so that we can have a 5% annual return?

In this notebook, we are going to use the historical data of SP500 to do some backtesting.

We will calculate the overall rate of return and the internal rate of return for two dollor cost averaging methods.


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

# A peak into the history

The first step here is to download data. I used a method introducted in a [datacamp tutorial](https://www.datacamp.com/community/tutorials/finance-python-trading).

The [total return index](https://www.investopedia.com/terms/t/total_return_index.asp#:~:text=The%20total%20return%20index%20is,representation%20of%20the%20index's%20performance) of SP500 between 1988 and 2020 is used here.


```python
# Get historical data
startdate = datetime.datetime(1988, 1, 1)
enddate = datetime.datetime(2021, 1, 1)

# SP 500 Total return index
SP = pdr.get_data_yahoo('^SP500TR', start=startdate, end=enddate)

# If you want to use the price index of SP500, you can use the ticker of '^GSPC'
# SP = pdr.get_data_yahoo('^GSPC', start=startdate, end=enddate)
```


```python
SP
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
      <th>Date</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1988-01-04</th>
      <td>256.019989</td>
      <td>256.019989</td>
      <td>256.019989</td>
      <td>256.019989</td>
      <td>0</td>
      <td>256.019989</td>
    </tr>
    <tr>
      <th>1988-01-05</th>
      <td>258.769989</td>
      <td>258.769989</td>
      <td>258.769989</td>
      <td>258.769989</td>
      <td>0</td>
      <td>258.769989</td>
    </tr>
    <tr>
      <th>1988-01-06</th>
      <td>259.029999</td>
      <td>259.029999</td>
      <td>259.029999</td>
      <td>259.029999</td>
      <td>0</td>
      <td>259.029999</td>
    </tr>
    <tr>
      <th>1988-01-07</th>
      <td>261.209991</td>
      <td>261.209991</td>
      <td>261.209991</td>
      <td>261.209991</td>
      <td>0</td>
      <td>261.209991</td>
    </tr>
    <tr>
      <th>1988-01-08</th>
      <td>243.550003</td>
      <td>243.550003</td>
      <td>243.550003</td>
      <td>243.550003</td>
      <td>0</td>
      <td>243.550003</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-08-17</th>
      <td>6954.669922</td>
      <td>6937.540039</td>
      <td>6940.890137</td>
      <td>6943.220215</td>
      <td>0</td>
      <td>6943.220215</td>
    </tr>
    <tr>
      <th>2020-08-18</th>
      <td>6971.020020</td>
      <td>6920.080078</td>
      <td>6954.689941</td>
      <td>6960.299805</td>
      <td>0</td>
      <td>6960.299805</td>
    </tr>
    <tr>
      <th>2020-08-19</th>
      <td>6981.410156</td>
      <td>6920.370117</td>
      <td>6967.069824</td>
      <td>6930.799805</td>
      <td>0</td>
      <td>6930.799805</td>
    </tr>
    <tr>
      <th>2020-08-20</th>
      <td>6963.520020</td>
      <td>6889.850098</td>
      <td>6901.520020</td>
      <td>6952.919922</td>
      <td>0</td>
      <td>6952.919922</td>
    </tr>
    <tr>
      <th>2020-08-21</th>
      <td>6982.970215</td>
      <td>6941.000000</td>
      <td>6954.390137</td>
      <td>6977.270020</td>
      <td>0</td>
      <td>6977.270020</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Generate group by keys
SP['date'] = SP.index
SP['year'] = SP['date'].apply(lambda d: f'{d.year}')

# Group by year, take the adjusted close price of the first business day in each year
df = SP[['year', 'Adj Close']].sort_index().groupby('year').first()

# Generate annual return
df['return'] = df['Adj Close'].pct_change()

# Show annual return by the beginning of next year
df['return_next_y'] = df['return'].shift(-1)

df = df.drop(columns='return')
df = df.reset_index()

df
```

<div>
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
      <td>256.019989</td>
      <td>0.115733</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1989</td>
      <td>285.649994</td>
      <td>0.351864</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1990</td>
      <td>386.160004</td>
      <td>-0.058836</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1991</td>
      <td>363.440002</td>
      <td>0.320273</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1992</td>
      <td>479.839996</td>
      <td>0.075358</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1993</td>
      <td>516.000000</td>
      <td>0.099031</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1994</td>
      <td>567.099976</td>
      <td>0.014848</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1995</td>
      <td>575.520020</td>
      <td>0.386954</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1996</td>
      <td>798.219971</td>
      <td>0.214014</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1997</td>
      <td>969.049988</td>
      <td>0.346721</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1998</td>
      <td>1305.040039</td>
      <td>0.278520</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1999</td>
      <td>1668.520020</td>
      <td>0.199932</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2000</td>
      <td>2002.109985</td>
      <td>-0.108011</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2001</td>
      <td>1785.859985</td>
      <td>-0.088109</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2002</td>
      <td>1628.510010</td>
      <td>-0.199778</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2003</td>
      <td>1303.170044</td>
      <td>0.241626</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2004</td>
      <td>1618.050049</td>
      <td>0.103155</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2005</td>
      <td>1784.959961</td>
      <td>0.075072</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2006</td>
      <td>1918.959961</td>
      <td>0.138075</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2007</td>
      <td>2183.919922</td>
      <td>0.040977</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2008</td>
      <td>2273.409912</td>
      <td>-0.340563</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2009</td>
      <td>1499.170044</td>
      <td>0.245396</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2010</td>
      <td>1867.060059</td>
      <td>0.145277</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2011</td>
      <td>2138.300049</td>
      <td>0.025300</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2012</td>
      <td>2192.399902</td>
      <td>0.171570</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2013</td>
      <td>2568.550049</td>
      <td>0.279590</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2014</td>
      <td>3286.689941</td>
      <td>0.146649</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2015</td>
      <td>3768.679932</td>
      <td>-0.001244</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2016</td>
      <td>3763.989990</td>
      <td>0.146411</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2017</td>
      <td>4315.080078</td>
      <td>0.218119</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2018</td>
      <td>5256.279785</td>
      <td>-0.050553</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2019</td>
      <td>4990.560059</td>
      <td>0.324358</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2020</td>
      <td>6609.290039</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Describe the annual return
df['return_next_y'].describe()
```
Output:

    count    32.000000
    mean      0.120554
    std       0.169035
    min      -0.340563
    25%       0.022687
    50%       0.141676
    75%       0.242569
    max       0.386954
    Name: return_next_y, dtype: float64


When checking the annual return rate, the minimum value is  -34.1% and maximum value is 38.7%.
So we should not be panic if we see a drop of 30-40% in the total value in a certain year. Try to buy more in the dip, and the bull market will follow in 1-3 years.

# Define functions


```python
def calc_dollar_cost_averaging_return(df_dc: pd.DataFrame, value_per_period: float = 1.0, 
                                      increase_investment_per_period: float = 0.0,
                                      display_returns: bool = True) -> pd.DataFrame:
    """
    This function calculates the overall rate of return and the internal rate of return of dollar cost averaging method. 
    :param df_dc: Original dataframe that contains the price at the beginning of each period
    :param value_per_period: Investment value per period
    :param increase_investment_per_period: Increase the investment by x each period
    :param display_returns: Whether to display return results or not
    :return: A dataframe that contains all the relevant info
    """
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
        
        # Calculate overall return
        overall_return = round(df_dc.loc[len(df_dc.index) - 2, 'overall_return_next']*100, 2)
        print(f"The overall return rate is {overall_return}%")

        # Calculate IRR
        cf = (-df_dc['should_invest']).tolist()
        cf[-1] = df_dc.loc[len(df_dc.index) - 2, 'total_value_next']
        print("The cash flow (the negative sign stands for investment):")
        display(cf)
        irr = round(100*np.irr(cf), 2)
        print(f"The internal return rate (IRR) is {irr}% per period")
        
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

# Plan 1: Yearly [dollar cost averaging](https://en.wikipedia.org/wiki/Dollar_cost_averaging) buying method

Imagine we were able to invest 1200 dollar per year into a SP500 index based ETF starting from 1988.

What's the overall return after 32 years?

What's the average annual return?


```python
df1 = calc_dollar_cost_averaging_return(df, value_per_period=1200)
```

Output:
    
    The overall return rate is 580.19%
    The cash flow (the negative sign stands for investment):
    
    [-1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     -1200.0,
     261192.11446741945]
    
    The internal return rate (IRR) is 9.92% per period
    The compound annual growth rate (CAGR) is 6.17% per period



df1:



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
      <td>256.019989</td>
      <td>0.115733</td>
      <td>1200.0</td>
      <td>1200.0</td>
      <td>1338.879804</td>
      <td>0.115733</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1989</td>
      <td>285.649994</td>
      <td>0.351864</td>
      <td>1200.0</td>
      <td>2400.0</td>
      <td>3432.220743</td>
      <td>0.430092</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1990</td>
      <td>386.160004</td>
      <td>-0.058836</td>
      <td>1200.0</td>
      <td>3600.0</td>
      <td>4359.680708</td>
      <td>0.211022</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1991</td>
      <td>363.440002</td>
      <td>0.320273</td>
      <td>1200.0</td>
      <td>4800.0</td>
      <td>7340.295930</td>
      <td>0.529228</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1992</td>
      <td>479.839996</td>
      <td>0.075358</td>
      <td>1200.0</td>
      <td>6000.0</td>
      <td>9183.879488</td>
      <td>0.530647</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1993</td>
      <td>516.000000</td>
      <td>0.099031</td>
      <td>1200.0</td>
      <td>7200.0</td>
      <td>11412.205047</td>
      <td>0.585028</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1994</td>
      <td>567.099976</td>
      <td>0.014848</td>
      <td>1200.0</td>
      <td>8400.0</td>
      <td>12799.465363</td>
      <td>0.523746</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1995</td>
      <td>575.520020</td>
      <td>0.386954</td>
      <td>1200.0</td>
      <td>9600.0</td>
      <td>19416.618801</td>
      <td>1.022564</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1996</td>
      <td>798.219971</td>
      <td>0.214014</td>
      <td>1200.0</td>
      <td>10800.0</td>
      <td>25028.857872</td>
      <td>1.317487</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1997</td>
      <td>969.049988</td>
      <td>0.346721</td>
      <td>1200.0</td>
      <td>12000.0</td>
      <td>35322.955610</td>
      <td>1.943580</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1998</td>
      <td>1305.040039</td>
      <td>0.278520</td>
      <td>1200.0</td>
      <td>13200.0</td>
      <td>46695.335609</td>
      <td>2.537525</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1999</td>
      <td>1668.520020</td>
      <td>0.199932</td>
      <td>1200.0</td>
      <td>14400.0</td>
      <td>57471.129235</td>
      <td>2.991051</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2000</td>
      <td>2002.109985</td>
      <td>-0.108011</td>
      <td>1200.0</td>
      <td>15600.0</td>
      <td>52333.999013</td>
      <td>2.354744</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2001</td>
      <td>1785.859985</td>
      <td>-0.088109</td>
      <td>1200.0</td>
      <td>16800.0</td>
      <td>48817.182741</td>
      <td>1.905785</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2002</td>
      <td>1628.510010</td>
      <td>-0.199778</td>
      <td>1200.0</td>
      <td>18000.0</td>
      <td>40024.865577</td>
      <td>1.223604</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2003</td>
      <td>1303.170044</td>
      <td>0.241626</td>
      <td>1200.0</td>
      <td>19200.0</td>
      <td>51185.872534</td>
      <td>1.665931</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2004</td>
      <td>1618.050049</td>
      <td>0.103155</td>
      <td>1200.0</td>
      <td>20400.0</td>
      <td>57789.735898</td>
      <td>1.832830</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2005</td>
      <td>1784.959961</td>
      <td>0.075072</td>
      <td>1200.0</td>
      <td>21600.0</td>
      <td>63418.196358</td>
      <td>1.936028</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2006</td>
      <td>1918.959961</td>
      <td>0.138075</td>
      <td>1200.0</td>
      <td>22800.0</td>
      <td>73540.339150</td>
      <td>2.225453</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2007</td>
      <td>2183.919922</td>
      <td>0.040977</td>
      <td>1200.0</td>
      <td>24000.0</td>
      <td>77802.957039</td>
      <td>2.241790</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2008</td>
      <td>2273.409912</td>
      <td>-0.340563</td>
      <td>1200.0</td>
      <td>25200.0</td>
      <td>52097.453233</td>
      <td>1.067359</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2009</td>
      <td>1499.170044</td>
      <td>0.245396</td>
      <td>1200.0</td>
      <td>26400.0</td>
      <td>66376.423781</td>
      <td>1.514258</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2010</td>
      <td>1867.060059</td>
      <td>0.145277</td>
      <td>1200.0</td>
      <td>27600.0</td>
      <td>77393.691544</td>
      <td>1.804119</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2011</td>
      <td>2138.300049</td>
      <td>0.025300</td>
      <td>1200.0</td>
      <td>28800.0</td>
      <td>80582.143634</td>
      <td>1.797991</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2012</td>
      <td>2192.399902</td>
      <td>0.171570</td>
      <td>1200.0</td>
      <td>30000.0</td>
      <td>95813.509570</td>
      <td>2.193784</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2013</td>
      <td>2568.550049</td>
      <td>0.279590</td>
      <td>1200.0</td>
      <td>31200.0</td>
      <td>124137.478352</td>
      <td>2.978765</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2014</td>
      <td>3286.689941</td>
      <td>0.146649</td>
      <td>1200.0</td>
      <td>32400.0</td>
      <td>143718.101728</td>
      <td>3.435744</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2015</td>
      <td>3768.679932</td>
      <td>-0.001244</td>
      <td>1200.0</td>
      <td>33600.0</td>
      <td>144737.758102</td>
      <td>3.307671</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2016</td>
      <td>3763.989990</td>
      <td>0.146411</td>
      <td>1200.0</td>
      <td>34800.0</td>
      <td>167304.672506</td>
      <td>3.807606</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2017</td>
      <td>4315.080078</td>
      <td>0.218119</td>
      <td>1200.0</td>
      <td>36000.0</td>
      <td>205258.694569</td>
      <td>4.701630</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2018</td>
      <td>5256.279785</td>
      <td>-0.050553</td>
      <td>1200.0</td>
      <td>37200.0</td>
      <td>196021.626888</td>
      <td>4.269399</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2019</td>
      <td>4990.560059</td>
      <td>0.324358</td>
      <td>1200.0</td>
      <td>38400.0</td>
      <td>261192.114467</td>
      <td>5.801878</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2020</td>
      <td>6609.290039</td>
      <td>NaN</td>
      <td>1200.0</td>
      <td>39600.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_changes(df1, title='Plan 1', xlabel='year')
```


![png](/assets/3_output_14_0.png "Plan 1"){:width="80%"}


# Plan 2: Increase the investment each year by 2%
In Plan 2, let's assume the initial_investment is 1200, and then we increase the investment each year by 2%.


```python
# Assuming initial_investment is 1200, and then increase investment each year by 2%
df2 = calc_dollar_cost_averaging_return(df, value_per_period=1200, increase_investment_per_period=0.02)
```
Output:

    The overall return rate is 486.51%
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
    The compound annual growth rate (CAGR) is 5.68% per period


df2:


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
      <td>256.019989</td>
      <td>0.115733</td>
      <td>1200.000000</td>
      <td>1200.000000</td>
      <td>1338.879804</td>
      <td>0.115733</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1989</td>
      <td>285.649994</td>
      <td>0.351864</td>
      <td>1224.000000</td>
      <td>2424.000000</td>
      <td>3464.665484</td>
      <td>0.429317</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1990</td>
      <td>386.160004</td>
      <td>-0.058836</td>
      <td>1248.480000</td>
      <td>3672.480000</td>
      <td>4435.844184</td>
      <td>0.207861</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1991</td>
      <td>363.440002</td>
      <td>0.320273</td>
      <td>1273.449600</td>
      <td>4945.929600</td>
      <td>7537.826024</td>
      <td>0.524046</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1992</td>
      <td>479.839996</td>
      <td>0.075358</td>
      <td>1298.918592</td>
      <td>6244.848192</td>
      <td>9502.668090</td>
      <td>0.521681</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1993</td>
      <td>516.000000</td>
      <td>0.099031</td>
      <td>1324.896964</td>
      <td>7569.745156</td>
      <td>11899.829221</td>
      <td>0.572025</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1994</td>
      <td>567.099976</td>
      <td>0.014848</td>
      <td>1351.394903</td>
      <td>8921.140059</td>
      <td>13447.972306</td>
      <td>0.507428</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1995</td>
      <td>575.520020</td>
      <td>0.386954</td>
      <td>1378.422801</td>
      <td>10299.562860</td>
      <td>20563.532573</td>
      <td>0.996544</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1996</td>
      <td>798.219971</td>
      <td>0.214014</td>
      <td>1405.991257</td>
      <td>11705.554117</td>
      <td>26671.303126</td>
      <td>1.278517</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1997</td>
      <td>969.049988</td>
      <td>0.346721</td>
      <td>1434.111082</td>
      <td>13139.665200</td>
      <td>37850.153571</td>
      <td>1.880603</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1998</td>
      <td>1305.040039</td>
      <td>0.278520</td>
      <td>1462.793304</td>
      <td>14602.458504</td>
      <td>50262.395731</td>
      <td>2.442050</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1999</td>
      <td>1668.520020</td>
      <td>0.199932</td>
      <td>1492.049170</td>
      <td>16094.507674</td>
      <td>62101.796628</td>
      <td>2.858571</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2000</td>
      <td>2002.109985</td>
      <td>-0.108011</td>
      <td>1521.890153</td>
      <td>17616.397827</td>
      <td>56751.625622</td>
      <td>2.221523</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2001</td>
      <td>1785.859985</td>
      <td>-0.088109</td>
      <td>1552.327957</td>
      <td>19168.725784</td>
      <td>53166.862347</td>
      <td>1.773625</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2002</td>
      <td>1628.510010</td>
      <td>-0.199778</td>
      <td>1583.374516</td>
      <td>20752.100299</td>
      <td>43812.361085</td>
      <td>1.111225</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2003</td>
      <td>1303.170044</td>
      <td>0.241626</td>
      <td>1615.042006</td>
      <td>22367.142305</td>
      <td>56403.853151</td>
      <td>1.521728</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2004</td>
      <td>1618.050049</td>
      <td>0.103155</td>
      <td>1647.342846</td>
      <td>24014.485152</td>
      <td>64039.465661</td>
      <td>1.666702</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2005</td>
      <td>1784.959961</td>
      <td>0.075072</td>
      <td>1680.289703</td>
      <td>25694.774855</td>
      <td>70653.449907</td>
      <td>1.749721</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2006</td>
      <td>1918.959961</td>
      <td>0.138075</td>
      <td>1713.895497</td>
      <td>27408.670352</td>
      <td>82359.450191</td>
      <td>2.004869</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2007</td>
      <td>2183.919922</td>
      <td>0.040977</td>
      <td>1748.173407</td>
      <td>29156.843759</td>
      <td>87554.082572</td>
      <td>2.002866</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2008</td>
      <td>2273.409912</td>
      <td>-0.340563</td>
      <td>1783.136875</td>
      <td>30939.980634</td>
      <td>58912.245650</td>
      <td>0.904082</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2009</td>
      <td>1499.170044</td>
      <td>0.245396</td>
      <td>1818.799613</td>
      <td>32758.780247</td>
      <td>75634.187986</td>
      <td>1.308822</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2010</td>
      <td>1867.060059</td>
      <td>0.145277</td>
      <td>1855.175605</td>
      <td>34613.955851</td>
      <td>88746.748765</td>
      <td>1.563901</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2011</td>
      <td>2138.300049</td>
      <td>0.025300</td>
      <td>1892.279117</td>
      <td>36506.234969</td>
      <td>92932.231838</td>
      <td>1.545654</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2012</td>
      <td>2192.399902</td>
      <td>0.171570</td>
      <td>1930.124699</td>
      <td>38436.359668</td>
      <td>111137.895169</td>
      <td>1.891478</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2013</td>
      <td>2568.550049</td>
      <td>0.279590</td>
      <td>1968.727193</td>
      <td>40405.086861</td>
      <td>144730.058188</td>
      <td>2.581976</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2014</td>
      <td>3286.689941</td>
      <td>0.146649</td>
      <td>2008.101737</td>
      <td>42413.188598</td>
      <td>168257.173136</td>
      <td>2.967096</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2015</td>
      <td>3768.679932</td>
      <td>-0.001244</td>
      <td>2048.263772</td>
      <td>44461.452370</td>
      <td>170093.499961</td>
      <td>2.825640</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2016</td>
      <td>3763.989990</td>
      <td>0.146411</td>
      <td>2089.229047</td>
      <td>46550.681418</td>
      <td>197392.199679</td>
      <td>3.240372</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2017</td>
      <td>4315.080078</td>
      <td>0.218119</td>
      <td>2131.013628</td>
      <td>48681.695046</td>
      <td>243042.959525</td>
      <td>3.992492</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2018</td>
      <td>5256.279785</td>
      <td>-0.050553</td>
      <td>2173.633901</td>
      <td>50855.328947</td>
      <td>232820.204951</td>
      <td>3.578089</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2019</td>
      <td>4990.560059</td>
      <td>0.324358</td>
      <td>2217.106579</td>
      <td>53072.435526</td>
      <td>311273.633352</td>
      <td>4.865072</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2020</td>
      <td>6609.290039</td>
      <td>NaN</td>
      <td>2261.448711</td>
      <td>55333.884237</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


```python
plot_changes(df2, title='Plan 2', xlabel='year')
```


![png](/assets/3_output_18_0.png "Plan 2"){:width="80%"}


# Conclusion

The compound annual growth rate of Plan 2 is lower than that of Plan 1. This is because Plan 2 invested more money at the later stage of the investment. But, in both plans, the compound annual growth rates are more than 5%.

The internal return rate of both plans are quite similiar to each other (almost 10%). 

It seems that SP500 is a good enough index that can bring us long term return. However, the historical data cannot predict the future. Please do your own research before investing.

In the next notebook, we will further discuss which strategy can further increase IRR.


