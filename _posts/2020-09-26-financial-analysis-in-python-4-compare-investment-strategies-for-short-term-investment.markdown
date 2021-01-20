---
layout: post
title: "Financial Analysis in Python #4: Compare Investment Strategies for Short-term Investment"
date: 2020-09-26T16:47:58+02:00
tags: [Finance, Python, visualization, long-term investment, retirement]
categories: [Finance, Data Science]
---

# Overview

This is the fourth blog post of my **Finaical Analysis in Python** series.

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

----

# Introduction


Imagine that we have a fixed amount of initial investment and we want to set up a short-term investment plan.

We can either decide to invest **all at once** into the stock market at the beginning of each year, or apply the **dollar-cost averaging** or **value averaging** on a **monthly** or **quarterly** basis. 

Which method could maximize our investment return?

Let's do some backtests using the SP500 index performance data. 


```python
# Import libraries
import datetime

from matplotlib import pyplot as plt
import pandas as pd
from pandas_datareader import data as pdr

plt.style.use('bmh')
```


```python
# Get historical data
startdate = datetime.datetime(1988, 1, 1)
enddate = datetime.datetime(2021, 1, 1)

# SP 500 total return index
SP = pdr.get_data_yahoo('^SP500TR', start=startdate, end=enddate)
SP['date'] = SP.index
```

----

# Compare all-in with [dollar cost averaging](https://en.wikipedia.org/wiki/Dollar_cost_averaging)

Let's first assume that the timeframe for the short-term investment plan is one year and the investment should start at the beginning of each year. Hence. each year between 1980 and 2020 is an individual sample. 

In this section, I will compare the annual return rates for these investment plans:

- Plan 1: All-in approach at the beginning of a year
- Plan 2: Divide the initial investment into 12 folds and invest at the beginning of each month
- Plan 3: Divide the initial investment into 4 folds and invest at the beginning of each quarter 


```python
# Plan 1: All-in
# Group by year
SP['date'] = SP.index
SP['year'] = SP['date'].apply(lambda d: f'{d.year}')
df = SP[['year', 'Adj Close']].groupby('year').first()
df['return'] = df['Adj Close'].pct_change()
df['return_next_y'] = df['return'].shift(-1)
```


```python
# Get a feeling of the return rate per year
df['return_next_y'].hist(bins=20);
```


![png](/assets/5_output_6_0.png)



```python
# Plan 2: Monthly dollar-cost averaging
# Group by month
SP['month'] = SP['date'].apply(lambda d: f'{d.month:02d}')
df_m = SP[['year', 'month', 'Adj Close']].groupby(['year','month']).first()
df_m['return'] = df_m['Adj Close'].pct_change()
df_m['return_next'] = df_m['return'].shift(-1)
```


```python
# Plan 3: Quarterly dollar-cost averaging
# Group by quarter
SP['quarter'] = SP['month'].apply(lambda m: int(m)//4 + 1)
df_q = SP[['year', 'quarter', 'Adj Close']].groupby(['year','quarter']).first()
df_q['return'] = df_q['Adj Close'].pct_change()
df_q['return_next'] = df_q['return'].shift(-1)
```

Here is how the dataframe for the quarterly dollar-cost averaging looks like:


```python
df_q.head()
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
      <th>year</th>
      <th>quarter</th>
      <th>Adj Close</th>
      <th>return</th>
      <th>return_next</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">1988</th>
      <th>1</th>
      <td>256.019989</td>
      <td>NaN</td>
      <td>0.009335</td>
    </tr>
    <tr>
      <th>2</th>
      <td>258.410004</td>
      <td>0.009335</td>
      <td>0.074881</td>
    </tr>
    <tr>
      <th>3</th>
      <td>277.760010</td>
      <td>0.074881</td>
      <td>0.015085</td>
    </tr>
    <tr>
      <th>4</th>
      <td>281.950012</td>
      <td>0.015085</td>
      <td>0.013123</td>
    </tr>
    <tr>
      <th>1989</th>
      <th>1</th>
      <td>285.649994</td>
      <td>0.013123</td>
      <td>0.085979</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Function for calcuting dollar-cost averaging returns
def calc_return_over_periods_list(
    investment_per_period_list: list, 
    return_per_period_list: list, 
    n_period: int, 
    print_values: bool = False) -> float:
    
    """
    This function calculates the overall rate of return of the dollar-cost averaging strategy.
    :param investment_per_period_list: Investment value at the beginning of each period.
    :param return_per_period_list: Historical return rate per period.
    :param n_period: number of periods for calculating the overall return rate.
    :param print_values: Whether to display the value each period and overall return rate or not
    :return: A dataframe that contains all the relevant info
    """
    
    value = 0 
    
    for i in range(n_period):       
        value += investment_per_period_list[i]
        value = value * (1 + return_per_period_list[i])
        if print_values:
            print(f'Peorid {i}: value = {round(value, 2)}')
            
    return_rate = value/sum(investment_per_period_list) - 1
    
    if print_values:
        print(f'The overall return rate is {round(return_rate*100,2)}%')
    
    return return_rate
```

Here is how the function works for calculating the overall return for a dollar-cost averaging strategy:



```python
# Assuming investing 300 dollars at the beginning of each quarter
calc_return_over_periods_list(
    investment_per_period_list = [300]*4, 
    return_per_period_list = df_q.loc['2019','return_next'].tolist(), 
    n_period=4, 
    print_values=True)
```

Output:

    Peorid 0: value = 344.45
    Peorid 1: value = 668.05
    Peorid 2: value = 1027.87
    Peorid 3: value = 1391.54
        
    The overall return rate is 15.96%.

---
```python
# Join dollar-cost averaging results together into one dataframe
df = df.drop(columns='return')

for year in SP.year.unique()[:-1]:
    # for monthly dollar-cost averaging annual return
    df.loc[f'{int(year)}', 'return_next_m'] = (
    calc_return_over_periods_list(
        [1]*12, 
        df_m.loc[(f'{year}'),'return_next'].tolist(), 
        12)
    )
    
    # for quarterly dollar-cost averaging annual return
    df.loc[f'{int(year)}', 'return_next_q'] = (
    calc_return_over_periods_list(
        [1]*4, 
        df_q.loc[(f'{year}'),'return_next'].tolist(), 
        4)
    )
    
df.round(3)
```

<div>
<table>
  <thead>
    <tr style="text-align: right;">
      <th>year</th>
      <th>Adj Close</th>
      <th>return_next_y</th>
      <th>return_next_m</th>
      <th>return_next_q</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1988</th>
      <td>256.02</td>
      <td>0.116</td>
      <td>0.059</td>
      <td>0.066</td>
    </tr>
    <tr>
      <th>1989</th>
      <td>285.65</td>
      <td>0.352</td>
      <td>0.152</td>
      <td>0.172</td>
    </tr>
    <tr>
      <th>1990</th>
      <td>386.16</td>
      <td>-0.059</td>
      <td>-0.008</td>
      <td>-0.031</td>
    </tr>
    <tr>
      <th>1991</th>
      <td>363.44</td>
      <td>0.320</td>
      <td>0.137</td>
      <td>0.165</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>479.84</td>
      <td>0.075</td>
      <td>0.064</td>
      <td>0.057</td>
    </tr>
    <tr>
      <th>1993</th>
      <td>516.00</td>
      <td>0.099</td>
      <td>0.047</td>
      <td>0.053</td>
    </tr>
    <tr>
      <th>1994</th>
      <td>567.10</td>
      <td>0.015</td>
      <td>0.015</td>
      <td>0.029</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>575.52</td>
      <td>0.387</td>
      <td>0.188</td>
      <td>0.198</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>798.22</td>
      <td>0.214</td>
      <td>0.121</td>
      <td>0.120</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>969.05</td>
      <td>0.347</td>
      <td>0.151</td>
      <td>0.172</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>1305.04</td>
      <td>0.279</td>
      <td>0.159</td>
      <td>0.139</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>1668.52</td>
      <td>0.200</td>
      <td>0.118</td>
      <td>0.120</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>2002.11</td>
      <td>-0.108</td>
      <td>-0.101</td>
      <td>-0.094</td>
    </tr>
    <tr>
      <th>2001</th>
      <td>1785.86</td>
      <td>-0.088</td>
      <td>-0.026</td>
      <td>-0.023</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>1628.51</td>
      <td>-0.200</td>
      <td>-0.078</td>
      <td>-0.097</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>1303.17</td>
      <td>0.242</td>
      <td>0.179</td>
      <td>0.182</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>1618.05</td>
      <td>0.103</td>
      <td>0.074</td>
      <td>0.071</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>1784.96</td>
      <td>0.075</td>
      <td>0.062</td>
      <td>0.053</td>
    </tr>
    <tr>
      <th>2006</th>
      <td>1918.96</td>
      <td>0.138</td>
      <td>0.096</td>
      <td>0.096</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>2183.92</td>
      <td>0.041</td>
      <td>-0.008</td>
      <td>0.013</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>2273.41</td>
      <td>-0.341</td>
      <td>-0.229</td>
      <td>-0.189</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>1499.17</td>
      <td>0.245</td>
      <td>0.249</td>
      <td>0.207</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>1867.06</td>
      <td>0.145</td>
      <td>0.141</td>
      <td>0.109</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>2138.30</td>
      <td>0.025</td>
      <td>0.018</td>
      <td>0.007</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>2192.40</td>
      <td>0.172</td>
      <td>0.079</td>
      <td>0.084</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>2568.55</td>
      <td>0.280</td>
      <td>0.145</td>
      <td>0.143</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>3286.69</td>
      <td>0.147</td>
      <td>0.087</td>
      <td>0.085</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>3768.68</td>
      <td>-0.001</td>
      <td>-0.009</td>
      <td>-0.020</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>3763.99</td>
      <td>0.146</td>
      <td>0.094</td>
      <td>0.084</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>4315.08</td>
      <td>0.218</td>
      <td>0.121</td>
      <td>0.124</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>5256.28</td>
      <td>-0.051</td>
      <td>-0.078</td>
      <td>-0.066</td>
    </tr>
    <tr>
      <th>2019</th>
      <td>4990.56</td>
      <td>0.324</td>
      <td>0.150</td>
      <td>0.160</td>
    </tr>
    <tr>
      <th>2020</th>
      <td>6609.29</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



With the dataframe above, we can easily compare the performance of these three short-term investment plans: 


```python
# Return_next_y is the annual return for all-in method
# Return_next_m is the annual return of the monthly dollar cost averaging method
# Return_next_q is the annual return of the quarterly dollar cost averaging method 
df[['return_next_y','return_next_m','return_next_q']].describe().round(3)
```




<div>
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>return_next_y</th>
      <th>return_next_m</th>
      <th>return_next_q</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>32.000</td>
      <td>32.000</td>
      <td>32.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.121</td>
      <td>0.068</td>
      <td>0.068</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.169</td>
      <td>0.099</td>
      <td>0.096</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.341</td>
      <td>-0.229</td>
      <td>-0.189</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.023</td>
      <td>0.009</td>
      <td>0.012</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.142</td>
      <td>0.083</td>
      <td>0.084</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.243</td>
      <td>0.142</td>
      <td>0.140</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.387</td>
      <td>0.249</td>
      <td>0.207</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(15,5))
plt.plot(df['return_next_y'], label='all-in')
plt.plot(df['return_next_m'], label='dollar-cost (monthly)')
plt.plot(df['return_next_q'], label='dollar-cost (quarterly)')
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('year')
plt.ylabel('return rate')
plt.xticks(rotation=90)
plt.title('Annual return rate of the all-in vs. dollar-cost average methods')
plt.legend()
plt.show()
```


![png](/assets/5_output_17_0.png)


## Observation


Given that you already have a fixed amount of initial investment and you want to invest it into an ETF based on SP500, all-in method performed better than both monthly and quarterly dollar-cost averaging methods in most of the past years. 

This is because SP500 has a tendency of increasing in most of the past years; and the all-in method maximized the gain. 

However, dollar-cost averaging method helps to reduce loss in a bear market.

It seems that only in the following years SP500 has a negative year-over-year return. 


```python
list(df.loc[df['return_next_y']<=0,:].index)
```




    ['1990', '2000', '2001', '2002', '2008', '2015', '2018']



The bear markets does not seem to last long using the SP 500 index (usually less than 2 years).

So, it might be a good time to start investing if the last year was a bear market.

If it is already a bull market, you can decide whether you still want to risk it with the all-in method.

If it is already a bear market, it might be a good time to start investing.

----

# Compare with [value averaging](https://en.wikipedia.org/wiki/Value_averaging)

In the last notebook, we already discussed the method of value averaging for long term investment, and we assumed that it might work for short term investment. 

Let's check the annual return of the value averaging method here:


```python
# This is an adjusted version of the function that we have introduced in the last notebook for value averaging
def calc_value_averaging_return(df: pd.DataFrame, value_per_period: float = 1.0, 
                                increase_investment_per_period: float = 0.0) -> float:
    """
    This function calculates the overall return rate of the value averaging strategy.
    :param df: Original dataframe that contains the price at the beginning of each period
    :param value_per_period: Investment value per period
    :param increase_investment_per_period: Increase the investment by x each period
    :return: overall return rate (might be negative if the total withdraw amount is too much)
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

    return df_va.loc[len(df_va.index)-1,'target_value']/df_va.loc[len(df_va.index)-1,'total_invest_or_withdraw']-1
```


```python
# Assuming fixed investment value each year
value_per_year = 1200
value_per_quarter = value_per_year/4
value_per_month = value_per_year/12

df_va_m = df_m['Adj Close'].reset_index()
df_va_q = df_q['Adj Close'].reset_index()

### Join results together into one dataframe
for year in SP.year.unique()[:-1]:
    df_va_m_sample = df_va_m.loc[df_va_m['year']==year,:].reset_index()
    df_va_q_sample = df_va_q.loc[df_va_q['year']==year,:].reset_index()
    annual_return_va_m = calc_value_averaging_return(df_va_m_sample, value_per_month)
    annual_return_va_q = calc_value_averaging_return(df_va_q_sample, value_per_quarter)
    df.loc[f'{int(year)}', 'return_va_m'] = annual_return_va_m
    df.loc[f'{int(year)}', 'return_va_q'] = annual_return_va_q
```


```python
# Join value averaging with dollar cost averaging results
df.round(3)
```




<div>
<table>
  <thead>
    <tr style="text-align: right;">
      <th>year</th>
      <th>Adj Close</th>
      <th>return_next_y</th>
      <th>return_next_m</th>
      <th>return_next_q</th>
      <th>return_va_m</th>
      <th>return_va_q</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1988</th>
      <td>256.02</td>
      <td>0.116</td>
      <td>0.059</td>
      <td>0.066</td>
      <td>0.049</td>
      <td>0.054</td>
    </tr>
    <tr>
      <th>1989</th>
      <td>285.65</td>
      <td>0.352</td>
      <td>0.152</td>
      <td>0.172</td>
      <td>0.128</td>
      <td>0.152</td>
    </tr>
    <tr>
      <th>1990</th>
      <td>386.16</td>
      <td>-0.059</td>
      <td>-0.008</td>
      <td>-0.031</td>
      <td>-0.011</td>
      <td>-0.037</td>
    </tr>
    <tr>
      <th>1991</th>
      <td>363.44</td>
      <td>0.320</td>
      <td>0.137</td>
      <td>0.165</td>
      <td>0.037</td>
      <td>0.065</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>479.84</td>
      <td>0.075</td>
      <td>0.064</td>
      <td>0.057</td>
      <td>0.052</td>
      <td>0.045</td>
    </tr>
    <tr>
      <th>1993</th>
      <td>516.00</td>
      <td>0.099</td>
      <td>0.047</td>
      <td>0.053</td>
      <td>0.038</td>
      <td>0.044</td>
    </tr>
    <tr>
      <th>1994</th>
      <td>567.10</td>
      <td>0.015</td>
      <td>0.015</td>
      <td>0.029</td>
      <td>-0.008</td>
      <td>0.005</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>575.52</td>
      <td>0.387</td>
      <td>0.188</td>
      <td>0.198</td>
      <td>0.171</td>
      <td>0.188</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>798.22</td>
      <td>0.214</td>
      <td>0.121</td>
      <td>0.120</td>
      <td>0.165</td>
      <td>0.170</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>969.05</td>
      <td>0.347</td>
      <td>0.151</td>
      <td>0.172</td>
      <td>0.161</td>
      <td>0.194</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>1305.04</td>
      <td>0.279</td>
      <td>0.159</td>
      <td>0.139</td>
      <td>0.128</td>
      <td>0.094</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>1668.52</td>
      <td>0.200</td>
      <td>0.118</td>
      <td>0.120</td>
      <td>0.080</td>
      <td>0.078</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>2002.11</td>
      <td>-0.108</td>
      <td>-0.101</td>
      <td>-0.094</td>
      <td>-0.072</td>
      <td>-0.068</td>
    </tr>
    <tr>
      <th>2001</th>
      <td>1785.86</td>
      <td>-0.088</td>
      <td>-0.026</td>
      <td>-0.023</td>
      <td>-0.040</td>
      <td>-0.041</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>1628.51</td>
      <td>-0.200</td>
      <td>-0.078</td>
      <td>-0.097</td>
      <td>-0.052</td>
      <td>-0.062</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>1303.17</td>
      <td>0.242</td>
      <td>0.179</td>
      <td>0.182</td>
      <td>0.145</td>
      <td>0.156</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>1618.05</td>
      <td>0.103</td>
      <td>0.074</td>
      <td>0.071</td>
      <td>0.067</td>
      <td>0.064</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>1784.96</td>
      <td>0.075</td>
      <td>0.062</td>
      <td>0.053</td>
      <td>0.062</td>
      <td>0.050</td>
    </tr>
    <tr>
      <th>2006</th>
      <td>1918.96</td>
      <td>0.138</td>
      <td>0.096</td>
      <td>0.096</td>
      <td>0.083</td>
      <td>0.086</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>2183.92</td>
      <td>0.041</td>
      <td>-0.008</td>
      <td>0.013</td>
      <td>0.009</td>
      <td>0.029</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>2273.41</td>
      <td>-0.341</td>
      <td>-0.229</td>
      <td>-0.189</td>
      <td>-0.275</td>
      <td>-0.236</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>1499.17</td>
      <td>0.245</td>
      <td>0.249</td>
      <td>0.207</td>
      <td>0.255</td>
      <td>0.216</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>1867.06</td>
      <td>0.145</td>
      <td>0.141</td>
      <td>0.109</td>
      <td>0.093</td>
      <td>0.054</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>2138.30</td>
      <td>0.025</td>
      <td>0.018</td>
      <td>0.007</td>
      <td>-0.001</td>
      <td>-0.019</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>2192.40</td>
      <td>0.172</td>
      <td>0.079</td>
      <td>0.084</td>
      <td>0.041</td>
      <td>0.044</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>2568.55</td>
      <td>0.280</td>
      <td>0.145</td>
      <td>0.143</td>
      <td>0.134</td>
      <td>0.131</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>3286.69</td>
      <td>0.147</td>
      <td>0.087</td>
      <td>0.085</td>
      <td>0.088</td>
      <td>0.085</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>3768.68</td>
      <td>-0.001</td>
      <td>-0.009</td>
      <td>-0.020</td>
      <td>0.040</td>
      <td>0.022</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>3763.99</td>
      <td>0.146</td>
      <td>0.094</td>
      <td>0.084</td>
      <td>0.063</td>
      <td>0.052</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>4315.08</td>
      <td>0.218</td>
      <td>0.121</td>
      <td>0.124</td>
      <td>0.102</td>
      <td>0.107</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>5256.28</td>
      <td>-0.051</td>
      <td>-0.078</td>
      <td>-0.066</td>
      <td>0.027</td>
      <td>0.039</td>
    </tr>
    <tr>
      <th>2019</th>
      <td>4990.56</td>
      <td>0.324</td>
      <td>0.150</td>
      <td>0.160</td>
      <td>0.104</td>
      <td>0.113</td>
    </tr>
    <tr>
      <th>2020</th>
      <td>6609.29</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(15,5))
plt.plot(df['return_next_y'], label='all-in')
plt.plot(df['return_va_m'], label='value-averaging (monthly)')
# plt.plot(df['return_va_q'], label='value-averaging (quarterly)')
plt.plot(df['return_next_m'], label='dollar-cost (monthly)')
# plt.plot(df['return_next_q'], label='dollar-cost (quarterly)')
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('year')
plt.ylabel('return rate')
plt.xticks(rotation=90)
plt.title('Annual return rate of the all-in, monthly dollar-cost averaging, and monthly value averaging methods')
plt.legend()
plt.show()
```


![png](/assets/5_output_26_0.png)



```python
def compare_strategy(df, str1, str2, str1_name, str2_name):
    print(f'number of times {str1_name} is better than {str2_name}: {df.loc[df[str1] > df[str2],:].shape[0]}')
    print(f'number of times {str1_name} is equal to {str2_name}: {df.loc[df[str1] == df[str2],:].shape[0]}')
    print(f'number of times {str1_name} is worse than {str2_name}: {df.loc[df[str1] < df[str2],:].shape[0]}')
```

----

```python
compare_strategy(df, 'return_va_m', 'return_next_m', 'monthly value-averaging', 'monthly dollar-cost-averaging')
```

Output:

    number of times monthly value-averaging is better than monthly dollar-cost-averaging: 9
    number of times monthly value-averaging is equal to monthly dollar-cost-averaging: 0
    number of times monthly value-averaging is worse than monthly dollar-cost-averaging: 23

----

```python
compare_strategy(df, 'return_next_y', 'return_next_m', 'all-in', 'monthly dollar-cost-averaging')
```

Output:

    number of times all-in is better than monthly dollar-cost-averaging: 26
    number of times all-in is equal to monthly dollar-cost-averaging: 0
    number of times all-in is worse than monthly dollar-cost-averaging: 6


## Observation

It seems that the annual return of the dollar-cost averaging and value averaging did not differ much, even though the monthly value averaging method was worse than the monthly dollar-cost-averaging in most years.

Again, all-in method brings the most gain but also involves the most risk.

----

# What if we use a multi-year timeframe?

The test below involves a multiple year short-term investment plan. The rate of return below indicates the overall rate of return of investing for x years starting from a certain year.

Let's see if using dollar cost averaging or value averaging helps to improve the peformance.


```python
SP
```

<div>
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
      <th>date</th>
      <th>year</th>
      <th>month</th>
      <th>quarter</th>
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
      <td>1988-01-04</td>
      <td>1988</td>
      <td>01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1988-01-05</th>
      <td>258.769989</td>
      <td>258.769989</td>
      <td>258.769989</td>
      <td>258.769989</td>
      <td>0</td>
      <td>258.769989</td>
      <td>1988-01-05</td>
      <td>1988</td>
      <td>01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1988-01-06</th>
      <td>259.029999</td>
      <td>259.029999</td>
      <td>259.029999</td>
      <td>259.029999</td>
      <td>0</td>
      <td>259.029999</td>
      <td>1988-01-06</td>
      <td>1988</td>
      <td>01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1988-01-07</th>
      <td>261.209991</td>
      <td>261.209991</td>
      <td>261.209991</td>
      <td>261.209991</td>
      <td>0</td>
      <td>261.209991</td>
      <td>1988-01-07</td>
      <td>1988</td>
      <td>01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1988-01-08</th>
      <td>243.550003</td>
      <td>243.550003</td>
      <td>243.550003</td>
      <td>243.550003</td>
      <td>0</td>
      <td>243.550003</td>
      <td>1988-01-08</td>
      <td>1988</td>
      <td>01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-09-21</th>
      <td>6757.350098</td>
      <td>6641.330078</td>
      <td>6757.350098</td>
      <td>6748.080078</td>
      <td>0</td>
      <td>6748.080078</td>
      <td>2020-09-21</td>
      <td>2020</td>
      <td>09</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2020-09-22</th>
      <td>6828.669922</td>
      <td>6727.319824</td>
      <td>6778.330078</td>
      <td>6819.080078</td>
      <td>0</td>
      <td>6819.080078</td>
      <td>2020-09-22</td>
      <td>2020</td>
      <td>09</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2020-09-23</th>
      <td>6835.209961</td>
      <td>6649.240234</td>
      <td>6828.919922</td>
      <td>6657.819824</td>
      <td>0</td>
      <td>6657.819824</td>
      <td>2020-09-23</td>
      <td>2020</td>
      <td>09</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2020-09-24</th>
      <td>6744.000000</td>
      <td>6602.359863</td>
      <td>6635.979980</td>
      <td>6678.040039</td>
      <td>0</td>
      <td>6678.040039</td>
      <td>2020-09-24</td>
      <td>2020</td>
      <td>09</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2020-09-25</th>
      <td>6802.140137</td>
      <td>6642.040039</td>
      <td>6657.830078</td>
      <td>6784.950195</td>
      <td>0</td>
      <td>6784.950195</td>
      <td>2020-09-25</td>
      <td>2020</td>
      <td>09</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>8250 rows 10 columns</p>
</div>



```python
def compare_returns(SP, n_years):
    """
    This function aims to compare 5 n-year short-term investment plans based on a certain ETF/stock's performance data.
    :param SP: Original dataframe that contains a certain ETF/stock's performance data
    :param n_years: Number of years set for the short-term investment plan
    :return: A dataframe that contains the overall return for the 5 plans
    """
    test_months = 12*n_years
    test_quarters = 4*n_years

    df = SP[['year', 'Adj Close']].groupby('year').first()

    # Compute all in return
    df['return'] = df['Adj Close'].pct_change(periods=n_years)
    df['return_next'] = df['return'].shift(-n_years)
    df = df.drop(columns='return')

    for year in SP.year.unique()[:-n_years]:
        # Compute monthly dollar cost averaging return
        df.loc[f'{int(year)}', 'return_next_m'] = (
        _calc_return_over_periods_list_simplified(
            [1]*test_months, 
            df_m.loc[(f'{year}'):(f'{int(year) + n_years - 1}'),'return_next'].tolist(), 
            test_months)
        )

        # Compute quarterly dollar cost averaging return
        df.loc[f'{int(year)}', 'return_next_q'] = (
        _calc_return_over_periods_list_simplified(
            [1]*test_quarters, 
            df_q.loc[(f'{year}'):(f'{int(year) + n_years - 1}'),'return_next'].tolist(), 
            test_quarters)
        )

    # Compute value averaging returns    
    df_va_m = df_m['Adj Close'].reset_index()
    df_va_q = df_q['Adj Close'].reset_index()

    ### Join results together into one dataframe
    for year in SP.year.unique()[:-n_years]:
        df_va_m_sample = df_va_m.loc[(df_va_m['year']>=year) & 
                                     (df_va_m['year']<=str(int(year) + n_years - 1)),:].reset_index()
        df_va_q_sample = df_va_q.loc[(df_va_q['year']>=year) & 
                                     (df_va_q['year']<=str(int(year) + n_years - 1)),:].reset_index()
        annual_return_va_m = _calc_value_averaging_return_simplified(df_va_m_sample)
        annual_return_va_q = _calc_value_averaging_return_simplified(df_va_q_sample)
        df.loc[f'{int(year)}', 'return_va_m'] = annual_return_va_m
        df.loc[f'{int(year)}', 'return_va_q'] = annual_return_va_q
    
    return df


def _calc_return_over_periods_list_simplified(
    investment_per_period_list: list, 
    return_per_period_list: list, 
    n_period: int, 
    print_values=False) -> float:
    
    value = 0
    for i in range(n_period):
        value += investment_per_period_list[i]
        value = value * (1 + return_per_period_list[i])
    return_rate = value/sum(investment_per_period_list) - 1
    
    return return_rate


def _calc_value_averaging_return_simplified(df_va, value_per_period=1):
    
    df_va.loc[0,'total_value'] = value_per_period
    df_va.loc[0,'shares_should_have_in_total'] = value_per_period/df_va.loc[0,'Adj Close']
    df_va.loc[0,'shares_to_buy_or_sell'] = value_per_period/df_va.loc[0,'Adj Close']
    df_va.loc[0,'should_invest'] = value_per_period
    df_va.loc[0,'total_invest'] = value_per_period

    for i in range(1,len(df_va.index)):
        df_va.loc[i,'total_value'] = value_per_period*(i+1)
        df_va.loc[i,'shares_should_have_in_total'] = df_va.loc[i,'total_value']/df_va.loc[i,'Adj Close']
        df_va.loc[i,'shares_to_buy_or_sell'] = df_va.loc[i,'shares_should_have_in_total'] - df_va.loc[i-1,'shares_should_have_in_total']
        df_va.loc[i,'should_invest'] = df_va.loc[i,'shares_to_buy_or_sell']*df_va.loc[i,'Adj Close']
        df_va.loc[i,'total_invest'] = df_va.loc[i-1,'total_invest'] + df_va.loc[i,'should_invest']
    
    return df_va.loc[len(df_va.index)-1,'total_value']/df_va.loc[len(df_va.index)-1,'total_invest']-1
```


```python
def plot_changes(df):
    plt.figure(figsize=(15,5))
    plt.plot(df['return_next'], label='all-in')
    plt.plot(df['return_va_m'], label='value-averaging (monthly)')
    # plt.plot(df['return_va_q'], label='value-averaging (quarterly)')
    plt.plot(df['return_next_m'], label='dollar-cost (monthly)')
    # plt.plot(df['return_next_q'], label='dollar-cost (quarterly)')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('year')
    plt.ylabel('return rate')
    plt.xticks(rotation=90)
    plt.title('Annual return rate of the all-in, monthly dollar-cost averaging, and monthly value averaging methods')
    plt.legend()
    plt.show()
```

----

```python
# Assuming we are setting up a 2 year plan and then compare the return
test_years = 2
df_2 = compare_returns(SP, test_years)
```


```python
df_2.round(3)
```

<div>
<table>
  <thead>
    <tr style="text-align: right;">
      <th>year</th>
      <th>Adj Close</th>
      <th>return_next</th>
      <th>return_next_m</th>
      <th>return_next_q</th>
      <th>return_va_m</th>
      <th>return_va_q</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1988</th>
      <td>256.02</td>
      <td>0.508</td>
      <td>0.291</td>
      <td>0.306</td>
      <td>0.296</td>
      <td>0.322</td>
    </tr>
    <tr>
      <th>1989</th>
      <td>285.65</td>
      <td>0.272</td>
      <td>0.038</td>
      <td>0.036</td>
      <td>0.039</td>
      <td>0.030</td>
    </tr>
    <tr>
      <th>1990</th>
      <td>386.16</td>
      <td>0.243</td>
      <td>0.223</td>
      <td>0.222</td>
      <td>0.127</td>
      <td>0.130</td>
    </tr>
    <tr>
      <th>1991</th>
      <td>363.44</td>
      <td>0.420</td>
      <td>0.143</td>
      <td>0.155</td>
      <td>0.138</td>
      <td>0.152</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>479.84</td>
      <td>0.182</td>
      <td>0.108</td>
      <td>0.107</td>
      <td>0.102</td>
      <td>0.102</td>
    </tr>
    <tr>
      <th>1993</th>
      <td>516.00</td>
      <td>0.115</td>
      <td>0.039</td>
      <td>0.049</td>
      <td>0.017</td>
      <td>0.026</td>
    </tr>
    <tr>
      <th>1994</th>
      <td>567.10</td>
      <td>0.408</td>
      <td>0.298</td>
      <td>0.313</td>
      <td>0.308</td>
      <td>0.340</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>575.52</td>
      <td>0.684</td>
      <td>0.281</td>
      <td>0.287</td>
      <td>0.367</td>
      <td>0.390</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>798.22</td>
      <td>0.635</td>
      <td>0.330</td>
      <td>0.340</td>
      <td>0.393</td>
      <td>0.437</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>969.05</td>
      <td>0.722</td>
      <td>0.315</td>
      <td>0.318</td>
      <td>0.312</td>
      <td>0.300</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>1305.04</td>
      <td>0.534</td>
      <td>0.254</td>
      <td>0.243</td>
      <td>0.239</td>
      <td>0.212</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>1668.52</td>
      <td>0.070</td>
      <td>-0.052</td>
      <td>-0.048</td>
      <td>-0.020</td>
      <td>-0.021</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>2002.11</td>
      <td>-0.187</td>
      <td>-0.103</td>
      <td>-0.098</td>
      <td>-0.105</td>
      <td>-0.109</td>
    </tr>
    <tr>
      <th>2001</th>
      <td>1785.86</td>
      <td>-0.270</td>
      <td>-0.149</td>
      <td>-0.157</td>
      <td>-0.114</td>
      <td>-0.112</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>1628.51</td>
      <td>-0.006</td>
      <td>0.161</td>
      <td>0.152</td>
      <td>0.133</td>
      <td>0.136</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>1303.17</td>
      <td>0.370</td>
      <td>0.187</td>
      <td>0.187</td>
      <td>0.189</td>
      <td>0.194</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>1618.05</td>
      <td>0.186</td>
      <td>0.109</td>
      <td>0.102</td>
      <td>0.113</td>
      <td>0.104</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>1784.96</td>
      <td>0.224</td>
      <td>0.153</td>
      <td>0.147</td>
      <td>0.146</td>
      <td>0.144</td>
    </tr>
    <tr>
      <th>2006</th>
      <td>1918.96</td>
      <td>0.185</td>
      <td>0.066</td>
      <td>0.077</td>
      <td>0.088</td>
      <td>0.099</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>2183.92</td>
      <td>-0.314</td>
      <td>-0.288</td>
      <td>-0.260</td>
      <td>-0.310</td>
      <td>-0.277</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>2273.41</td>
      <td>-0.179</td>
      <td>0.104</td>
      <td>0.109</td>
      <td>0.116</td>
      <td>0.143</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>1499.17</td>
      <td>0.426</td>
      <td>0.286</td>
      <td>0.246</td>
      <td>0.258</td>
      <td>0.204</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>1867.06</td>
      <td>0.174</td>
      <td>0.094</td>
      <td>0.072</td>
      <td>0.082</td>
      <td>0.046</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>2138.30</td>
      <td>0.201</td>
      <td>0.136</td>
      <td>0.132</td>
      <td>0.108</td>
      <td>0.097</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>2192.40</td>
      <td>0.499</td>
      <td>0.263</td>
      <td>0.265</td>
      <td>0.275</td>
      <td>0.278</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>2568.55</td>
      <td>0.467</td>
      <td>0.200</td>
      <td>0.198</td>
      <td>0.217</td>
      <td>0.212</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>3286.69</td>
      <td>0.145</td>
      <td>0.038</td>
      <td>0.032</td>
      <td>0.095</td>
      <td>0.078</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>3768.68</td>
      <td>0.145</td>
      <td>0.115</td>
      <td>0.104</td>
      <td>0.091</td>
      <td>0.074</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>3763.99</td>
      <td>0.396</td>
      <td>0.227</td>
      <td>0.222</td>
      <td>0.222</td>
      <td>0.219</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>4315.08</td>
      <td>0.157</td>
      <td>-0.007</td>
      <td>0.001</td>
      <td>0.111</td>
      <td>0.120</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>5256.28</td>
      <td>0.257</td>
      <td>0.185</td>
      <td>0.199</td>
      <td>0.154</td>
      <td>0.168</td>
    </tr>
    <tr>
      <th>2019</th>
      <td>4990.56</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2020</th>
      <td>6609.29</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_changes(df_2)
```


![png](/assets/5_output_37_0.png)



```python
compare_strategy(df_2, 'return_va_m', 'return_next_m', 'monthly value-averaging', 'monthly dollar-cost-averaging')
```

Output:

    number of times monthly value-averaging is better than monthly dollar-cost-averaging: 15
    number of times monthly value-averaging is equal to monthly dollar-cost-averaging: 0
    number of times monthly value-averaging is worse than monthly dollar-cost-averaging: 16

----

```python
compare_strategy(df_2, 'return_next', 'return_next_m', 'all-in', 'monthly dollar-cost-averaging')
```

Output:

    number of times all-in is better than monthly dollar-cost-averaging: 26
    number of times all-in is equal to monthly dollar-cost-averaging: 0
    number of times all-in is worse than monthly dollar-cost-averaging: 5

----

```python
# Assuming we are setting up a 3 year plan and then compare the return
test_years = 3
df_3 = compare_returns(SP, test_years)

plot_changes(df_3)
```


![png](/assets/5_output_40_0.png)



```python
compare_strategy(df_3, 'return_va_m', 'return_next_m', 'monthly value-averaging', 'monthly dollar-cost-averaging')
```

Output:

    number of times monthly value-averaging is better than monthly dollar-cost-averaging: 22
    number of times monthly value-averaging is equal to monthly dollar-cost-averaging: 0
    number of times monthly value-averaging is worse than monthly dollar-cost-averaging: 8

----

```python
compare_strategy(df_3, 'return_next', 'return_next_m', 'all-in', 'monthly dollar-cost-averaging')
```

Output:

    number of times all-in is better than monthly dollar-cost-averaging: 25
    number of times all-in is equal to monthly dollar-cost-averaging: 0
    number of times all-in is worse than monthly dollar-cost-averaging: 5

----

```python
# Assuming we are setting up a 4 year plan and then compare the return
test_years = 4
df_4 = compare_returns(SP, test_years)

plot_changes(df_4)
```


![png](/assets/5_output_43_0.png)



```python
compare_strategy(df_4, 'return_va_m', 'return_next_m', 'monthly value-averaging', 'monthly dollar-cost-averaging')
```

Output:

    number of times monthly value-averaging is better than monthly dollar-cost-averaging: 24
    number of times monthly value-averaging is equal to monthly dollar-cost-averaging: 0
    number of times monthly value-averaging is worse than monthly dollar-cost-averaging: 5

----

```python
compare_strategy(df_4, 'return_next', 'return_next_m', 'all-in', 'monthly dollar-cost-averaging')
```

Output:

    number of times all-in is better than monthly dollar-cost-averaging: 23
    number of times all-in is equal to monthly dollar-cost-averaging: 0
    number of times all-in is worse than monthly dollar-cost-averaging: 6


----

# Conclusion

When we have a fixed amount of money, in most cases, using the all-in method performs the best, but it also involves higher risk.

The value averaging strategy did not differ much from the dollar cost averaging. When it is a one or two year short investment plan, the dollar cost averaging is more likely to perform better; if it is a three or four year investment plan, the value averaging strategy is more likely to perform better.

Given that the value averaging strategy is a bit tedious to implement, I think it is better to use either all-in or dollar-cost averaging for short term investment.

Please note that, in this notebook, I did not use the adjusted version of the value averaging method with the minimum and maximum thresholds of investment per period (as introduced in BP3 Plan 3). It might bring you some extra return if it is a longer term investment plan.

