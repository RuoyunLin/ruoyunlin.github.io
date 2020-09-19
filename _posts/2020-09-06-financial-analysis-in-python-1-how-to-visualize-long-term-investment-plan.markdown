---
layout: post
title: "Financial Analysis in Python #1: How to Visualize Long Term Investment Plan"
date: 2020-09-06T16:54:58+02:00
tags: [Finance, Python, visualization, long-term investment, retirement]
categories: [Finance, Data Science]
---

# Overview

It is always important to learn how to invest your money properly so that you do not need to worry about money after retirement.

Starting from this blog post, I am going to share how I used python to set up my own investment plan.

I will try to cover a few topics like:

- how to visualize the long term investment plan
- backtest **dollar-cost averaging** strategy for **long-term** investment
- backtest **value averaging** strategy for **long-term** investment
- compare different investment strategies for **short-term** investment

The jupyter notebooks can be downloaded [here](https://github.com/RuoyunLin/code_snippets/tree/master/finance).

# Disclaimer

Investing money into anything is often involved with risk. Please do your own research before investing and be responsible for your own investment decisions.

I am just learning investment on my own and want to share some codes that I have written that might be useful for others.

The content here is only for informational purpose (instead of taking them as professional investment advice).


# Introduction

This blogpost contains a function that helps you visualize your long term investment plan.

Step 1: Calculate how much savings do you have for now and estimate how much you can save per year

Step 2: Think about how much money do you need after retirement

- 3,000 Euro/Month --> 36,000 Euro/Year

- Assuming we can have 5% interest rate per year, then we need to accumulate at least 720,000 Euro before retirement (36,000/0.05 = 720,000)

- Please also note that, assuming the inflation rate of 3% per year, the buying power of 3000 Euro in 2050 might only worth 1235 Euro in 2020. Please adjust the inflation rate by yourself when estimating how much money do you need after retirement

Step 3: Estmiate (roughly) how much money do you need to invest per year and calculate it's long term return

## Define functions

```python
# Import libraries
from matplotlib import pyplot as plt
import pandas as pd

plt.style.use('bmh')
```


```python
def calc_return_over_periods(initial_investment: int, investment_per_period: int,
                             return_per_period: float, n_period: int,
                             increase_investment_per_period: float = 0.00,
                             invest_at_period_begin: bool = True) -> pd.DataFrame:
    """
    This function calculates the overall rate of return across n periods given a certain return rate pre period
    :param initial_investment: Amount of initial investment
    :param investment_per_period: Amount of investment per period
    :param return_per_period: Assumed rate of return per period
    :param n_period: Total number of periods
    :param increase_investment_per_period: Increase the amount of investment by x for each period after the first investment period
    :param invest_at_period_begin: whether to invest at the begining of each period or not
    :return: dataframe contains the rate of return for n periods.
    """
    df_result = pd.DataFrame()

    for i in range(n_period):
        if i == 0:
            if invest_at_period_begin:
                period_begin_investment = initial_investment + investment_per_period
            if not invest_at_period_begin:
                period_begin_investment = initial_investment
            df_result.loc[i, 'period_begin_investment'] = period_begin_investment
            df_result.loc[i, 'total_investment'] = period_begin_investment
            value = period_begin_investment * (1 + return_per_period)
            df_result.loc[i, 'value_by_next_period'] = value
        else:
            if invest_at_period_begin:
                period_begin_investment = investment_per_period * ((1 + increase_investment_per_period) ** i)
            if not invest_at_period_begin:
                period_begin_investment = investment_per_period * ((1 + increase_investment_per_period) ** (i - 1))
            value += period_begin_investment
            df_result.loc[i, 'period_begin_investment'] = period_begin_investment
            df_result.loc[i, 'total_investment'] = (df_result.loc[i - 1, 'total_investment'] +
                                                    df_result.loc[i, 'period_begin_investment'])
            value = value * (1 + return_per_period)
            df_result.loc[i, 'value_by_next_period'] = value

    df_result['return_by_next_period'] = df_result['value_by_next_period'] / df_result['total_investment'] - 1

    return df_result.round(2)


def plot_changes(df: pd.DataFrame,
                 y1: str = 'total_investment', y2: str = 'value_by_next_period',
                 xlabel: str = 'period', ylabel: str = 'value',
                 title: str = 'Visualize your saving plan and its future value',
                 target: int = None) -> None:
    """
    This function visualizes the total investment and the total value of the investment plan across time
    :param df: A dataframe contains total investment and total value
    :param y1: Column name
    :param y2: Column name
    :param xlabel: X label
    :param ylabel: Y label
    :param title: Title of the graph
    :param target: Final value one wants to achieve
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

## Case 1

Imagine that you are around 30 years old and have some savings (e.g., 20k Euro), and you are determined to invest 10000 Euro each year in order to have enough money for your retirement.

If you invest your savings into a ETF (e.g., SP500), you are likely to get about 5% annual return on average.

If everything works well, you want to retire around 60, so about 30 years to go!

Let's use the function above to check if we can have enough money saved for retirement in this case.


```python
# Each period is a year

# Assuming investing 20000 Euro initially 
initial_investment = 20000

# Assuming investing 10000 Euro in the begining of each year
investment_per_period = 10000

# Assuming the average investment return rate is about 5%
return_per_period = 0.05

# Assuming the plan lasts for 30 years
n_period = 30

df_plan1 = calc_return_over_periods(initial_investment, investment_per_period, 
                                   return_per_period, n_period)
df_plan1
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
<table style="margin-left:auto;margin-right:auto;width:80%" border="1" class="minimalistBlack">
  <thead>
    <tr>
      <th></th>
      <th>period_begin_investment</th>
      <th>total_investment</th>
      <th>value_by_next_period</th>
      <th>return_by_next_period</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30000.0</td>
      <td>30000.0</td>
      <td>31500.00</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10000.0</td>
      <td>40000.0</td>
      <td>43575.00</td>
      <td>0.09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10000.0</td>
      <td>50000.0</td>
      <td>56253.75</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000.0</td>
      <td>60000.0</td>
      <td>69566.44</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10000.0</td>
      <td>70000.0</td>
      <td>83544.76</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10000.0</td>
      <td>80000.0</td>
      <td>98222.00</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10000.0</td>
      <td>90000.0</td>
      <td>113633.10</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10000.0</td>
      <td>100000.0</td>
      <td>129814.75</td>
      <td>0.30</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10000.0</td>
      <td>110000.0</td>
      <td>146805.49</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10000.0</td>
      <td>120000.0</td>
      <td>164645.76</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10000.0</td>
      <td>130000.0</td>
      <td>183378.05</td>
      <td>0.41</td>
    </tr>
    <tr>
      <th>11</th>
      <td>10000.0</td>
      <td>140000.0</td>
      <td>203046.95</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>12</th>
      <td>10000.0</td>
      <td>150000.0</td>
      <td>223699.30</td>
      <td>0.49</td>
    </tr>
    <tr>
      <th>13</th>
      <td>10000.0</td>
      <td>160000.0</td>
      <td>245384.27</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th>14</th>
      <td>10000.0</td>
      <td>170000.0</td>
      <td>268153.48</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>15</th>
      <td>10000.0</td>
      <td>180000.0</td>
      <td>292061.16</td>
      <td>0.62</td>
    </tr>
    <tr>
      <th>16</th>
      <td>10000.0</td>
      <td>190000.0</td>
      <td>317164.21</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>17</th>
      <td>10000.0</td>
      <td>200000.0</td>
      <td>343522.42</td>
      <td>0.72</td>
    </tr>
    <tr>
      <th>18</th>
      <td>10000.0</td>
      <td>210000.0</td>
      <td>371198.54</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>19</th>
      <td>10000.0</td>
      <td>220000.0</td>
      <td>400258.47</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>20</th>
      <td>10000.0</td>
      <td>230000.0</td>
      <td>430771.40</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>21</th>
      <td>10000.0</td>
      <td>240000.0</td>
      <td>462809.97</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>22</th>
      <td>10000.0</td>
      <td>250000.0</td>
      <td>496450.46</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>23</th>
      <td>10000.0</td>
      <td>260000.0</td>
      <td>531772.99</td>
      <td>1.05</td>
    </tr>
    <tr>
      <th>24</th>
      <td>10000.0</td>
      <td>270000.0</td>
      <td>568861.64</td>
      <td>1.11</td>
    </tr>
    <tr>
      <th>25</th>
      <td>10000.0</td>
      <td>280000.0</td>
      <td>607804.72</td>
      <td>1.17</td>
    </tr>
    <tr>
      <th>26</th>
      <td>10000.0</td>
      <td>290000.0</td>
      <td>648694.95</td>
      <td>1.24</td>
    </tr>
    <tr>
      <th>27</th>
      <td>10000.0</td>
      <td>300000.0</td>
      <td>691629.70</td>
      <td>1.31</td>
    </tr>
    <tr>
      <th>28</th>
      <td>10000.0</td>
      <td>310000.0</td>
      <td>736711.19</td>
      <td>1.38</td>
    </tr>
    <tr>
      <th>29</th>
      <td>10000.0</td>
      <td>320000.0</td>
      <td>784046.75</td>
      <td>1.45</td>
    </tr>
  </tbody>
</table>
</div>


```python
target_value = 720000

plot_changes(df_plan1, title='Case 1', target = target_value, xlabel='year')
```


![png](/assets/2_output_7_0.png "Case 1"){:width="80%"}


Conguraturations! It seems that if you just stick to this plan, you will have enough money for your retirement.

## Case 2

Due to inflation and the increase of your salary, you might decided to save 2% more each year, let's check how that is going to change the final value


```python
# Each period is a year

# Assuming investing 20000 Euro initially 
initial_investment = 20000

# Assuming investing 10000 Euro in the begining of each year
investment_per_period = 10000

# Assuming the average investment return rate is about 5%
return_per_period = 0.05

# Assuming the plan lasts for 30 years
n_period = 30

# Assuming increase the amount of investment each year by 2%
increase_investment_per_period = 0.02

df_plan2 = calc_return_over_periods(initial_investment, investment_per_period, 
                                   return_per_period, n_period,
                                   increase_investment_per_period)

df_plan2
```

<div>
<table style="margin-left:auto;margin-right:auto;width:80%" border="1" class="minimalistBlack">
  <thead>
    <tr>
      <th></th>
      <th>period_begin_investment</th>
      <th>total_investment</th>
      <th>value_by_next_period</th>
      <th>return_by_next_period</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30000.00</td>
      <td>30000.00</td>
      <td>31500.00</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10200.00</td>
      <td>40200.00</td>
      <td>43785.00</td>
      <td>0.09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10404.00</td>
      <td>50604.00</td>
      <td>56898.45</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10612.08</td>
      <td>61216.08</td>
      <td>70886.06</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10824.32</td>
      <td>72040.40</td>
      <td>85795.90</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>5</th>
      <td>11040.81</td>
      <td>83081.21</td>
      <td>101678.54</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>6</th>
      <td>11261.62</td>
      <td>94342.83</td>
      <td>118587.17</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>7</th>
      <td>11486.86</td>
      <td>105829.69</td>
      <td>136577.73</td>
      <td>0.29</td>
    </tr>
    <tr>
      <th>8</th>
      <td>11716.59</td>
      <td>117546.28</td>
      <td>155709.04</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>9</th>
      <td>11950.93</td>
      <td>129497.21</td>
      <td>176042.96</td>
      <td>0.36</td>
    </tr>
    <tr>
      <th>10</th>
      <td>12189.94</td>
      <td>141687.15</td>
      <td>197644.55</td>
      <td>0.39</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12433.74</td>
      <td>154120.90</td>
      <td>220582.21</td>
      <td>0.43</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12682.42</td>
      <td>166803.32</td>
      <td>244927.86</td>
      <td>0.47</td>
    </tr>
    <tr>
      <th>13</th>
      <td>12936.07</td>
      <td>179739.38</td>
      <td>270757.12</td>
      <td>0.51</td>
    </tr>
    <tr>
      <th>14</th>
      <td>13194.79</td>
      <td>192934.17</td>
      <td>298149.51</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>15</th>
      <td>13458.68</td>
      <td>206392.85</td>
      <td>327188.60</td>
      <td>0.59</td>
    </tr>
    <tr>
      <th>16</th>
      <td>13727.86</td>
      <td>220120.71</td>
      <td>357962.28</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>17</th>
      <td>14002.41</td>
      <td>234123.12</td>
      <td>390562.93</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>18</th>
      <td>14282.46</td>
      <td>248405.59</td>
      <td>425087.66</td>
      <td>0.71</td>
    </tr>
    <tr>
      <th>19</th>
      <td>14568.11</td>
      <td>262973.70</td>
      <td>461638.56</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>20</th>
      <td>14859.47</td>
      <td>277833.17</td>
      <td>500322.94</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>21</th>
      <td>15156.66</td>
      <td>292989.84</td>
      <td>541253.58</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>22</th>
      <td>15459.80</td>
      <td>308449.63</td>
      <td>584549.05</td>
      <td>0.90</td>
    </tr>
    <tr>
      <th>23</th>
      <td>15768.99</td>
      <td>324218.62</td>
      <td>630333.94</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>24</th>
      <td>16084.37</td>
      <td>340303.00</td>
      <td>678739.23</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>25</th>
      <td>16406.06</td>
      <td>356709.06</td>
      <td>729902.55</td>
      <td>1.05</td>
    </tr>
    <tr>
      <th>26</th>
      <td>16734.18</td>
      <td>373443.24</td>
      <td>783968.57</td>
      <td>1.10</td>
    </tr>
    <tr>
      <th>27</th>
      <td>17068.86</td>
      <td>390512.10</td>
      <td>841089.31</td>
      <td>1.15</td>
    </tr>
    <tr>
      <th>28</th>
      <td>17410.24</td>
      <td>407922.35</td>
      <td>901424.53</td>
      <td>1.21</td>
    </tr>
    <tr>
      <th>29</th>
      <td>17758.45</td>
      <td>425680.79</td>
      <td>965142.12</td>
      <td>1.27</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_changes(df_plan2, title='Case 2', target=target_value, xlabel='year')
```


![png](/assets/2_output_11_0.png "Case 2"){:width="80%"}


It seems that it only takes about 26 years instead of the 30 years in order to have enough savings for retirement. However, the return rate was reduced a bit due to the investment of more money at the later stage of this plan.

## Case 3

Let's imagine that you are at your early 20s and just got your first job recently and do not have any savings for now. 

From next month on, you will be able to save 500 each month.


```python
# Each period is a month

# No initial investment
initial_investment = 0

# Invest 500 Euro at the end of each month
investment_per_period = 500

# Assuming the average investment return rate is about 5% per year
return_per_year = 0.05
return_per_period = (return_per_year + 1)**(1/12) - 1 

# Assuming the plan lasts for 40 years
n_years = 40
n_period = n_years*12

df_plan3 = calc_return_over_periods(initial_investment, investment_per_period, 
                                   return_per_period, n_period,
                                   invest_at_period_begin=False)

df_plan3
```


<div>
<table style="margin-left:auto;margin-right:auto;width:80%" border="1" class="minimalistBlack">
  <thead>
    <tr>
      <th></th>
      <th>period_begin_investment</th>
      <th>total_investment</th>
      <th>value_by_next_period</th>
      <th>return_by_next_period</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>500.0</td>
      <td>500.0</td>
      <td>502.04</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>500.0</td>
      <td>1000.0</td>
      <td>1006.12</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>500.0</td>
      <td>1500.0</td>
      <td>1512.26</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>500.0</td>
      <td>2000.0</td>
      <td>2020.45</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>475</th>
      <td>500.0</td>
      <td>237500.0</td>
      <td>726824.57</td>
      <td>2.06</td>
    </tr>
    <tr>
      <th>476</th>
      <td>500.0</td>
      <td>238000.0</td>
      <td>730287.78</td>
      <td>2.07</td>
    </tr>
    <tr>
      <th>477</th>
      <td>500.0</td>
      <td>238500.0</td>
      <td>733765.10</td>
      <td>2.08</td>
    </tr>
    <tr>
      <th>478</th>
      <td>500.0</td>
      <td>239000.0</td>
      <td>737256.59</td>
      <td>2.08</td>
    </tr>
    <tr>
      <th>479</th>
      <td>500.0</td>
      <td>239500.0</td>
      <td>740762.30</td>
      <td>2.09</td>
    </tr>
  </tbody>
</table>
</div>


```python
plot_changes(df_plan3, title='Case 3', target=target_value, xlabel='month')
```


![png](/assets/2_output_15_0.png "Case 3"){:width="80%"}


Even with little amount of savings, you can still have enough money for your retirement by 60. 

## Conclusion

We can observe that, in Case 3, the rate of return looks way better than Case 1 and and Case 2. So start investing earlier is always a good idea! 

Hope this post is useful for you to set up a long term investment plan :) 

Starting from the next post, we are going to do some backtesting using the total return index of SP500.

