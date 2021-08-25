# PER/PBR, ROE 전략
# 0. 데이터 가져오기
# 1. PER/PBR, ROE 순위 산출하기
# 2. 수익률 산출하기

# 0.S&P데이터 가져오기
import FinanceDataReader as fdr
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
pd.set_option('display.float_format', lambda x: '%.2f' % x)

sp500 = fdr.StockListing('S&P500')

import pandas as pd
import requests
from bs4 import BeautifulSoup
import re

# 종목별 valuation 지표 생성
def stock_factors(sym):
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get('https://finviz.com/quote.ashx?t={}'.format(sym.lower()), headers=headers)
    soup = BeautifulSoup(r.text)
    snapshot_table2 = soup.find('table', attrs={'class': 'snapshot-table2'})
    tables = pd.read_html(str(snapshot_table2))
    df = tables[0]
    df.columns = ['key', 'value'] * 6

    ## 컬럼을 행으로 만들기
    df_list = [df.iloc[:, i*2: i*2+2] for i in range(6)]
    df_factor = pd.concat(df_list, ignore_index=True)
    df_factor.set_index('key', inplace=True)

    v = df_factor.value
    marcap = _conv_to_float(v['Market Cap'])
    dividend = _conv_to_float(v['Dividend %'])
    per = _conv_to_float(v['P/E'])
    pbr = _conv_to_float(v['P/B'])
    beta = _conv_to_float(v['Beta'])
    roe = _conv_to_float(v['ROE'])

    return {'MarCap':marcap, 'Dividend':dividend, 'PER':per, 'PBR':pbr, 'Beta':beta, 'ROE':roe}

# 데이터 전처리 변환
def _conv_to_float(s):
    if s[-1] == '%':
        s = s.replace('%', '')
    if s[-1] in list('BMK'):
        powers = {'B': 10 ** 9, 'M': 10 ** 6, 'K': 10 ** 3, '': 1}
        m = re.search("([0-9\.]+)(M|B|K|)", s)
        if m:
            val, mag = m.group(1), m.group(2)
            return float(val) * powers[mag]
    try:
        result = float(s)
    except:
        result = None
    return result

# 데이터 내려받기 (JSON)
folder = "sp500/"
re_map_sym = {'BRKB': 'BRK-B', 'BR': 'BRK-A', 'BFB': 'BF-B'}

if not os.path.isdir(folder):
    os.mkdir(folder)

for ix, row in sp500.iterrows():
    sym, name = row['Symbol'], row['Name']
    json_fn = folder + '%s.json' % (sym)
    if os.path.exists(json_fn):
        print('skip', json_fn)
        continue
    if sym in re_map_sym:
        sym = re_map_sym[sym]
    factors = stock_factors(sym)
    with open(json_fn, 'w') as f:
        json.dump(factors, f)
    print(sym, name)

# 데이터 읽기 (JSON)
for ix, row in sp500.iterrows():
    sym, name = row['Symbol'], row['Name']
    json_fn = folder + '%s.json' % (sym)

    with open(json_fn, 'r') as f:
        factors = json.load(f)

        for f in ['MarCap', 'Dividend', 'PER', 'PBR', 'Beta', 'ROE']:
            sp500.loc[ix, f] = factors[f]


# 데이터 통계
import matplotlib.pyplot as plt
def make_colors(n, colormap=plt.cm.Spectral):
    return colormap(np.linspace(0.1, 1.0, n))
def make_explode(n):
    explodes = np.zeros(n)
    explodes[0] = 0.15
    return explodes
sector_marcap = sp500.groupby('Sector')['MarCap'].sum().sort_values(ascending=False)
values = sector_marcap.values
labels = sector_marcap.index
n = len(labels)

plt.figure(figsize=(15,12))
plt.title('Sector MarketCap', fontsize = 20, fontweight = 'bold')
plt.pie(values, labels=labels, colors=make_colors(n), explode=make_explode(n), autopct='%1.1f%%', shadow=True, startangle=135)
plt.axis('equal')
plt.show()

sp500.groupby('Sector').describe()['PER'].sort_values('mean', ascending=False)


# 1. PER/PBR, ROE rank 산출하기

PER_rank = sp500['PER'].rank(ascending = True, na_option = 'bottom')
PBR_rank = sp500['PBR'].rank(ascending = True, na_option = 'bottom')
ROE_rank = sp500['ROE'].rank(ascending = False, na_option = 'bottom')
DIV_rank = sp500['Dividend'].rank(ascending = False, na_option = 'bottom')

PERROE_rank = PER_rank + ROE_rank
PBRROE_rank = PBR_rank + ROE_rank


# 2. 수익률 계산 PER + ROE

selected = sp500.loc[np.where(PBRROE_rank<=400)]
start_date = '20210401'
end_date = datetime.today()

def get_return(selected, start_date, end_date) :
    for ind, val in enumerate(selected['Symbol'].values):
        symbol = selected.loc[selected['Symbol'] == val, 'Symbol'].values[0]
        df = fdr.DataReader(val, start_date, end_date)
        if ind == 0 :
            return_df = pd.DataFrame(index = df.index)
        df['rtn'] = df['Close'].pct_change(periods=1)
        df['cum_rtn'] = (1+df['rtn']).cumprod() - 1
        tmp = df.loc[:,['cum_rtn']].rename(columns = {'cum_rtn':symbol})
        return_df = return_df.join(tmp, how = 'left')
        return_df.dropna()
        df = None
    return return_df

result = get_return(selected, start_date, end_date)


# Cumulative Compounded Returns for valueStrategu
plt.figure(figsize=(17,7))
plt.title('PBR ROE Startegy')
plt.plot(result)
# plt.plot(return_df,label = [ i for i in return_df.columns])
plt.legend()
plt.show()