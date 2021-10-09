

import yahoo_fin.stock_info as si

ticker = 'gs'
df_prc = si.get_data(ticker, start_date = '2018-01-01', interval="1mo")
df_prc.adjclose.plot()
df = si.get_income_statement(ticker, yearly = True)
dff  = df.loc[['totalRevenue','operatingIncome','netIncome'],:]
dff = dff.transpose()
dff = dff.sort_index(ascending=True)
dff['REV_G'] = ( dff['totalRevenue'] - dff['totalRevenue'].shift(1) ) / abs(dff['totalRevenue'].shift(1))
dff['OP_G'] = ( dff['operatingIncome'] - dff['operatingIncome'].shift(1) ) / abs(dff['operatingIncome'].shift(1))
dff['NI_G'] = ( dff['netIncome'] - dff['netIncome'].shift(1) ) / abs(dff['netIncome'].shift(1))
dfff = dff[['REV_G', 'OP_G', 'NI_G']]
dfff.plot()
df = si.get_income_statement(ticker, yearly = False)
dff  = df.loc[['totalRevenue','operatingIncome','netIncome'],:]
dff = dff.transpose()
dff = dff.sort_index(ascending=True)
dff['REV_G'] = ( dff['totalRevenue'] - dff['totalRevenue'].shift(1) ) / abs(dff['totalRevenue'].shift(1))
dff['OP_G'] = ( dff['operatingIncome'] - dff['operatingIncome'].shift(1) ) / abs(dff['operatingIncome'].shift(1))
dff['NI_G'] = ( dff['netIncome'] - dff['netIncome'].shift(1) ) / abs(dff['netIncome'].shift(1))
dfff = dff[['REV_G', 'OP_G', 'NI_G']]
dfff.plot()