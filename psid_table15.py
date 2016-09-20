import pandas as pd
import numpy as np

folder89 = 'C:/Users/j0el/Documents/Wisconsin/899/Data/PSID/1989'
folder94 = 'C:/Users/j0el/Documents/Wisconsin/899/Data/PSID/1994'

# Read in Stata files

df89 = pd.read_stata(folder89+'/1989_clean.dta')
df94 = pd.read_stata(folder94+'/1994_clean.dta')

# Read in separate Wealth files

df89w = pd.read_stata(folder89+'/1989_wealth_clean.dta')
df94w = pd.read_stata(folder94+'/1994_wealth_clean.dta')

# Merge 89 and 94 family and wealth data on yearly interview number

df89merge = pd.merge(df89,df89w,left_on='V16302',right_on='S201',how='inner')
df94merge = pd.merge(df94,df94w,left_on='ER2002',right_on='S301',how='inner')

# Datasets will be matched based on 1968 household identifier and age of the head of household

	# Advance age of head in 1989 five years to match with 1994 age

	df89merge['age1994'] = df89merge.V16319+5

# Merge all datasets on 1968 identifier and age in 1994

df = pd.merge(df89merge,df94merge,left_on=['V16317','age1994'],right_on=['ER2005G','ER2007'],how='inner')

	# Define earnings, income, and wealth

	df['earnings89'] = df.V16435
	df['earnings94'] = df.ER4146

	df['income89'] = df.V16413 + df.V16420 + df.V16452 + df.V16473 + df.V16430
	df['income94'] = df.ER4147 + df.ER3139 + df.ER3479 + df.ER3310 + df.ER3355 + df.ER3218 + df.ER3524

	df['wealth89'] = df.S217
	df['wealth94'] = df.S317

	# Data cleaning

		# Delete improper values (according to documentation)

		df = df[(df.ER4146 != 9999999) & (df.ER3139 != 9999999) & (df.ER3139 != 9999998) & (df.ER3479 != 9999999)
		& (df.ER3479 != 9999998) & (df.ER3310 != 99998) & (df.ER3310 != 99999) & (df.ER3355 != 999999) 
		& (df.ER3355 != 999998) & (df.ER3218 != 999999) & (df.ER3218 != 999998) & (df.ER3524 != 999999)
		& (df.ER3524 != 999998)]

# For each observation (household), identify which quintile they belong to in each year

df['quintearn89'] = pd.qcut(df['earnings89'],5,labels=[1,2,3,4,5])
df['quintearn94'] = pd.qcut(df['earnings94'],5,labels=[1,2,3,4,5])

df['quintinc89'] = pd.qcut(df['income89'],5,labels=[1,2,3,4,5])
df['quintinc94'] = pd.qcut(df['income94'],5,labels=[1,2,3,4,5])

df['quintwealth89'] = pd.qcut(df['wealth89'],5,labels=[1,2,3,4,5])
df['quintwealth94'] = pd.qcut(df['wealth94'],5,labels=[1,2,3,4,5])

# Create Table 15 (by section)

table15_earn = pd.crosstab(df['quintearn89'],df['quintearn94']
	,rownames=['1989 Earnings Quintile']
	,colnames=['1994 Earnings Quintile']).apply(lambda r: 
	round(r*100/r.sum(),1),axis=0)
table15_income = pd.crosstab(df['quintinc89'],df['quintinc94']
	,rownames=['1989 Income Quintile']
	,colnames=['1994 Income Quintile']).apply(lambda r: 
	round(r*100/r.sum(),1),axis=0)
table15_wealth = pd.crosstab(df['quintwealth89'],df['quintwealth94']
	,rownames=['1989 Wealth Quintile']
	,colnames=['1994 Wealth Quintile']).apply(lambda r: 
	round(r*100/r.sum(),1),axis=0)

# Generate LaTeX code

print(table15_earn.to_latex())
print(table15_income.to_latex())
print(table15_wealth.to_latex())