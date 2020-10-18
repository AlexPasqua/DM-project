import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

df = pd.read_csv(__file__[:-8] + 'customer_supermarket.csv', sep='\t', index_col=0)
columns = list(df)
print(columns)

#print(df['Sale'])
#print("-------------")
df['Sale'].replace(to_replace=r'(\d+),(\d*)', value=r'\1.\2', regex=True, inplace=True)
df['Sale'] = df['Sale'].astype(float)
#print(df['Sale'])
#print("-------------")


print(df.isnull().any())

print("-------------")


print(np.count_nonzero(df["ProdID"].unique()))
print(df["ProdID"].unique().size)

#fai una ricerca usando come chiave il prodotto e filtrando le righe facendo si che il prodotto sia quello e guardando la size dello unique per quanto riguarda ordine (e anche cliente)
#dove però la quantità non deve essere negativa sennò nonsense
orders_per_products = {}
clients_per_products = {}

df.info()

iter = 1
total = df["ProdID"].unique().size

for prod in df["ProdID"].unique():
    print("Prod:, {iter}/{total}".format(iter = iter, total = total))
    orders_per_products[prod] = df.loc[df["ProdID"] == prod].BasketID.unique().size
    iter = iter + 1
    #print(orders_per_products[prod])

max_used = max(orders_per_products, key = orders_per_products.get)
min_used = min(orders_per_products, key = orders_per_products.get)

print("Most present: {max} with {values} values".format(max = max_used, values = orders_per_products[max_used]))
print("Less present: {min} with {values} values".format(min = min_used, values = orders_per_products[min_used]))

df.info()

print("I did it!")


