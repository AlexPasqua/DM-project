import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import re

#---------------------------------------------------------------------------------------------------------#
#FUNCTS

def _write_to_file(text, title):
    path_to_file = __file__[:-8] + title + ".txt"
    print("PATH TO RESULT FILE: ", path_to_file)
    with open(path_to_file, "w") as f:
        f.write(text)

def _create_customers_per_product(filtered_dataframe):
    np_customers = filtered_dataframe.to_numpy()
    nan_array = np.isnan(np_customers)
    not_nan_array = ~nan_array
    return np.unique(np_customers[not_nan_array]).size


#---------------------------------------------------------------------------------------------------------#
#MAIN CODE


df = pd.read_csv(__file__[:-8] + 'customer_supermarket.csv', sep='\t', index_col=0)

df['Sale'].replace(to_replace=r'(\d+),(\d*)', value=r'\1.\2', regex=True, inplace=True)
df['Sale'] = df['Sale'].astype(float)


#print(df.isnull().any())

result_text = ""

result_text += "Unique ProdID count:" + str(df["ProdID"].unique().size)
result_text += "\nUnique BasketID count:" + str(df["BasketID"].unique().size)
result_text += "\nUnique CustomerID count:" + str(df["CustomerID"].unique().size)

#fai una ricerca usando come chiave il prodotto e filtrando le righe facendo si che il prodotto sia quello e guardando la size dello unique per quanto riguarda ordine (e anche cliente)
#dove però la quantità non deve essere negativa sennò nonsense
orders_per_products = {}
customers_per_products = {}

df.info()

iter = 1
total = df["ProdID"].unique().size
print(df.keys())

starting_non_digits = {}

for basketID in df.BasketID.unique():
    if re.match("\D[0-9]+|[0-9]+\D", basketID) != None:
        if basketID[0] not in starting_non_digits:
            starting_non_digits[basketID[0]] = 0
        starting_non_digits[basketID[0]] += 1

result_text += "\n-----BASKET IDS CAN START WITH STRANGE VALUES------\n\n" + str(starting_non_digits) + "\n\n---------------\n"

for prod in df["ProdID"].unique():
    print("Prod: {iter}/{total}".format(iter = iter, total = total))
    orders_per_products[prod] = df.loc[df["ProdID"] == prod].BasketID.unique().size
    customers_per_products[prod] = _create_customers_per_product(df.loc[df["ProdID"] == prod,["CustomerID"]])

    iter = iter + 1



max_used_o = max(orders_per_products, key = orders_per_products.get)
min_used_o = min(orders_per_products, key = orders_per_products.get)

max_used_c = max(customers_per_products, key = customers_per_products.get)
min_used_c = min(customers_per_products, key = customers_per_products.get)

result_text += "\n\nMost present product in orders is code {max} with {values} and {description}".format(max = max_used_o, values = orders_per_products[max_used_o], description = df.loc[df["ProdID"] == max_used_o, ["ProdDescr"]].to_numpy()[0])

result_text += "\nLess present product in orders: {min} with {values} values and {description}".format(min = min_used_o, values = orders_per_products[min_used_o], description = df.loc[df["ProdID"] == min_used_o, ["ProdDescr"]].to_numpy()[0])

result_text += "\n\nMost present product in clients: {max} with {values} values and {description}".format(max = max_used_c, values = customers_per_products[max_used_c], description = df.loc[df["ProdID"] == max_used_c, ["ProdDescr"]].to_numpy()[0])
result_text += "\nLess present product in clients: {min} with {values} values and {description}".format(min = min_used_c, values = customers_per_products[min_used_c], description = df.loc[df["ProdID"] == min_used_c, ["ProdDescr"]].to_numpy()[0])

_write_to_file(result_text, "results")

df.info()

print("I did it!")


