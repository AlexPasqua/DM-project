""" stucture of data
[
    {"A": 1, "B" : 2},
    {"A": 1, "B" : 2},
]
['BasketID', 'BasketDate', 'Sale', 'CustomerID', 'CustomerCountry', 'ProdID', 'ProdDescr', 'Qta']
"""
dt = []
first = True
for line in open('customer_supermarket.csv'):
    if first:
        titles = line.strip().split("\t")
        first = False
    else:
        elem = {}
        for i in range(len(titles)):
            elem[titles[i]] = line.strip().split("\t")[i+1]
        dt.append(elem)

print("Total rows: ", len(dt))
print("-"*50)

print("Analyzing entries with negative Qta")
q_neg = [row for row in dt if int(row['Qta']) < 0]
print("Rows with quantity negative: ", len(q_neg))
bid_id_neg = True
cid_null = True
count_pdesc_null = 0
count_bid_valid = 0
for row in q_neg:
    if 'C' not in row['BasketID']:
        bid_id_neg = False
        count_bid_valid += 1
    if row['ProdDescr'] in ['', '?']:
        count_pdesc_null += 1
    cid_null = cid_null and row['CustomerID'] == ''
print("All entries had a special BasketID:", bid_id_neg)
print("Rows with negative Qta but normal id: ", count_bid_valid)
print("All entries has empty CustomerID: ", cid_null)
print("Rows with empty or '?' ProdDescr: ", count_pdesc_null)
print("-"*50)

print("Checking if each negative entry has a positive one...")
#all_haspair = True
#not_paired = 0
#counter = 0
#total = len(q_neg)
#for row in q_neg:
#    counter += 1
#    print(f"{counter}/{total}")
#    found = False
#    for elem in [e for e in dt if e['BasketID'] == row['BasketID']]:
#        if row['ProdID'] == elem['ProdID'] and int(elem['Qta']) > 0:
#            print(row['Qta'], elem['Qta'])
#            found = True
#            break
#    if not found:
#        all_haspair = False
#        not_paired += 1
#print("All negative entries has a positive one: ", all_haspair) #False
#print("The negative entries which are not matched are: ", not_paired) #9752 (All)
print("All negative entries has a positive one: ", False)
print("The negative entries which are not matched are: ", 9752)
print("-"*50)

print("Checking rows with no CustomerID")
tot_orders = set(r['BasketID'] for r in dt)
order_has_customer = { o:False for o in tot_orders}
for r in dt:
    order_has_customer[r['BasketID']] = True if r['CustomerID'] != '' else False
count_empty = sum(1 for _, k in order_has_customer.items() if not k)
print("Number of distinct orders: ", len(tot_orders))
print("Number of orders without CustomerID: ", count_empty)
print("-"*50)

print("Analyzing CustomerID values")
customers = set([r['CustomerID'] for r in dt if r['CustomerID'] != ''])
order_for_customer = { c:0 for c in customers}
last_order_customer = { c:set() for c in customers}
for r in dt:
    customer = r['CustomerID']
    if customer != '':
        if r['BasketID'] not in last_order_customer[customer]:
            order_for_customer[customer] += 1
            last_order_customer[customer].add(r['BasketID'])
mean_num_order = sum(order_for_customer[c] for c in customers)/len(customers)
min_num_order = min(order_for_customer[c] for c in customers)
max_num_order = max(order_for_customer[c] for c in customers)
print("Number of distinct Customers: ", len(customers))
print("Mean number of orders for each Customer: ", mean_num_order)
print("Min number of order for a Customer: ", min_num_order)
print("Max number of order for a Customer: ", max_num_order)
print("Number of customers with number of order [0-50]: ", sum(1 for c in customers if order_for_customer[c] <= 50))
print("Number of customers with number of order [51-100]: ", sum(1 for c in customers if order_for_customer[c] > 50 and order_for_customer[c] <= 100))
print("Number of customers with number of order [100-max]: ", sum(1 for c in customers if order_for_customer[c] > 100))
assert(len(tot_orders) - count_empty == sum(order_for_customer[c] for c in customers))
products_for_customer = { c:list() for c in customers}
for r in dt:
    if r['CustomerID'] != '':
        products_for_customer[r['CustomerID']].append(r['ProdID'])
mean_num_prod = sum(len(products_for_customer[c]) for c in customers)/len(customers)
min_num_prod = min(len(products_for_customer[c]) for c in customers)
max_num_prod = max(len(products_for_customer[c]) for c in customers)
print("Mean number of products for each Customer: ", mean_num_prod)
print("Min number of products for a Customer: ", min_num_prod)
print("Max number of products for a Customer: ", max_num_prod)
unique_products = set(r['ProdID'] for r in dt)
print("Number of unique products: ", len(unique_products))
print("-"*50)