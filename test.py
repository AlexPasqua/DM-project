""" stucture of data
[
    {"A": 1, "B" : 2},
    {"A": 1, "B" : 2},
    .
    .
    .
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
count_notvalid_rows = 0
for row in q_neg:
    if 'C' not in row['BasketID']:
        #print(row)
        bid_id_neg = False
        cid_null = cid_null and row['CustomerID'] == ''
        count_notvalid_rows += 1
print("All entries had a special BasketID:", bid_id_neg)
print("Rows with negative Qta but not strange id: ", count_notvalid_rows)
print("All entries has empty CustomerID: ", cid_null)
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
