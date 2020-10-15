""" stucture of data
[
    {"A": 1, "B" : 2},
    {"A": 1, "B" : 2},
    .
    .
    .
]
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

print(len(dt))

print(sum(1 for row in dt if int(row['Qta']) < 0))