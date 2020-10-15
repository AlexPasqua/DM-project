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
count = 0
for line in open('customer_supermarket.csv'):
    if count > 10:
        print(*dt, sep="\n")
        exit(1)
    if first:
        titles = line.strip().split("\t")
        first = False
    else:
        count += 1
        elem = {}
        for i in range(len(titles)):
            elem[titles[i]] = line.strip().split("\t")[i+1]
        dt.append(elem)