import copy
inp = ["Name/Place/Name place/Animal", "Youtube"]
inp1 = [x.lower() for x in inp]
print(inp1)
inp2 = [x.split('/') for x in inp1]
print(inp2)

for row in inp2:
    for item in row:
        temp = copy.deepcopy(row)
        temp.remove(item)
        supString = "/".join(temp)
        isSubString = supString.find(item) != -1
        print(temp, item, isSubString)
