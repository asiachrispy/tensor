value = 99
if value == 99:
    print('it is fast')
elif value > 200:
    print('it is too fast')
else:
    print('is is safe')

##
for i in range(5):
    print(i)
##
i = 9
while i < 10:
    print(i)
    i +=1
##
mylist = [1,2,3]
print("ze val :%d" % mylist[0])
mylist.append(5)
print("list len: %d" % len(mylist))

##
mydict = {'a': 1, 'b': 2, 'c': 3}
print("A value: %d" % mydict['a'])
mydict['a'] = 11
print("A value: %d" % mydict['a'])
print("Keys: %s" % mydict.keys())
print("Values: %s" % mydict.values())
for key in mydict.keys():
    print(mydict[key])

#

# Sum function
def mysum(x, y):
    return x + y
# Test sum function
result = mysum(1, 3)
print(result)


##
import numpy
mylist = [[1, 2, 3], [3, 4, 5]]
myarray = numpy.array(mylist)
print(myarray)
print(myarray.shape)
print("First row: %s") % myarray[0]
print("Last row: %s") % myarray[-1]
print("Specific row and col: %s") % myarray[0, 2]
print("Whole col: %s") % myarray[:, 2]