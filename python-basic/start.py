#! /usr/bin/env python
#coding:utf-8

import math

a = math.floor(32.8)
b = math.sqrt(4)

print("py" + "thon")
print("python".isupper())

c = 'this is python'
print(c.__sizeof__()) #63
print(len(c)) #14

print(c[0])
print(c[1])
print(c[2])
print(c[3])
print(c[4])

print(c[-1])
print(c[13])

print(c[:4])

print(c[5:7])

#coding:utf-8

print("please write your name:")
name = raw_input()
print("Hello,%s"%name)

import math
a = math.floor(32.8)
b = math.sqrt(4)

print a
print b

type(a) #<type 'int'>

c='hello'
type(c) #<type 'str'>

print "py"+"thon"
print "one is %d"%1

print "what's your name?"   #双引号包裹单引号，单引号是字符

print 'what "is your" name' #单引号包裹双引号，双引号是字符

## for
a = 199
b = 'free'

##error print a+b
print b + `a` #注意，` `是反引号，不是单引号，就是键盘中通常在数字1左边的那个，在英文半角状态下输入的符号
print b + str(a)    #str(a)实现将整数对象转换为字符串对象
print b + repr(a)   #repr(a)与上面的类似

def add_function(a,b):
    c = a+b
    print c

if __name__=="__main__":
    add_function(2,3)

