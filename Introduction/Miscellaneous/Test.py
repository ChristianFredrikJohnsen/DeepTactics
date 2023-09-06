from collections import defaultdict
from itertools import product
import numpy as np
import torch
import wandb
import sys

print("Hello World!")

dictor = defaultdict(lambda: [0.0] * 2)


dictor['Peter'][0] = 4.5
dictor['Peter'][1] = 9

print(dictor)

fred = ("bob", "petra", "petter")

print(fred[1])

print(int(100_000))
print(100_000)


def my_function():
    a = 10  # 'a' is local to this function
    if True:
        b = 5  # 'b' is also local to this function, not just to the 'if' block
    print(a, b)  # prints: 10 5

my_function()

def klubb():
    for i in range (10):
        b = i
    print(b)

def kløbb():
    i = 0
    while (i < 10):
        i += 1
        b = i

    print(b)

klubb()
kløbb()

grog = [1, 3, 4]
ghu = [1, 1, 1]
gruss = [a + b for a, b in zip(grog, ghu)]
print(grog + ghu)
print(gruss)

print("Klaus incoming: \n")
gogg = [[1, 2], [1, 8], [1, 3]]
sad_gogg = [[1, 2], [1, 8], [1]]

klaus = zip(*gogg)
klaus_bro = zip(*sad_gogg)

for key in klaus:
    print(key)

print("Klaus bro:\n")
for key in klaus_bro:
    print(key)

print(2*ghu)

dealer = [i for i in range(1, 11)]
player = [i for i in range(1, 22)]
states = product(dealer, player)
# for state in states:
#     print(state)

grubb = np.zeros(5)
grabb = np.zeros(5)

grabb[0], grabb[1] = 8, 2

print(grubb + grabb)

# print(torch.cuda.is_available())

print(np.array([[1, 2], [1, 2]]))

print(sys.path)
