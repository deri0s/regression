import sys
from pathlib import Path

FILE = Path(__file__).resolve()
root = FILE.parents[0]

print('FILE ', FILE)
print('root ', root)

a = []
b = [1,1,1,1,1]
c = [2,2,2]
a.extend(b)
print('a: \n', a)