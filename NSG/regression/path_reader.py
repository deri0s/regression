import sys
from pathlib import Path

FILE = Path(__file__).resolve()
root = FILE.parents[0]

print('FILE ', FILE)
print('root ', root)