from time import time
from mutator import Race
a = time()
x = Race(use = "save0.txt")
x.grow(200, thres = 1000, show = 10, save_period = 200)
print(time()-a)

#462