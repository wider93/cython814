import pstats, cProfile

import pyximport
pyximport.install()
import os
if not os.path.exists("pf"):
    os.makedirs("pf")
i = 0
while os.path.exists(f"pf/Profile{i}.prof"):
    i += 1

profile_name = f"pf/Profile{i}.prof"

import mutator_partof as M

cProfile.runctx("x = M.Race(use = 'save2.txt'); x.grow(2000, thres = 550, show = 10, save_period = 500)", globals(), locals(), profile_name)
#cProfile.runctx("x = M.Race(use = 'save0.txt'); x.grow(50, thres = 1000, show = 10, save_period = 50)", globals(), locals(), profile_name)

s = pstats.Stats(profile_name)
s.strip_dirs().sort_stats("time").print_stats()