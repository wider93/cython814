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

import mutator

cProfile.runctx("x = mutator.Race(use = 'save1.txt'); x.grow(200, thres = 1000, show = 10)", globals(), locals(), profile_name)

s = pstats.Stats(profile_name)
s.strip_dirs().sort_stats("time").print_stats()