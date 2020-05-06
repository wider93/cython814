# cython: profile=True
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
# distutils: language=c++

import xxhash
cimport cython
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdlib cimport malloc, free
from time import time
from tqdm import tqdm
import random
import os

rng = np.random.default_rng()
ctypedef (int, int) inttuple
cdef vector[inttuple] dxdy = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
cdef int r = 8, c = 14, rc = 112 # r*c
cdef int make_log = 1

cdef char* foldername = b'storage'
cdef char* usedindexname = b'used.txt'
cdef str log_name = 'log_'
if not os.path.exists(foldername):
    os.makedirs(foldername)

cdef vector[string] frag_path = os.listdir(foldername)
cdef size_t frag_len = frag_path.size()

# 32bit or 64bit
DTYPE = np.int64
ctypedef np.int64_t DTYPE_t
hashing = xxhash.xxh32()

ctypedef unsigned int uint
ctypedef long long lld

'''cdef extern from "cfunc.cpp":
    # C++ is include here so that it doesn't need to be compiled externally
    pass
cdef extern from "cfunc.h":
    cdef vector[vector[int]] vecvec(int size)'''

'''cdef vector[vector[int]] vecvec(int size):
    cdef vector[vector[int]] ans = []
    cdef vector[int] tmp
    ans.reserve(size)
    cdef int i
    for i in range(size):
        tmp = []
        ans.push_back(tmp)
    return ans'''


cdef vector[vector[int]] grid_map
cdef vector[int] unused_points
cdef int *used = <int *>malloc(rc * sizeof(int))
grid_map = []

cdef void fix_used():
    global used, unused_points
    cdef int i
    cdef str line, j
    assert os.path.exists(usedindexname)
    with open(usedindexname, 'r') as file:
        line = file.readline().rstrip()
        for i, j in enumerate(line):
            used[i] = int(j)

    for m in range(rc):
        if not used[m]:
            unused_points.push_back(m)

cdef void calc_map(int restrict = 1):
    cdef int x, y, nx, ny, i, j, m
    cdef vector[int] path
    global grid_map
    grid_map = []
    for m in range(rc):
        path = []
        x = m // c
        y = m % c
        for i, j in dxdy:
            nx = x + i
            ny = y + j
            if 0 <= nx < r and 0 <= ny < c:
                path.push_back(c*nx+ny)
        grid_map.push_back(path)

fix_used()
calc_map(1)
np.random.seed(int(time()))

cdef int bound_by_global_max = 1000
cdef int bound_10 = 100

cdef void renew_global_max(int k):
    cdef int m
    global bound_by_global_max, bound_10
    m = (k + 49) // 50
    if m > 10 and m > bound_10:
        bound_10 = m
        bound_by_global_max = m * 10

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int score(lld [:] f):
    cdef int current, x, nx, top, j
    cdef size_t i
    cdef vector[vector[int]] first, second
    cdef vector[int] que, new_que
    cdef int[112] check
    cdef int[10] fastcount
    first = [[] for i in range(10)]#vecvec(10)
    for x in range(rc):
        first[f[x]].push_back(x)
    for j in range(1, 10):
        if not first[j].size():
            return j - 1
    current = 9

    for j in range(rc): check[j] = 0
    for top in range(1, bound_10+1):
        second = [[] for i in range(10)]#vecvec(10)
        with nogil:
            que = first[top]
            for i in range(que.size()):
                for nx in grid_map[que[i]]:
                    if check[nx] == 0:
                        second[f[nx]].push_back(nx)
                        check[nx] = 1
        for i in range(10):
            new_que = second[i]
            if new_que.size() == 0:
                return current
            for j in new_que: check[j] = 0
            current += 1
            first.push_back(new_que)

    fastcount = [0]*10
    for top in range(bound_10 + 1, bound_by_global_max):
        with nogil:
            for i in range(10): fastcount[i] = 0
            for j in range(rc): check[j] = 0
            que = first[top]
            for i in range(que.size()):
                for nx in grid_map[que[i]]:
                    if not check[nx]:
                        fastcount[f[nx]] = 1
                        check[nx] = 1
        for i in range(10):
            if not fastcount[i]:
                return current
            current += 1

cdef int int_16(j):
    return int(j) if j != 'a' else rng.integers(10)

cdef np.ndarray parse_to_array(list note):
    cdef str liststring = ''.join(note)
    return np.array([int_16(j) for j in liststring], dtype = DTYPE)

cdef class Individual:
    cdef readonly int grade
    cdef public np.ndarray gene
    cdef readonly unsigned long hash

    def __init__(self, np.ndarray[DTYPE_t, ndim = 1] f):
        self.gene = f
        self.grade = score(self.gene)
        hashing.update(f)
        self.hash = hashing.intdigest()
        hashing.reset()

    def __hash__(self):
        return self.hash

    def __eq__(self, Individual other):
        return self.hash == other.hash and self.grade == other.grade and (self.gene == other.gene).all()

    def __ge__(self, Individual other):
        return self.grade >= other.grade

    def __gt__(self, Individual other):
        return self.grade > other.grade

    def __le__(self, Individual other):
        return self.grade <= other.grade

    def __lt__(self, Individual other):
        return self.grade < other.grade

    def __str__(self):
        zy = self.gene.reshape((r, c))
        a = '\n'.join(np.array2string(zyy, separator = '', prefix = '', suffix = '')[1:-1] for zyy in zy)
        return '\n'.join([f'ans:{self.grade}', a])



# Generate with no input

cpdef Individual fully_new():
    return Individual(rng.integers(10, size = rc))

cpdef np.ndarray read_from_file():
    cdef size_t i
    cdef string use
    cdef list pile, one
    cdef int j
    cdef np.ndarray[DTYPE_t, ndim = 1] two
    if frag_len == 0:
        return fully_new()
    j = rng.integers(frag_len)
    use = frag_path[j]
    with open(os.path.join(foldername, use), 'r') as file:
        pile = file.readlines()
        one = [row.rstrip() for row in pile[1:]]
        two = parse_to_array(one) # Note: Here we assume unassigned values are stored in one letter, i.e. 'a'
    return two

cdef Individual random_new():
    cdef int k = rng.integers(5)
    if k:
        return Individual(read_from_file())
    return fully_new()

# Using one
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray random_balance(np.ndarray[DTYPE_t, ndim = 1] f):
    cdef np.ndarray[DTYPE_t, ndim = 1] g
    cdef int m
    m = rng.integers(10)
    g = (f + np.arange(m, m + rc, dtype = DTYPE)) % 10
    np.random.shuffle(g)
    return g

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray mutate_some(np.ndarray[DTYPE_t, ndim = 1] f):
    cdef np.ndarray[DTYPE_t, ndim = 1] g
    cdef size_t num
    g = f.copy()
    num = rng.integers(1, 3)
    g[np.random.choice(rc, num)] = rng.integers(10, size = num)
    return g

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray mutate_more(np.ndarray[DTYPE_t, ndim = 1] f):
    cdef np.ndarray[DTYPE_t, ndim = 1] g
    cdef size_t num
    g = f.copy()
    num = rng.integers(1, 5)
    g[np.random.choice(rc, num)] = rng.integers(10, size = num)
    return g

cpdef np.ndarray push_some_row(np.ndarray[DTYPE_t, ndim = 1] f):
    cdef np.ndarray[DTYPE_t, ndim = 1] g, x, rands, px
    cdef int i, a, b, p
    g = f.copy()
    rands = rng.integers(3, size = r)
    px = rng.integers(c, size = 2*r)
    for i in range(r):
        if rands[i] == 0:
            a = c * i
            b = a + c
            x = g[a:b]
            p = px[2*i]
            x[p:c-1] = x[p+1:]
            p = px[2*i+1]
            x[p+1:] = x[p:c-1]
            x[p] = rng.integers(10)
            g[a:b] = x
    return g

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray swap_in_row(np.ndarray[DTYPE_t, ndim = 1] f):
    cdef np.ndarray[DTYPE_t, ndim = 1] g
    cdef int i, k, j, tmp
    g = f.copy()
    for i in range(3):
        k = rng.integers(r)
        j = rng.integers(c*k, c*(k+1)-1)
        g[j], g[j+1] = g[j+1], g[j]
    return g

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray swap_in_col(np.ndarray[DTYPE_t, ndim = 1] f):
    cdef np.ndarray[DTYPE_t, ndim = 1] g
    cdef int j
    g = f.copy()
    for j in rng.integers(rc - c, size = 4):
        g[j], g[j+c] = g[j+c], g[j]
    return g

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray swap_next(np.ndarray[DTYPE_t, ndim = 1] f):
    cdef np.ndarray[DTYPE_t, ndim = 1] g
    cdef int i, j, k, n, m, tmp
    g = f.copy()
    n = rng.integers(2, 6)
    for i in range(n):
        j = rng.integers(rc)
        m = rng.integers(len(grid_map[j]))
        k = grid_map[j][m]
        g[j], g[k] = g[k], g[j]
    return g

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray shuffle_square(np.ndarray[DTYPE_t, ndim = 1] f):
    cdef np.ndarray[DTYPE_t, ndim = 1] g, g1, g2
    cdef int a, b, k
    g = f.copy()
    k = rng.integers(2, 5)
    a = rng.integers(r-k)
    b = rng.integers(c-k)
    g1 = (np.arange(a, a+k, dtype = DTYPE).reshape((k, 1)) * c + np.arange(b, b+k, dtype = DTYPE).reshape((1, k))).reshape(-1)
    g2 = g1.copy()
    np.random.shuffle(g1)
    g[g1] = g[g2]
    return g

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray apply_permutation(np.ndarray[DTYPE_t, ndim = 1] f):
    cdef np.ndarray[DTYPE_t, ndim = 1] perm, g
    perm = np.arange(10, dtype = DTYPE)
    np.random.shuffle(perm)
    g = perm[f]
    return g

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray force_next(np.ndarray[DTYPE_t, ndim = 1] f):
    cdef int current, x, nx, top, j
    cdef size_t i, ii, iii, m, mi, a
    cdef np.ndarray[DTYPE_t, ndim = 1] g
    cdef vector[vector[int]] first, second
    cdef vector[int] que, new_que
    cdef int[112] check
    cdef const lld[:] view = f
    first = [[] for i in range(10)]#vecvec(10)
    for x in range(rc):
        first[view[x]].push_back(x)
    for j in range(1, 10):
        if not first[j].size():
            return rng.integers(10, size = rc)

    for j in range(rc): check[j] = 0
    current = 9
    g = f.copy()
    for top in range(1, bound_by_global_max):
        second = [[] for i in range(10)]#vecvec(10)
        with nogil:
            que = first[top]
            for j in que:
                for nx in grid_map[j]:
                    if check[nx] == 0:
                        second[view[nx]].push_back(nx)
                        check[nx] = 1
            for i in range(10):
                new_que = second[i]
                if new_que.size() == 0:
                    m = 0
                    for iii in range(10):
                        mi = second[iii].size()
                        if mi > m:
                            m = mi
                            ii = iii
                    with gil:
                        j = rng.integers(m)
                        a = second[ii][j]
                        g[a] = i
                        return g
                current += 1
                for j in new_que:
                    check[j] = 0
                if top <= bound_10:
                    first.push_back(new_que)
    return f # just in case

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray another_template(np.ndarray[DTYPE_t, ndim = 1] f):
    cdef np.ndarray[DTYPE_t, ndim = 1] h
    cdef lld[:] view
    cdef int[56] rand
    cdef int i
    h = read_from_file()
    view = h
    with nogil:
        for i in unused_points:
            view[i] = f[i]
    return h


table = [(mutate_some, 0.1 ), (apply_permutation, 0.4), (mutate_more, 0.6), (push_some_row, 1.0), (force_next, 1.),
         (shuffle_square, 1.4), (swap_in_col, .5), (swap_in_row, .5), (swap_next, .5), (random_balance, .5), (another_template, .5)]

# List of functions which takes 2 arguments

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray split_row(np.ndarray[DTYPE_t, ndim = 1] f, np.ndarray[DTYPE_t, ndim = 1] g):
    cdef np.ndarray[DTYPE_t, ndim = 1] h
    cdef lld[:] view
    cdef int a, b, i, j, x, y
    h = f.copy()
    view = h
    a = rng.integers(2)
    b = rng.integers(1, c)
    if a == 0:
        x, y = b, c
    else:
        x, y = 0, b
    for i in range(x, y):
        for j in range(i, rc, c):
            view[j] = g[j]
    return h

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray split_col(np.ndarray[DTYPE_t, ndim = 1] f, np.ndarray[DTYPE_t, ndim = 1] g):
    cdef np.ndarray[DTYPE_t, ndim = 1] h
    cdef int a, b, j, x, y
    cdef lld[:] view
    h = f.copy()
    view = h
    a = rng.integers(2)
    b = rng.integers(1, r)*c
    if a == 0:
        x, y = b, c
    else:
        x, y = 0, b
    for j in range(x, y):
        view[j] = g[j]
    return h

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray flavor(np.ndarray[DTYPE_t, ndim = 1] f, np.ndarray[DTYPE_t, ndim = 1] g):
    cdef np.ndarray[DTYPE_t, ndim = 1] h
    cdef int[14] x
    cdef int a, b
    cdef lld[:] view
    h = f.copy()
    view = h
    b = rng.integers(rc)
    x = rng.integers(r, size = c)
    for i in range(c):
        a = c*x[i]+i+b
        if a >= rc: a -= rc
        view[a] = g[a]
    return h

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray replace_rect(np.ndarray[DTYPE_t, ndim = 1] f, np.ndarray[DTYPE_t, ndim = 1] g):
    cdef np.ndarray[DTYPE_t, ndim = 1] h, g1, g2
    cdef int a, b, k, i
    k = rng.integers(2, 5)
    a = rng.integers(r-k)
    b = rng.integers(c-k)
    g1 = (np.arange(a, a+k, dtype = DTYPE).reshape((k, 1)) * c + np.arange(b, b+k, dtype = DTYPE).reshape((1, k))).reshape(-1)
    a = rng.integers(r-k)
    b = rng.integers(c-k)
    g2 = (np.arange(a, a+k, dtype = DTYPE).reshape((k, 1)) * c + np.arange(b, b+k, dtype = DTYPE).reshape((1, k))).reshape(-1)
    np.random.shuffle(g2)
    i = rng.integers(2)
    if i == 0:
        h = f.copy()
        h[g1] = g[g2]
    else:
        h = g.copy()
        h[g1] = f[g2]
    return h

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray pairing(np.ndarray[DTYPE_t, ndim = 1] f, np.ndarray[DTYPE_t, ndim = 1] g):
    cdef np.ndarray[DTYPE_t, ndim = 1] h
    cdef lld[:] view
    cdef int[56] rand
    cdef int i, j
    h = f.copy()
    view = h
    rand = rng.integers(2, size = rc//2)
    for i in range(rc//2):
        j = 2*i+rand[i]
        view[j] = g[j]
    return h

cpdef np.ndarray add(np.ndarray[DTYPE_t, ndim = 1] f, np.ndarray[DTYPE_t, ndim = 1] g):
    cdef np.ndarray[DTYPE_t, ndim = 1] h
    h = (f + g[rc-1::-1]) % 10
    return h

table2 = [(split_col, 1.), (split_row, .7), (flavor, .6), (replace_rect, 1.), (pairing, 1.), (add, .7)]


cdef generate_table(list table):
    cdef list pool = []
    cdef vector[float] cum_weights = []
    cdef float accumulated = 0., weight
    cdef n = len(table)
    cum_weights.reserve(n)
    for func, weight in table:
        accumulated += weight
        pool.append(func)
        cum_weights.push_back(accumulated)
    return pool, cum_weights

cdef list generators_pool, generators_pool2
cdef vector[float] generators_weight, generators_weight2

generators_pool, generators_weight = generate_table(table)
generators_pool2, generators_weight2 = generate_table(table2)

cdef vector[int] stack_pop(list population):
    cdef Individual ele
    cdef int a = 0
    cdef size_t n
    cdef vector[int] stack
    n = len(population)
    #n = 1000
    stack.reserve(n)
    for ele in population:
        a += ele.grade
        stack.push_back(a)
    return stack


cdef class Race:
    cdef readonly int max_, prev_max_
    cdef readonly list population
    cdef readonly str use_, log_
    def __init__(self, use = ''):
        population = []
        renew_global_max(1000)
        if use:
            if not os.path.exists(use):
                with open(use, 'w'):
                    pass
            with open(use, 'r') as file:
                piles = file.readlines()
                print(f"raw file lines: {len(piles)}")
                m = r+1
                for k in range(0, len(piles), m):
                    pile = piles[k+1:k+m]
                    one = [row.rstrip() for row in pile]
                    two = Individual(parse_to_array(one))
                    population.append(two)
            print(f"length = {len(population)}")
            self.use_ = use
        else:
            self.use_ = "save0.txt"

        print(f"using {self.use_}")
        n = len(population)
        self.population = population
        self.renew()
        print(f"neo_length = {len(population)}")
        self.population.sort(reverse = True)
        self.prev_max_ = self.max_ = self.population[0].grade
        renew_global_max(self.max_)
        print("initialize complete")
        self.log_ = ''.join([log_name, use])

    cdef void progress(self):
        cdef set new_population
        cdef int t, a, n = 200, best_ = 150, survives_ = 750, rands_ = 100
        cdef list p_list, q, generators, generators2
        cdef np.ndarray[np.double_t, ndim = 1] prob
        cdef np.ndarray[DTYPE_t, ndim = 1] tmp2
        cdef vector[int] cum_grade
        cdef Individual ele, ele2, tmp
        generators = random.choices(generators_pool, cum_weights = generators_weight, k = 5)  #
        new_population = set(self.population)
        # step 1: modifying one element
        for ele in self.population:
            func = random.choice(generators)
            tmp2 = func(ele.gene)
            tmp = Individual(tmp2)
            new_population.add(tmp)
        for ele in self.population:
            func, func2 = random.choices(generators, k = 2)
            tmp2 = func2(func(ele.gene))
            tmp = Individual(tmp2)
            new_population.add(tmp)
        # step 2: combining two elements
        cum_grade = stack_pop(self.population)
        q = random.choices(self.population, cum_weights = cum_grade, k = 2*n)

        func, func2 = random.choices(generators_pool2, cum_weights = generators_weight2, k = 2)
        for a in range(n):
            ele = q[a]
            ele2 = q[a + n]
            tmp2 = func(ele.gene, ele2.gene)
            tmp = Individual(tmp2)
            new_population.add(tmp)
            tmp2 = func(ele2.gene, func2(ele2.gene, ele.gene))
            tmp = Individual(tmp2)
            new_population.add(tmp)

        # Now update population
        p_list = sorted(new_population)
        # First, pop best m candidates
        self.population = p_list[:-best_-1:-1]
        p_list[-best_:] = []
        t = self.population[0].grade - self.max_
        if t > 0:
            self.max_ += t
            renew_global_max(self.max_)
            if make_log:
                with open(self.log_, 'a') as file:
                    tmp = self.population[0]
                    file.writelines(str(tmp))
                    file.writelines('\n')

        # And choose randomly from the rest
        cum_grade = stack_pop(p_list)
        t = cum_grade[cum_grade.size()-1]
        prob = np.array(cum_grade, dtype = np.double)
        prob[1:] -= prob[:-1]
        prob /= t
        for i in np.random.choice(len(p_list), survives_, p = prob, replace = False):
            self.population.append(p_list[i])

        # Always generate some fresh meat
        for a in range(rands_):
            self.population.append(random_new())

    cdef recalc_elements(self, int level):
        cdef Individual ele
        calc_map(level)
        self.population = [Individual(ele.gene) for ele in self.population]
        self.population.sort(reverse = True)
        self.max_ = self.population[0].grade
        print(f"MAX SCORE = {self.max_}")

    cdef cutoff(self, int thres):
        # bisect
        cdef int mid, left = 0, right = 1000 # len(population)
        while right > left + 1:
            mid = (right + left) >> 1
            if self.population[mid].grade >= thres:
                left = mid
            else:
                right = mid
        self.population[mid:] = []

    cdef renew(self):
        cdef int n = len(self.population)
        for i in range(1000 - n):
            self.population.append(fully_new())

    cpdef grow(self, uint epoch, int thres = 1000, uint show = 10, uint save_period = 200):
        cdef uint it
        cdef Individual x
        cdef double check
        assert save_period > 0 and show > 0 and save_period % show == 0
        check = time()
        for it in range(1, epoch + 1):
            self.progress()

            if it % show == 0:
                print(f"epoch = {it:<5}, max_score = {self.max_:<4}, elapsed = {time() - check:.2f}s, num = {len(self.population)}")
                if it % save_period == 0:
                    self.population.sort(reverse = True)
                    self.cutoff(thres)
                    if self.max_ > self.prev_max_:
                        print(f"previous max = {self.prev_max_}, now max = {self.max_}. Saving new values...")
                        with open(self.use_, 'w') as file:
                            for x in tqdm(self.population):
                                #if x.grade >= thres:  #already cut-off
                                file.writelines(str(x))
                                file.writelines('\n')

                        self.prev_max_ = self.max_
                        print("save complete.")
                    self.renew()
                    np.random.seed(int(check))
                check = time()