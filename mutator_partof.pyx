# cython: profile=True
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
# distutils: language=c++

'''
This file is to generating good fragment for good random initials.

'''

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
cdef int use_num = 80, nouse_num = rc - use_num
cdef int make_log = 0
cdef int gtmp

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

cdef vector[vector[int]] grid_map
cdef vector[int] unused_points, used_points # will be sorted
cdef int *used = <int *>malloc(rc * sizeof(int))
grid_map = []

cdef void fix_used():
    global used
    cdef int i
    cdef str line, j
    if os.path.exists(usedindexname):
        with open(usedindexname, 'r') as file:
            line = file.readline().rstrip()
            for i, j in enumerate(line):
                used[i] = int(j)
    else:
        for i in range(rc):
            used[i] = 0
        for gtmp in rng.choice(rc, size = use_num, replace = False):
            used[gtmp] = 1
        with open(usedindexname, 'w') as file:
            for i in range(rc):
                file.write(str(used[i]))
            file.write('\n')

np.random.seed(8140)
fix_used()

cdef void calc_map(int restrict = 1):
    cdef int x, y, nx, ny, i, j, m
    cdef vector[int] path
    global grid_map, unused_points, used_points
    grid_map = []
    unused_points = []
    for m in range(rc):
        path = []
        if restrict or used[m] == 1:
            x = m // c
            y = m % c
            for i, j in dxdy:
                nx = x + i
                ny = y + j
                if 0 <= nx < r and 0 <= ny < c and used[c*nx+ny]:
                    path.push_back(c*nx+ny)
        grid_map.push_back(path)

    for m in range(rc):
        if grid_map[m].size():
            used_points.push_back(m)
        else:
            unused_points.push_back(m)

calc_map(0)
np.random.seed(int(time()*814) % (1 << 32))

print(used_points.size())

cdef int bound_by_global_max = 100
cdef int bound_10 = 10

cdef void renew_global_max(int k):
    cdef int m
    global bound_by_global_max, bound_10
    m = (k + 49) // 50
    if m > 10 and m > bound_10:
        bound_10 = m
        bound_by_global_max = m * 10

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef int score(np.ndarray[DTYPE_t, ndim = 1] f):
    cdef int current, x, nx, top, j
    cdef size_t i
    cdef vector[vector[int]] first, second
    cdef vector[int] que, new_que
    cdef int[112] check
    cdef int[10] fastcount
    first = [[] for j in range(10)]
    for x in used_points:
        first[f[x]].push_back(x)
    for j in range(1, 10):
        if not first[j].size():
            return j - 1
    current = 9
    for j in range(rc): check[j] = 0
    for top in range(1, bound_10+1):
        second = [[] for i in range(10)]
        que = first[top]
        for j in que:
            for nx in grid_map[j]:
                if check[nx] == 0:
                    second[f[nx]].push_back(nx)
                    check[nx] = 1
        for i in range(10):
            new_que = second[i]
            if new_que.size() == 0:
                return current
            current += 1
            first.push_back(new_que)
            for j in new_que:
                check[j] = 0

    fastcount = [0]*10
    for top in range(bound_10 + 1, bound_by_global_max):
        for i in range(10): fastcount[i] = 0
        for j in used_points: check[j] = 0
        que = first[top]
        for j in que:
            for nx in grid_map[j]:
                if not check[nx]:
                    fastcount[f[nx]] = 1
                    check[nx] = 1
        for i in range(10):
            if not fastcount[i]:
                return current
            current += 1

cdef long_term_score


cdef int int_fit(str j):
    return int(j) if j != 'a' else rng.integers(10)

cdef np.ndarray parse_to_array(list note):
    cdef str j, liststring = ''.join(note)
    cdef list tolist = [int_fit(j) for j in liststring]
    return np.array(tolist, dtype = DTYPE)

cdef class Individual:
    cdef readonly int grade
    cdef public np.ndarray gene
    cdef readonly unsigned long hash
    def __init__(self, np.ndarray[DTYPE_t, ndim = 1] f):
        self.gene = f
        self.grade = score(f)
        g = f.copy()
        g[unused_points] = 10
        hashing.update(g)
        self.hash = hashing.intdigest()
        hashing.reset()

    def __hash__(self):
        return self.hash

    def __eq__(self, Individual other):
        if self.hash != other.hash and self.grade != other.grade:
            return False
        for i in used_points:
            if self.gene[i] != other.gene[i]:
                return False
        return True

    def __ge__(self, Individual other):
        return self.grade >= other.grade

    def __gt__(self, Individual other):
        return self.grade > other.grade

    def __le__(self, Individual other):
        return self.grade <= other.grade

    def __lt__(self, Individual other):
        return self.grade < other.grade

    def __str__(self):
        a = []
        for ir in range(0, rc, c):
            b = [*map(str, self.gene[ir:ir+c])]
            a.append(b)
        for u in unused_points:
            i, j = u // c, u % c
            a[i][j] = 'a'
        return '\n'.join([f'ans:{self.grade}',*(''.join(k) for k in a)])



# Generate with no input

cpdef Individual fully_new():
    return Individual(rng.integers(10, size = rc))

cpdef Individual read_from_file():
    cdef size_t i
    cdef string use
    cdef list pile, one
    cdef int j
    cdef np.ndarray[DTYPE_t, ndim = 1] two
    if frag_len == 0:
        return fully_new()
    i = rng.integers(frag_len)
    use = frag_path[i]
    with open(use, 'r') as file:
        pile = file.readlines()
        one = [row.rstrip() for row in pile[1:]]
        two = parse_to_array(one) # Note: Here we assume unassigned values are stored in one letter, i.e. 'a'
    return Individual(two)

# Using one
cpdef np.ndarray random_balance(np.ndarray[DTYPE_t, ndim = 1] f):
    cdef np.ndarray[DTYPE_t, ndim = 1] g
    cdef int m
    m = rng.integers(10)
    g = (f + np.arange(m, m + rc, dtype = DTYPE)) % 10
    np.random.shuffle(g)
    return g


cpdef np.ndarray mutate_more(np.ndarray[DTYPE_t, ndim = 1] f):
    cdef np.ndarray[DTYPE_t, ndim = 1] g, h
    cdef size_t num
    cdef int j, i
    g = f.copy()
    num = rng.integers(1, 6)
    h = rng.integers(10, size = num)
    for i, j in enumerate(np.random.choice(use_num, num, replace = False)):
        g[used_points[j]] = h[i]
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

cpdef np.ndarray swap_next(np.ndarray[DTYPE_t, ndim = 1] f):
    cdef np.ndarray[DTYPE_t, ndim = 1] g
    cdef int i, j, k, l, n, m, tmp
    g = f.copy()
    n = rng.integers(2, 6)
    for i in range(n):
        l = rng.integers(use_num)
        j = used_points[l]
        m = rng.integers(len(grid_map[j]))
        k = grid_map[j][m]
        g[j], g[k] = g[k], g[j]
    return g

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

cpdef np.ndarray apply_permutation(np.ndarray[DTYPE_t, ndim = 1] f):
    cdef np.ndarray[DTYPE_t, ndim = 1] perm, g
    perm = np.arange(10, dtype = DTYPE)
    np.random.shuffle(perm)
    g = perm[f]
    return g

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef np.ndarray force_next(np.ndarray[DTYPE_t, ndim = 1] f):
    cdef int current, x, nx, top, j, mx
    cdef size_t i, ii, iii, m, mi, a, s
    cdef np.ndarray[DTYPE_t, ndim = 1] g
    cdef vector[vector[int]] first, second
    cdef vector[int] que, new_que
    cdef int[112] check
    first = [[] for j in range(10)]
    g = f.copy()
    for x in used_points:
        first[f[x]].push_back(x)
    for j in range(1, 10):
        if not first[j].size():
            return rng.integers(10, size = rc)

    for j in range(rc): check[j] = 0
    current = 9
    for top in range(1, bound_by_global_max):
        second = [[] for i in range(10)]
        que = first[top]
        s = 0
        for j in que:
            for nx in grid_map[j]:
                if check[nx] == 0:
                    s += 1
                    second[f[nx]].push_back(nx)
                    check[nx] = 1
        if s <= 10:
            m = 0
            mx = first[top//10][rng.integers(first[top//10].size())]
            new_que = grid_map[mx]
            a = new_que[rng.integers(new_que.size())]
            g[a] = top % 10
            return g
        for i in range(10):
            new_que = second[i]
            if new_que.size() == 0:
                m = 0
                for iii in range(10):
                    mi = second[iii].size()
                    if mi > m:
                        m = mi
                        ii = iii
                a = second[ii][rng.integers(m)]
                g[a] = i
                return g
            for j in new_que:
                check[j] = 0
            if top <= bound_10:
                first.push_back(new_que)
    return f # just in case


table = [(apply_permutation, 0.4), (mutate_more, .8), (push_some_row, 1.), (force_next, .5), (force_next, .5),
         (shuffle_square, 1.5), (swap_next, .5), (random_balance, .5), ]

# List of functions which takes 2 arguments


cpdef np.ndarray split_row(np.ndarray[DTYPE_t, ndim = 1] f, np.ndarray[DTYPE_t, ndim = 1] g):
    cdef np.ndarray[DTYPE_t, ndim = 1] h
    cdef int a, b, i
    h = f.copy()
    a = rng.integers(2)
    b = rng.integers(1, c)
    if a == 0:
        for i in range(b, c):
            h[i::c] = g[i::c]
    else:
        for i in range(b):
            h[i::c] = g[i::c]
    return h

cpdef np.ndarray split_col(np.ndarray[DTYPE_t, ndim = 1] f, np.ndarray[DTYPE_t, ndim = 1] g):
    cdef np.ndarray[DTYPE_t, ndim = 1] h
    cdef int a, b
    h = f.copy()
    a = rng.integers(2)
    b = rng.integers(1, r)*c
    if a == 0:
        h[b:] = g[b:]
    else:
        h[:b] = g[:b]
    return h

cpdef np.ndarray flavor(np.ndarray[DTYPE_t, ndim = 1] f, np.ndarray[DTYPE_t, ndim = 1] g):
    cdef np.ndarray[DTYPE_t, ndim = 1] h
    cdef int[14] x
    cdef int a, b
    h = f.copy()
    b = rng.integers(rc)
    x = rng.integers(r, size = 14)
    for i in range(c):
        a = c*x[i]+i+b
        if a >= rc: a -= rc
        h[a] = g[a]
    return h

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

cpdef np.ndarray pairing(np.ndarray[DTYPE_t, ndim = 1] f, np.ndarray[DTYPE_t, ndim = 1] g):
    cdef np.ndarray[DTYPE_t, ndim = 1] h
    h = f.copy()
    cdef int[56] rand
    cdef int i
    rand = rng.integers(2, size = rc//2)
    for i in range(rc//2):
        if rand[i] == 0:
            h[2 * i + 1] = g[2 * i + 1]
        else:
            h[2 * i] = g[2 * i]
    return h


cpdef np.ndarray add(np.ndarray[DTYPE_t, ndim = 1] f, np.ndarray[DTYPE_t, ndim = 1] g):
    cdef np.ndarray[DTYPE_t, ndim = 1] h
    h = (f + g[rc-1::-1]) % 10
    return h

table2 = [(split_col, 1.), (split_row, 1.), (flavor, 1.), (replace_rect, 1.), (pairing, 1.), (add, 1.)]


cdef generate_table(table):
    cdef list pool = []
    cdef vector[float] cum_weights = []
    cdef float accumulated = 0.
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
        renew_global_max(1000) # No danger while loading
        population = []
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
                    three = parse_to_array(one)
                    two = Individual(three)
                    population.append(two)
            print(f"length = {len(population)}")
            self.use_ = use
        else:
            self.use_ = "save2.txt"

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
            self.population.append(fully_new())

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
        cdef uint it, kk
        cdef Individual x
        cdef double check
        assert save_period > 0 and show > 0 and save_period % show == 0
        check = time()
        for it in range(1, epoch + 1):
            self.progress()

            if it % show == 0:
                print(f"epoch = {it:<5}|max_score = {self.max_:<4}|elapsed = {time() - check:.2f}s|num = {len(self.population)}")
                if it % save_period == 0:
                    self.population.sort(reverse = True)
                    self.cutoff(100) # certainly fail with thres 1000, tragically...
                    if self.max_ > self.prev_max_:
                        print(f"previous max = {self.prev_max_}, now max = {self.max_}. Saving new values...")
                        for kk, x in tqdm(enumerate(self.population)):
                            if x.grade <= self.prev_max_:
                                break
                            with open(foldername+bytes(f'/{it}_{kk}.txt'.encode('UTF-8')), 'w') as file:
                                file.writelines(str(x))
                                file.writelines('\n')
                        self.prev_max_ = self.max_
                        print("save complete.")
                    self.renew()
                    np.random.seed(int(check*814) % (1 << 32))
                check = time()
        # With all process done:
        with open(self.use_, 'w') as file:
            for x in tqdm(self.population):
                if x.grade < 100:
                    continue
                file.writelines(str(x))
                file.writelines('\n')