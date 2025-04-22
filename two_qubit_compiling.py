import numpy as np 
import sympy as sp
import itertools

# This is a function for finding the largest power of 2 that divides a.
# It will be used in the following functions.
def trailing_zeros(a):
    count = 0
    while a:
        if (a&1):
            break
        a >>= 1
        count += 1
    return count

# We work exclusively with numbers of the form (p + q*sqrt(2))/sqrt(2)^r.
# This function puts those numbers in their reduced form.
def reduce_z(a):
    if a[0] == 0 and a[1] == 0:
        return (0,0,0)
    elif a[0] == 0:
        k = min([trailing_zeros(a[1]),a[2]>>1])
    elif a[1] == 0:
        k = min([trailing_zeros(a[0]),a[2]>>1])
    else:
        k = min([trailing_zeros(a[0]),trailing_zeros(a[1]),a[2]>>1])
    a[0] >>= k
    a[1] >>= k
    k <<= 1
    a[2] -= k
    if bin(a[0])[-1] == '0' and a[2]>0:
        a[0],a[1] = [a[1],a[0]]
        a[1] >>= 1
        a[2] -= 1
    return tuple(a)

# This function adds two numbers of the form (p + q*sqrt(2))/sqrt(2)^r
# in tuple form (p,q,r) and reduces the sum.
def quick_sum(a,b):
    a_temp = [a[i] for i in range(3)]
    b_temp = [b[i] for i in range(3)]
    temp = [0,0,0]
    if a_temp[2] < b_temp[2]:
        a_temp,b_temp = [b_temp,a_temp]
    k = a_temp[2]-b_temp[2]
    temp[2] = a_temp[2]
    b_temp[2] = a_temp[2]
    if bin(k)[-1] == '0':
        k >>= 1
        b_temp[0] <<= k
        b_temp[1] <<= k
    else:
        k >>= 1
        b_temp[0] <<= k
        b_temp[1] <<= k
        b_temp[0],b_temp[1] = [b_temp[1],b_temp[0]]
        b_temp[0] <<= 1
    temp[0] = a_temp[0] + b_temp[0]
    temp[1] = a_temp[1] + b_temp[1]
    return reduce_z(temp)

# This function subtracts two numbers of the form (p + q*sqrt(2))/sqrt(2)^r
# in tuple form (p,q,r) and reduces the difference.
def quick_diff(a,b):
    a_temp = [a[i] for i in range(3)]
    b_temp = [b[i] for i in range(3)]
    temp = [0,0,0]
    sign = 1
    if a_temp[2] < b_temp[2]:
        sign = -1
        a_temp,b_temp = [b_temp,a_temp]
    k = a_temp[2]-b_temp[2]
    temp[2] = a_temp[2]
    b_temp[2] = a_temp[2]
    if bin(k)[-1] == '0':
        k >>= 1
        b_temp[0] <<= k
        b_temp[1] <<= k
    else:
        k >>= 1
        b_temp[0] <<= k
        b_temp[1] <<= k
        b_temp[0],b_temp[1] = [b_temp[1],b_temp[0]]
        b_temp[0] <<= 1
    temp[0] = sign*(a_temp[0] - b_temp[0])
    temp[1] = sign*(a_temp[1] - b_temp[1])
    return reduce_z(temp)

# Function to multiply by 1/sqrt(2)
def divide_by_root_2(a):
    return tuple([a[0],a[1],a[2]+1])

#Function to left multiply by M_p,q
def pq_left_prod(A,p,q):
    if p != q:
        A = A.copy()
        v = [quick_diff(divide_by_root_2(A[p-1,i]),divide_by_root_2(A[q-1,i])) for i in range(6)]
        w = [quick_sum(divide_by_root_2(A[p-1,i]),divide_by_root_2(A[q-1,i])) for i in range(6)]
        A[p-1] = v
        A[q-1] = w
    return A

#Function to right multiply by M_p,q
def pq_right_prod(A,p,q):
    A = A.copy()
    v = [quick_sum(divide_by_root_2(A[i,p-1]),divide_by_root_2(A[i,q-1])) for i in range(6)]
    w = [quick_diff(divide_by_root_2(A[i,q-1]),divide_by_root_2(A[i,p-1])) for i in range(6)]
    A[:,p-1] = v
    A[:,q-1] = w
    return A

# This function multiplies two numbers of the form (p + q*sqrt(2))/sqrt(2)^r
# in tuple form (p,q,r) and reduces the product.
def quick_product(a,b):
    temp = [0,0,0]
    temp[2] = a[2]+b[2]
    temp[0] = int(a[0])*int(b[0])+2*int(a[1])*int(b[1])
    temp[1] = int(a[0])*int(b[1])+int(a[1])*int(b[0])
    return reduce_z(temp)

# Initializes a zero matrix.
def zero_matrix():
    A = np.empty((6,6),dtype=object)
    for i in range(6):
        for j in range(6):
            A[i,j] = (0,0,0)
    return A

# Converts a tuple into something more readable. Used below.
def convert(a):
    return (a[0] + a[1]*sp.sqrt(2))/sp.sqrt(2)**a[2]

# Converts a matrix of tuples into something more readable.
def convert_matrix(A):
    B = sp.zeros(6,6)
    for i in range(6):
        for j in range(6):
            B[i,j] = convert(A[i,j])
    return B

# Initializes the matrices M(p,q) from the paper.
def m(p,q):
    temp = zero_matrix()
    for i in range(6):
        temp[i,i] = (1,0,0)
    if p == q:
        return temp
    temp[p-1,p-1] = (1,0,1)
    temp[q-1,q-1] = (1,0,1)
    temp[p-1,q-1] = (-1,0,1)
    temp[q-1,p-1] = (1,0,1)
    return temp

# Returns a list of indices that would sort the sequence seq.
def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__,reverse=True)

# We introduce a lexicographic ordering on the numbers above with 
# (p1+q1*sqrt(2))/sqrt(2)^r1 > (p2+q2*sqrt(2))/sqrt(2)^r2 if and only if
# r1 > r2 or (r1 = r2 and |p1| > |p2|) or (r1 = r2 and |p1| = |p2| and sgn(p1)*q1 > sgn(p2)*q2).
# This function returns a tuple for this ordering.
def tuple_configure(a):
    return (a[2],np.abs(a[0]),np.sign(a[0])*a[1])

# Returns a sorted list of entries of a vector sorted by the above order from largest to smallest
# with tuples transformed as above.
def sorted_vector(v,reverse=True):
    return tuple(sorted([tuple_configure(a) for a in v],reverse=reverse))

# Returns a of entries of a vector with tuples transformed as above.
def vector_abs(v):
    return tuple([tuple_configure(a) for a in v])

# Returns v or -v such that the first entry has a positive integer part
def vector_sign(v):
    temp = [np.sign(v[i][0]) for i in range(6) if v[i][0] != 0]
    sign = temp[0]
    if sign == 1:
        return v,sign
    else:
        return [(sign*v[i][0],sign*v[i][1],v[i][2]) for i in range(6)],sign

def frequency(A):
    return [tuple([len(set(sorted_vector(A[i])))]+list(sorted_vector(A[i]))) for i in range(6)]

def equivalent_sets(v):
    v = [v[i] for i in argsort(v)]
    rows = [[0]]
    for i in range(len(v)-1):
        if v[i+1] == v[i]:
            rows[-1].append(i+1)
        else:
            rows.append([i+1])
    return rows

def sign_change(A,rows,columns):
    A = A.copy()
    for i in rows:
        A[i] = [(-A[i,j][0],-A[i,j][1],A[i,j][2]) for j in range(6)]
    for j in columns:
        A[:,j] = [(-A[i,j][0],-A[i,j][1],A[i,j][2]) for i in range(6)]
    return A

def pivot_finder(A):
    pivot_col = []
    fixed_col = []
    row = 0
    while row < 6:
        fnz = next((i for i, x in enumerate(A[row]) if x != (0,0,0)), None)
        if fnz not in fixed_col:
            pivot_col.append(fnz)
        for i in range(fnz,6):
            if A[row,i] != (0,0,0) and (i not in fixed_col):
                fixed_col.append(i)
        row += 1
    return pivot_col

def fixed_sign_change(A):
    A = A.copy()
    fixed_col = pivot_finder(A)
    row = 0
    while row < 6:
        fnz = next((i for i, x in enumerate(A[row]) if x != (0,0,0)), None)
        if A[row,fnz][0] < 0:
            A = sign_change(A,[row],[])
        for i in range(fnz,6):
            if A[row,i][0] < 0 and (i not in fixed_col):
                A = sign_change(A,[],[i])
                fixed_col.append(i)
            elif A[row,i][0] > 0 and (i not in fixed_col):
                fixed_col.append(i)
        row += 1
    return A

def best_sign_change(A):
    temp_A = A.copy()
    pivots = pivot_finder(A)
    for r in range(len(pivots)+1):
        for q in range(len(pivots)+1):
            for rows in itertools.combinations(pivots,r):
                for columns in itertools.combinations(pivots,q):
                    if tuple(fixed_sign_change(sign_change(A,rows,columns)).reshape(-1,)) > tuple(temp_A.reshape(-1,)):
                        temp_A = fixed_sign_change(sign_change(A,rows,columns))
    return temp_A

# Returns a canonical representation of an SO(6) matrix
def canonical_form(A):
    A = A.copy()
    x = frequency(A)
    rows = equivalent_sets(x)
    A = A[argsort(x)]
    y = [vector_abs(A[:,i]) for i in range(6)]
    A = A[:,argsort(y)]
    check_A = best_sign_change(A)
    if tuple(check_A.reshape(-1,)) == tuple(m(1,1).reshape(-1)):
        return m(1,1)
    if tuple(check_A.reshape(-1,)) == tuple(sign_change(m(1,2),[],[1]).reshape(-1)):
        return m(1,2)
    row_perms = []
    for i in rows:
        row_perms.append(list(itertools.permutations(i)))
    for i in itertools.product(*row_perms):
        temp_rows = [x for xs in i for x in xs]
        temp_A = A[temp_rows,:].copy()
        y = [vector_abs(temp_A[:,i]) for i in range(6)]
        col_index = argsort(y)
        temp_A = temp_A[:,col_index]
        cols = equivalent_sets(y)
        col_perms = []
        for j in cols:
            col_perms.append(list(itertools.permutations(j)))
        for j in itertools.product(*col_perms):
            temp_cols = [x for xs in j for x in xs]
            temp_A = temp_A[:,temp_cols].copy()
            temp_A = best_sign_change(temp_A)
            if tuple(temp_A.reshape(-1,)) > tuple(check_A.reshape(-1,)):
                check_A = temp_A
    return check_A

# Converts a matrix to a string to be stored in a dictionary.
def mat_to_str(A):
    return str(list(A.reshape(-1,)))[1:-1]

# Converts a string back to a matrix for manipulation.
def str_to_mat(s):
    temp = np.empty(36,dtype=object)
    temp[:] = eval(s)
    return temp.reshape(6,6)

# Creates a random list of k T-gates and outputs the sequence and corresponding matrix
def random_operation(c):
    temp = m(1,1)
    seq = []
    for i in range(c):
        p,q = np.random.choice([1,2,3,4,5,6],2,replace=False)
        temp = pq_left_prod(temp,p,q)
        seq = [(p,q)] + seq
    return temp, seq

# Performs an single step in the compilation of 2-qubit Clifford-T operations
def comp_step(A):
    pattern = np.zeros((6,6),dtype='int')
    parity = np.zeros((6,6),dtype='int')
    L = np.zeros((6,6),dtype='int')
    lde = max([A[i,j][2] for i in range(6) for j in range(6)])
    for i in range(6):
        for j in range(6):
            pattern[i,j] = -int(A[i,j][2] == lde)
            parity[i,j] = A[i,j][1]%2 * pattern[i,j]
            L[i,j] = A[i,j][2]
    row_index = np.argsort(np.sum(pattern,axis=1),kind='mergesort')
    pattern = pattern[row_index]
    parity = parity[row_index]
    L = L[row_index]
    col_index = np.argsort(pattern[0],kind='mergesort')
    pattern = pattern[:,col_index]
    parity = parity[:,col_index]
    L = L[:,col_index]
    row_index_1 = argsort([tuple(-pattern[i]) for i in range(6)])
    pattern = pattern[row_index_1]
    parity = parity[row_index_1]
    L = L[row_index_1]
    row_index = row_index[row_index_1]
    col_index_1 = argsort([tuple(-pattern[:,i]) for i in range(6)])
    pattern = pattern[:,col_index_1]
    parity = parity[:,col_index_1]
    L = L[:,col_index_1]
    col_index = col_index[col_index_1]
    pattern = -pattern
    
    if tuple(np.sum(pattern,axis=0)) == (2,2,0,0,0,0) and tuple(np.sum(pattern,axis=1)) == (2,2,0,0,0,0):
        p = col_index[0] + 1
        q = col_index[1] + 1
        return pq_right_prod(A,p,q),'R',[(p,q)]
        
    elif tuple(np.sum(pattern,axis=0)) == (2,2,2,2,0,0) and tuple(np.sum(pattern,axis=1)) == (4,4,0,0,0,0):
        p = row_index[0] + 1
        q = row_index[1] + 1
        return pq_left_prod(A,p,q),'L',[(p,q)]
        
    elif tuple(np.sum(pattern,axis=1)) == (2,2,2,2,0,0) and tuple(np.sum(pattern,axis=0)) == (4,4,0,0,0,0):
        p = col_index[0] + 1
        q = col_index[1] + 1
        return pq_right_prod(A,p,q),'R',[(p,q)]
        
    elif tuple(np.sum(pattern,axis=0)) == (4,4,4,4,0,0) and tuple(np.sum(pattern,axis=1)) == (4,4,4,4,0,0):
        row_index1 = np.argsort(np.sum(parity[:4,:4],axis=1),kind='mergesort')
        parity[:4] = parity[row_index1]
        L[:4] = L[row_index1]
        row_index = row_index[list(row_index1)+[4,5]]
        col_index1 = np.argsort(parity[0,:4],kind='mergesort')
        parity[:,:4] = parity[:,col_index1]
        L[:,:4] = L[:,col_index1]
        col_index = col_index[list(col_index1)+[4,5]]
        row_index_1 = argsort([tuple(-parity[i,:4]) for i in range(4)])
        parity[:4] = parity[row_index_1]
        L[:4] = L[row_index_1]
        row_index = row_index[row_index_1+[4,5]]
        col_index_1 = argsort([tuple(-parity[:4,i]) for i in range(4)])
        parity[:,:4] = parity[:,col_index_1]
        L[:,:4] = L[:,col_index_1]
        col_index = col_index[col_index_1+[4,5]]
        
        if np.sum(parity[:2,:2])%4 == 0 and np.sum(parity[2:4,:2])%4 == 0:
            p1 = col_index[0] + 1
            q1 = col_index[1] + 1
            p2 = col_index[2] + 1
            q2 = col_index[3] + 1
            return pq_right_prod(pq_right_prod(A,p1,q1),p2,q2),'R',[(p1,q1),(p2,q2)]
            
        elif np.sum(parity[:2,:2])%4 == 0 and np.sum(parity[2:4,:2])%4 == 2 and parity[2,2] == parity[3,2]:
            p1 = row_index[0] + 1
            q1 = row_index[1] + 1
            p2 = row_index[2] + 1
            q2 = row_index[3] + 1
            return pq_left_prod(pq_left_prod(A,p1,q1),p2,q2),'L',[(p2,q2),(p1,q1)]
            
        elif np.sum(parity[:2,:2])%4 == 0 and np.sum(parity[2:4,:2])%4 == 2 and parity[2,2] != parity[3,2]:
            print('3c')
            ij = argsort(L[4,:4])
            p = col_index[ij[0]] + 1
            q = col_index[ij[1]] + 1
            return pq_right_prod(A,p,q),'R',[(p,q)]
            
        else:
            if (L[4,0] == lde-1 and lde-1 == L[4,1]) or (L[4,0] < lde-1 and lde-1 > L[4,1]):
                p = col_index[0] + 1
                q = col_index[1] + 1
                return pq_right_prod(A,p,q),'R',[(p,q)]
                
            elif (L[4,0] == lde-1 and lde-1 == L[4,2]) or (L[4,0] < lde-1 and lde-1 > L[4,2]):
                print('3e')
                p = col_index[0] + 1
                q = col_index[2] + 1
                return pq_right_prod(A,p,q),'R',[(p,q)]
                
            elif (L[4,1] == lde-1 and lde-1 == L[4,3]) or (L[4,1] < lde-1 and lde-1 > L[4,3]):
                print('3f')
                p = col_index[1] + 1
                q = col_index[3] + 1
                return pq_right_prod(A,p,q),'R',[(p,q)]
                
            elif (L[0,4] == lde-1 and lde-1 == L[1,4]) or (L[0,4] < lde-1 and lde-1 > L[1,4]):
                p = row_index[0] + 1
                q = row_index[1] + 1
                return pq_left_prod(A,p,q),'L',[(p,q)]
                
            elif (L[0,4] == lde-1 and lde-1 == L[2,4]) or (L[0,4] < lde-1 and lde-1 > L[2,4]):
                print('3h')
                p = row_index[0] + 1
                q = row_index[2] + 1
                return pq_left_prod(A,p,q),'L',[(p,q)]
                
            elif (L[1,4] == lde-1 and lde-1 == L[3,4]) or (L[1,4] < lde-1 and lde-1 > L[3,4]):
                print('3i')
                p = row_index[1] + 1
                q = row_index[3] + 1
                return pq_left_prod(A,p,q),'L',[(p,q)]
                
            elif L[5,5] < lde-1:
                print('3j')
                p1 = col_index[0] + 1
                q1 = col_index[1] + 1
                p2 = col_index[2] + 1
                q2 = col_index[3] + 1
                return pq_right_prod(pq_right_prod(A,p1,q1),p2,q2),'R',[(p1,q1),(p2,q2)]
                
            elif L[5,5] == lde-1:
                if L[0,5] == lde-1:
                    p1 = col_index[0] + 1
                    q1 = col_index[2]+ 1
                    p2 = col_index[1] + 1
                    q2 = col_index[3] + 1
                if L[0,5] < lde-1:
                    p1 = col_index[0] + 1
                    q1 = col_index[1]+ 1
                    p2 = col_index[2] + 1
                    q2 = col_index[3] + 1
                return pq_right_prod(pq_right_prod(A,p1,q1),p2,q2),'R',[(p1,q1),(p2,q2)]
            
    elif tuple(np.sum(pattern,axis=0)) == (4,4,2,2,0,0) and tuple(np.sum(pattern,axis=1)) == (4,4,2,2,0,0):
        p = col_index[0] + 1
        q = col_index[1] + 1
        return pq_right_prod(A,p,q),'R',[(p,q)]
        
    elif tuple(np.sum(pattern,axis=0)) == (2,2,2,2,0,0) and tuple(np.sum(pattern,axis=1)) == (2,2,2,2,0,0):
        p = col_index[0] + 1
        q = col_index[1] + 1
        return pq_right_prod(A,p,q),'R',[(p,q)]
        
    elif tuple(np.sum(pattern,axis=0)) == (4,4,2,2,2,2) and tuple(np.sum(pattern,axis=1)) == (4,4,4,4,0,0):
        p = row_index[0] + 1
        q = row_index[1] + 1
        return pq_left_prod(A,p,q),'L',[(p,q)]
        
    elif tuple(np.sum(pattern,axis=1)) == (4,4,2,2,2,2) and tuple(np.sum(pattern,axis=0)) == (4,4,4,4,0,0):
        p = col_index[0] + 1
        q = col_index[1] + 1
        return pq_right_prod(A,p,q),'R',[(p,q)]
        
    elif tuple(np.sum(pattern,axis=0)) == (2,2,2,2,2,2) and tuple(np.sum(pattern,axis=1)) == (2,2,2,2,2,2):
        p = col_index[0] + 1
        q = col_index[1] + 1
        return pq_right_prod(A,p,q),'R',[(p,q)]
        
    elif tuple(np.sum(pattern,axis=0)) == (4,4,4,4,4,4) and tuple(np.sum(pattern,axis=1)) == (4,4,4,4,4,4):
        if parity[0,0] == parity[0,1]:
            p1 = col_index[0] + 1
            q1 = col_index[1] + 1
            p2 = col_index[2] + 1
            q2 = col_index[3] + 1
            p3 = col_index[4] + 1
            q3 = col_index[5] + 1
            return pq_right_prod(pq_right_prod(pq_right_prod(A,p1,q1),p2,q2),p3,q3),'R',[(p1,q1),(p2,q2),(p3,q3)]
            
        elif parity[0,0] == parity[1,0]:
            p1 = row_index[0] + 1
            q1 = row_index[1] + 1
            p2 = row_index[2] + 1
            q2 = row_index[3] + 1
            p3 = row_index[4] + 1
            q3 = row_index[5] + 1
            return pq_left_prod(pq_left_prod(pq_left_prod(A,p1,q1),p2,q2),p3,q3),'L',[(p3,q3),(p2,q2),(p1,q1)]
            
        else:
            parity11 = np.argmin(parity[0,:2])
            parity12 = np.argmax(parity[0,:2])
            parity21 = np.argmin(parity[0,2:4])+2
            parity22 = np.argmax(parity[0,2:4])+2
            p1 = col_index[parity11] + 1
            q1 = col_index[parity21] + 1
            p2 = col_index[parity12] + 1
            q2 = col_index[parity22] + 1
            A = pq_right_prod(pq_right_prod(A,p1,q1),p2,q2)
            right = [(p1,q1),(p2,q2)]
            parity = np.zeros((6,6),dtype='int')
            lde = max([A[i,j][2] for i in range(6) for j in range(6)])
            times = (2**(lde>>1)*((lde+1)%2),2**(lde>>1)*(lde%2),0)
            for i in range(6):
                for j in range(6):
                    parity[i,j] = (np.array(quick_product(times,A[i,j]))%2)[1]*-(np.array(quick_product(times,A[i,j]))%2)[0]
            row_index = np.argsort(np.sum(parity,axis=1),kind='mergesort')
            parity = parity[row_index]
            col_index = np.argsort(parity[0],kind='mergesort')
            parity = parity[:,col_index]
            row_index_1 = argsort([tuple(-parity[i]) for i in range(6)])
            parity = parity[row_index_1]
            row_index = row_index[row_index_1]
            col_index_1 = argsort([tuple(-parity[:,i]) for i in range(6)])
            parity = parity[:,col_index_1]
            col_index = col_index[col_index_1]
            p1 = row_index[0] + 1
            q1 = row_index[1] + 1
            p2 = row_index[2] + 1
            q2 = row_index[3] + 1
            return pq_left_prod(pq_left_prod(A,p2,q2),p1,q1),'LR',[(p2,q2),(p1,q1)]+right

# Decomposes a Clifford-T operation
def compile(A):
    A = A.copy()
    lde = max([A[i,j][2] for i in range(6) for j in range(6)])
    left_steps = []
    right_steps = []
    steps = []
    count = 1
    matrix = [A]
    while lde > 0:
        A,side,M = comp_step(A)
        if side == 'R':
            right_steps = right_steps + M
        elif side == 'L':
            left_steps = M + left_steps
        else:
            left_steps = M[:2] + left_steps
            right_steps = right_steps + M[2:]
        lde = max([A[i,j][2] for i in range(6) for j in range(6)])
    left_steps = [(p[1],p[0]) for p in (left_steps)][::-1]
    right_steps = [(p[1],p[0]) for p in right_steps[::-1]]
    perm = np.array(list((np.array([1,2,3,4,5,6],dtype='int').reshape(1,-1))@(convert_matrix(A)))).reshape(-1,)
    new_right = []
    for i in right_steps:
        t = tuple(perm[np.array(i)-1])
        if t[0]*t[1] > 0:
            new_right.append((np.abs(t[0]),np.abs(t[1])))
        else:
            new_right.append((np.abs(t[1]),np.abs(t[0])))
    return left_steps+new_right,A
    return [(p[1],p[0]) for p in (left_steps)],[(p[1],p[0]) for p in right_steps[::-1]], A


# Finds the most efficient compilation of an SO(6) when conjugated by Clifford operations
def preprocessing(A):
    min_i = [0,1,2,3,4,5]
    min_j = [0,1,2,3,4,5]
    min_rows = 5000000
    for i in itertools.permutations(range(6)):
        z = compile(A[:,list(i)])
        t = len(z[0])
        if t < min_rows:
            min_rows = t
            min_i = list(i)
    for i in itertools.permutations(range(6)):
        z = compile(A[:,min_i][list(i),:])
        t = len(z[0])
        if t < min_rows:
            min_rows = t
            min_j = list(i)
    return compile(A[:,min_i][min_j,:])

# Uses a look-up table of T-count k-1 to reduce compiled list of T-operations
def postprocessing(T_list,T_table,k):
    index = 0
    updated_list = T_list.copy()
    clifford = convert_matrix(m(1,1))
    while index+k <= len(updated_list):
        test = m(1,1)
        for i in range(index,index+k):
            test = pq_right_prod(test,updated_list[i][0],updated_list[i][1])
        if mat_to_str(canonical_form(test)) in T_table:
            test,test_L,test_R = canonical_form(test,True)
            shorter,shorter_L,shorter_R = canonical_form(str_to_mat(T_table[mat_to_str(test)][0]),True)
            L = test_L.T@shorter_L
            R = shorter_R@test_R.T
            perm = np.array(list((np.array([1,2,3,4,5,6],dtype='int').reshape(1,-1))@L),dtype='int').reshape(-1,)
            shorter_list = []
            for i in T_table[mat_to_str(test)][1][::-1]:
                t = tuple(perm[np.array(i,dtype='int')-1])
                if t[0]*t[1] > 0:
                    shorter_list.append((np.abs(t[0]),np.abs(t[1])))
                else:
                    shorter_list.append((np.abs(t[1]),np.abs(t[0])))
            perm = np.array(list((np.array([1,2,3,4,5,6],dtype='int').reshape(1,-1))@L@R)).reshape(-1,)
            updated  = []
            for i in updated_list[(index+k):]:
                t = tuple(perm[np.array(i,dtype='int')-1])
                if t[0]*t[1] > 0:
                    updated.append((np.abs(t[0]),np.abs(t[1])))
                else:
                    updated.append((np.abs(t[1]),np.abs(t[0])))
            updated_list = updated_list[:index] + shorter_list + updated
            index = 0
            clifford = L@R@clifford
        else:
            index += 1
    return updated_list, clifford

# Function to find all SO(6) matrices of T-length one more than current_list and two more than check_list
def look_up_table(current_list,check_list):
    new_list = dict()
    for t in current_list:
        for p in range(1,7):
            for q in range(p+1,7):
                temp = pq_left_prod(str_to_mat(current_list[t][0]),p,q)
                keep = mat_to_str(temp)
                temp = canonical_form(temp)
                temp = mat_to_str(temp)
                if temp not in new_list and temp not in check_list:
                    new_list[temp] = [keep,current_list[t][1].copy()+[(p,q)]]
    return new_list

# Function for assembling a look up table for T-length up to k
# Options: Provide pre-compiled look up table for T-length up to (and including) j
def full_look_up_table(k,table=False,j=1):
    if type(table) == bool:
        check_list = {mat_to_str(m(1,1)):[mat_to_str(m(1,1)),[]]}
        current_list = {mat_to_str(canonical_form(m(1,2))):[mat_to_str(m(1,2)),[(1,2)]]}
        table = {**current_list,**check_list}
    else:
        check_list = table
        current_list = table
    for i in range(j,k):
        new_list = look_up_table(current_list,check_list)
        table = {**new_list,**table}
        check_list = current_list.copy()
        current_list = new_list.copy()
    return table
        
    

