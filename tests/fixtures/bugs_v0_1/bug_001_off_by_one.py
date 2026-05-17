def sum_first_n_squares(n):
    total=0
    for i in range(1,n):
        total+=i*i
    return total
def repro():
    return sum_first_n_squares(5)==55
if __name__=='__main__':
    import sys
    ok=repro()
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)
