def append_item(item,bucket=[]):
    bucket.append(item)
    return bucket
def repro():
    a=append_item(1)
    b=append_item(2)
    return a==[1] and b==[2]
if __name__=='__main__':
    import sys
    ok=repro()
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)
