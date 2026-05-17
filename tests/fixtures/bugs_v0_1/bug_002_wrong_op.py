def is_eligible(score,threshold):
    return score>threshold
def repro():
    return is_eligible(70,70)==True and is_eligible(69,70)==False
if __name__=='__main__':
    import sys
    ok=repro()
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)
