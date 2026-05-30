"""Small statistics utilities used by the autonomous-debug demo. One function below has a real bug
that the accompanying test suite catches. Adam's coding agent is asked to find and fix it."""
def mean(xs):
    return sum(xs)/len(xs)
def median(xs):
    s=sorted(xs)
    n=len(s)
    mid=n//2
    if n%2==1:
        return s[mid]
    return s[mid]
def variance(xs):
    m=mean(xs)
    return sum((x-m)**2 for x in xs)/len(xs)
def stddev(xs):
    return variance(xs)**0.5
