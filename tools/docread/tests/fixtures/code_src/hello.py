import os
import sys

GREETING = "hello"
LIMIT = 10


def greet(name, times=1):
    """Say hello."""
    return (GREETING + " " + name + "\n") * times


@staticmethod
def helper(values):
    total = 0
    for v in values:
        total += v

    return total


class Garden:
    def __init__(self, rows):
        self.rows = rows

    def count(self):
        return len(self.rows)


async def main():
    g = Garden(["tulips", "beans"])
    print(greet("world", g.count()))


if __name__ == "__main__":
    main()
