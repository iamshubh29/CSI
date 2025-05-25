def lower_triangular(n):
    print("Lower Triangular Pattern:")
    for i in range(1, n + 1):
        print("* " * i)
    print()  


def upper_triangular(n):
    print("Upper Triangular Pattern:")
    for i in range(n, 0, -1):
        print("* " * i)
    print()  


def pyramid(n):
    print("Pyramid Pattern:")
    for i in range(n):
        print(" " * (n - i - 1) + "* " * (i + 1))
    print()  
rows = 5
lower_triangular(rows)
upper_triangular(rows)
pyramid(rows)
