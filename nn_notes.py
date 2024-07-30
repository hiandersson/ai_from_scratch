
### On dot product (c where each element is calculated as the dot product of the corresponding row of a and column of b):

# Note A: The number of columns in A must match rows in B

# Note B: The final size of matrix C is rows of A x columns of B

# SMALL

# a = [[1, 2], 
#      [3, 4]]
# b = [[5, 6], 
#      [7, 8]]

# resulting c is then

# c[0][0] = 1 * 5 + 2 * 7 = 19
# c[0][1] = 1 * 6 + 2 * 8 = 22

# c[1][0] = 3 * 5 + 4 * 7 = 43
# c[1][1] = 3 * 6 + 4 * 8 = 50

# MEDIUM

# a = [[1, 2, 3], 
#      [4, 5, 6]]
# b = [[7, 8], 
#      [9, 10],
#      [11, 12],

# c[0][0] = 1 * 7 + 2 * 9 + 3 * 11
# c[1][0] = 1 * 8 + 2 * 10 + 3 * 12
# c[1][0] = 4 * 7 + 5 * 9 + 6 * 11
# c[1][1] = 4 * 8 + 5 * 10 + 6 * 12

# LARGE

# a = [[1, 2, 3], 
#      [4, 5, 6],
#      [7, 8, 9]]
# b = [[10, 11], 
#      [12, 13],
#      [14, 15],

# c[0][0] = 1 * 10 + 2 * 12 + 3 * 14
# c[1][0] = 1 * 11 + 2 * 13 + 3 * 15

# c[0][1] = 4 * 10 + 5 * 12 + 6 * 14
# c[1][1] = 4 * 11 + 5 * 13 + 6 * 15

# c[0][2] = 7 * 10 + 8 * 12 + 9 * 14
# c[1][2] = 7 * 11 + 8 * 13 + 9 * 15