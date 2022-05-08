import random

matrix_size = 3 * 700
DIST_LIMIT = 10000
fname = "case_5.txt"

f = open(fname, "w")

f.write(str(matrix_size) + "\n")
for y in range(0, matrix_size):
  to_write = ''
  for x in range(0, matrix_size):
    if random.randint(0, 10) > 7:
      to_write += str(random.randint(0, matrix_size)) + " "
    else:
      to_write += str(DIST_LIMIT + 1) + " "
  to_write = to_write.rstrip()
  f.write(to_write + '\n')

f.close()