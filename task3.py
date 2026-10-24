import numpy as np

A = np.array([[4, 4, 6],
              [1, 1, 2],
              [1, 2, 4]], dtype=float)

b = np.array([460, 130, 240], dtype=float)

detA = float(np.linalg.det(A))
print("det(A) =", detA)

x = np.linalg.solve(A, b)
print("x =", x)  # [x1, x2, x3]

check = A @ x
print("A @ x =", check)
print("b     =", b)
print("Match =", np.allclose(check, b))
