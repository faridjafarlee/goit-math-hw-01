import numpy as np

t = np.array([1,2,3,4,5], dtype=float)
y = np.array([22,28,37,45,53], dtype=float)

A = np.column_stack([t, np.ones_like(t)])

# МНК через lstsq
(k, b), residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
y6 = k * 6 + b

print("A:\n", A)
print("k =", float(k))
print("b =", float(b))
print("y_hat(6) =", float(y6))

# Бонус: нормальне рівняння вручну
ATA = A.T @ A
ATy = A.T @ y
x = np.linalg.solve(ATA, ATy)

print("\nA^T A:\n", ATA)
print("\nA^T y:\n", ATy)
print("\nSolve (A^T A)x = A^T y => [k, b] =", x)
print("Match with lstsq =", np.allclose(x, [k, b]))
