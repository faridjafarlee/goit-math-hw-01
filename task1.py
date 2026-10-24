import numpy as np

M = np.array([
    [100, 150, 200],
    [ 50, 100, 150],
    [  0,  50, 100],
], dtype=float)

E = np.array([
    [20, 30, 40],
    [10, 20, 30],
    [ 5, 10, 15],
], dtype=float)

# 1) Contrast: 0.5 * M
contrast = 0.5 * M

# 2) Brightness: M + 25
brightness = M + 25

# 3) Blending: 0.8 * M + 0.2 * E
blending = 0.8 * M + 0.2 * E

print("M:\n", M)
print("\nE:\n", E)

print("\n1) Contrast (0.5*M):\n", contrast)
print("\n2) Brightness (M+25):\n", brightness)
print("\n3) Blending (0.8*M + 0.2*E):\n", blending)

# Перевірка з аналітикою
expected_contrast = np.array([[50, 75, 100],
                              [25, 50, 75],
                              [ 0, 25, 50]], dtype=float)

expected_brightness = np.array([[125, 175, 225],
                                [ 75, 125, 175],
                                [ 25,  75, 125]], dtype=float)

expected_blending = np.array([[84, 126, 168],
                              [42,  84, 126],
                              [ 1,  42,  83]], dtype=float)

assert np.allclose(contrast, expected_contrast)
assert np.allclose(brightness, expected_brightness)
assert np.allclose(blending, expected_blending)