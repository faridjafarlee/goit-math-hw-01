import numpy as np

u = np.array([8, 2, 5], dtype=float)
vA = np.array([9, 1, 2], dtype=float)
vB = np.array([1, 9, 8], dtype=float)
vC = np.array([7, 2, 6], dtype=float)

def cosine_similarity_verbose(a, b, name=""):
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    denom = norm_a * norm_b
    cos = dot / denom

    print("==== {} ====".format(name))
    print("a =", a)
    print("b =", b)
    print("dot(a,b) = {:.6f}".format(dot))
    print("||a|| = {:.6f}".format(norm_a))
    print("||b|| = {:.6f}".format(norm_b))
    print("||a||*||b|| = {:.6f}".format(denom))
    print("cosine = dot / (||a||*||b||) = {:.6f}".format(cos))
    print()

    return cos

scores = {
    "A (Action Movie)": cosine_similarity_verbose(u, vA, "A (Action Movie)"),
    "B (Comedy Movie)": cosine_similarity_verbose(u, vB, "B (Comedy Movie)"),
    "C (Drama Movie)": cosine_similarity_verbose(u, vC, "C (Drama Movie)"),
}

print("---- Summary ----")
for name, score in scores.items():
    print("{}: {:.6f}".format(name, score))

best = max(scores, key=scores.get)
print("\nBest match:", best)
