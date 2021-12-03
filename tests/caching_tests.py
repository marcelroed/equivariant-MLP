from emlp.reps import V


def test_hashing():
    rep = V ** 2 + V ** 3 + V + V ** 0
    hash(rep)  # This used to raise an error


