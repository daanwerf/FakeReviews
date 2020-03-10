def mine_part_of_speech_pattern(D, T, minsup, minadherence, max_length):
    C = [[]] * len(T)
    for tag in D:
        C[1][tag] += 1

    F = []
    n = len(pos_sequence)
    for f in C.keys():
        if (C[f] / n) > minsup:
            F.append(f)

    Spl = F
    for k in range(2, max_length+1):
        C = candidate_gen(F, )


def candidate_gen(F, tagset):
    C = []

    for c in F:
        for t in tagset:
            c = add_suffix(c, t)
            C.append(c)

    return C


def add_suffix(c, t):
    return str(c) + "_" + str(t)
