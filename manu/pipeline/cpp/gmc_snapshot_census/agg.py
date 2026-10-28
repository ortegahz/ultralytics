import re, sys, collections
path = sys.argv[1]
rows = []
for line in open(path):
    m = re.match(r"\[TRACE\] abs=(-?\d+) reads=(\d+) unique=(\d+) depths=(.*)$", line.strip())
    if not m: continue
    abs_, nreads, uniq = int(m.group(1)), int(m.group(2)), int(m.group(3))
    depths = collections.Counter()
    for tok in m.group(4).split():
        d, c = tok.split(":"); depths[int(d)] = int(c)
    rows.append((abs_, nreads, uniq, depths))
print("frames traced:", len(rows))
# steady state = frames whose whole 0..42 window is inside the stream
evens = set(range(0, 43, 2))
steady = [(a,n,u,d) for (a,n,u,d) in rows if set(d.keys()) == evens]
print("steady-state frames (depth set == {0,2,..,42}):", len(steady))
if steady:
    sets = {tuple(sorted(d.items())) for (a,n,u,d) in steady}
    print("distinct depth/count signatures in steady state:", len(sets))
    for s in list(sets)[:3]:
        print("   ", " ".join("%d:%d" % kv for kv in s))
    print("   unique-index count:", sorted({u for (a,n,u,d) in steady}))
    print("   read-count:", sorted({n for (a,n,u,d) in steady}))
# cold start: which depths are collapsed onto first_ (the clamp target)
print("\ncold start (first 6 frames):")
for (a,n,u,d) in rows[:6]:
    print("   abs=%-3d reads=%-3d unique=%-3d depths=%s" % (a,n,u," ".join("%d:%d"%kv for kv in sorted(d.items()))))
# union of ALL physical offsets relative to t_ that were EVER read
allo = set()
for (a,n,u,d) in rows: allo |= set(d.keys())
print("\nunion of all depths ever read:", sorted(allo))
