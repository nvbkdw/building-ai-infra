"""Threads vs asyncio micro-benchmarks. Each case runs in a fresh subprocess:
    python3 bench.py <case> <args...>   -> one JSON line
    python3 bench.py                    -> driver: runs everything, prints tables
"""
import asyncio, threading, time, resource, os, sys, gc, subprocess, socket, json, statistics

def rss_kb():
    if sys.platform == "linux":
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") // 1024
    return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                              capture_output=True, text=True).stdout)

class Meter:
    """Wall time, CPU time (user+sys, all threads), context switches, RSS delta."""
    def __enter__(self):
        gc.collect()
        self.rss0 = rss_kb()
        r = resource.getrusage(resource.RUSAGE_SELF)
        self.u0, self.s0, self.v0, self.i0 = r.ru_utime, r.ru_stime, r.ru_nvcsw, r.ru_nivcsw
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        self.wall = time.perf_counter() - self.t0
        r = resource.getrusage(resource.RUSAGE_SELF)
        self.user, self.sys = r.ru_utime - self.u0, r.ru_stime - self.s0
        self.vcs, self.ics = r.ru_nvcsw - self.v0, r.ru_nivcsw - self.i0
        self.rss_mb = (rss_kb() - self.rss0) / 1024
        self.peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**10 if sys.platform == 'linux' else 2**20)  # KB on Linux, bytes on macOS
    def json(self, **extra):
        return json.dumps(dict(wall_ms=self.wall*1e3, cpu_ms=(self.user+self.sys)*1e3,
                               usr_ms=self.user*1e3, sys_ms=self.sys*1e3, vcs=self.vcs, ics=self.ics,
                               rss_mb=self.rss_mb, peak_mb=self.peak_mb, **extra))

# ---------------------------------------------------------------- 1. footprint at rest
def case_thread_footprint(n):
    go = threading.Event(); threads = []; created = 0
    try:
        with Meter() as m:
            for _ in range(n):
                t = threading.Thread(target=go.wait); t.start(); threads.append(t); created += 1
    except RuntimeError as e:                       # "can't start new thread"
        go.set(); [t.join() for t in threads]
        print(json.dumps(dict(error=str(e), created=created))); return
    go.set(); [t.join() for t in threads]
    print(m.json(n=n, per_unit_us=m.wall/n*1e6, per_unit_kb=m.rss_mb*1024/n))

async def _task_footprint(n):
    go = asyncio.Event()
    async def wait(): await go.wait()
    with Meter() as m:
        tasks = [asyncio.create_task(wait()) for _ in range(n)]
        await asyncio.sleep(0)                       # every task runs to its first await
    go.set(); await asyncio.gather(*tasks)
    print(m.json(n=n, per_unit_us=m.wall/n*1e6, per_unit_kb=m.rss_mb*1024/n))
def case_task_footprint(n): asyncio.run(_task_footprint(n))

# ---------------------------------------------------------------- 2. context-switch cost
def case_thread_pingpong(rounds):
    a, b = threading.Semaphore(0), threading.Semaphore(0)
    def ping():
        for _ in range(rounds): b.release(); a.acquire()
    def pong():
        for _ in range(rounds): b.acquire(); a.release()
    t = threading.Thread(target=pong)
    with Meter() as m:
        t.start(); ping(); t.join()
    print(m.json(rounds=rounds, per_switch_us=m.wall/(2*rounds)*1e6))

async def _task_pingpong(rounds):
    a, b = asyncio.Semaphore(0), asyncio.Semaphore(0)
    async def ping():
        for _ in range(rounds): b.release(); await a.acquire()
    async def pong():
        for _ in range(rounds): await b.acquire(); a.release()
    with Meter() as m:
        await asyncio.gather(ping(), pong())
    print(m.json(rounds=rounds, per_switch_us=m.wall/(2*rounds)*1e6))
def case_task_pingpong(rounds): asyncio.run(_task_pingpong(rounds))

async def _task_sleep0(rounds):
    async def spin():
        for _ in range(rounds): await asyncio.sleep(0)
    with Meter() as m:
        await asyncio.gather(spin(), spin())
    print(m.json(rounds=rounds, per_switch_us=m.wall/(2*rounds)*1e6))
def case_task_sleep0(rounds): asyncio.run(_task_sleep0(rounds))

def case_thread_pairs(pairs, rounds):
    def pair():
        a, b = threading.Semaphore(0), threading.Semaphore(0)
        def pong():
            for _ in range(rounds): b.acquire(); a.release()
        t = threading.Thread(target=pong); t.start()
        for _ in range(rounds): b.release(); a.acquire()
        t.join()
    threads = [threading.Thread(target=pair) for _ in range(pairs)]
    with Meter() as m:
        for t in threads: t.start()
        for t in threads: t.join()
    print(m.json(pairs=pairs, switches=2*rounds*pairs, per_switch_us=m.wall/(2*rounds*pairs)*1e6))

async def _task_pairs(pairs, rounds):
    async def pair():
        a, b = asyncio.Semaphore(0), asyncio.Semaphore(0)
        async def ping():
            for _ in range(rounds): b.release(); await a.acquire()
        async def pong():
            for _ in range(rounds): await b.acquire(); a.release()
        await asyncio.gather(ping(), pong())
    with Meter() as m:
        await asyncio.gather(*(pair() for _ in range(pairs)))
    print(m.json(pairs=pairs, switches=2*rounds*pairs, per_switch_us=m.wall/(2*rounds*pairs)*1e6))
def case_task_pairs(pairs, rounds): asyncio.run(_task_pairs(pairs, rounds))

# ---------------------------------------------------------------- 3. concurrency sweep (simulated I/O)
def case_thread_sweep(n, iters, ms):
    d = ms/1e3
    def work():
        for _ in range(iters): time.sleep(d)
    threads = [threading.Thread(target=work) for _ in range(n)]
    try:
        with Meter() as m:
            for t in threads: t.start()
            for t in threads: t.join()
    except RuntimeError as e:
        print(json.dumps(dict(error=str(e), n=n))); return
    print(m.json(n=n, ideal_ms=iters*ms))

async def _asyncio_sweep(n, iters, ms):
    d = ms/1e3
    async def work():
        for _ in range(iters): await asyncio.sleep(d)
    with Meter() as m:
        async with asyncio.TaskGroup() as tg:
            for _ in range(n): tg.create_task(work())
    print(m.json(n=n, ideal_ms=iters*ms))
def case_asyncio_sweep(n, iters, ms): asyncio.run(_asyncio_sweep(n, iters, ms))

# ---------------------------------------------------------------- 4. real sockets
def case_echo_server():
    async def main():
        async def handle(r, w):
            try:
                while data := await r.read(64): w.write(data); await w.drain()
            finally: w.close()
        srv = await asyncio.start_server(handle, "127.0.0.1", 0)
        print(srv.sockets[0].getsockname()[1], flush=True)
        async with srv: await srv.serve_forever()
    asyncio.run(main())

def case_thread_clients(n, rtts, port):
    def client():
        with socket.create_connection(("127.0.0.1", port)) as s:
            for _ in range(rtts):
                s.sendall(b"x"*16); s.recv(64)
    threads = [threading.Thread(target=client) for _ in range(n)]
    try:
        with Meter() as m:
            for t in threads: t.start()
            for t in threads: t.join()
    except RuntimeError as e:
        print(json.dumps(dict(error=str(e), n=n))); return
    print(m.json(n=n))

async def _asyncio_clients(n, rtts, port):
    async def client():
        r, w = await asyncio.open_connection("127.0.0.1", port)
        for _ in range(rtts):
            w.write(b"x"*16); await w.drain(); await r.readexactly(16)
        w.close(); await w.wait_closed()
    with Meter() as m:
        async with asyncio.TaskGroup() as tg:
            for _ in range(n): tg.create_task(client())
    print(m.json(n=n))
def case_asyncio_clients(n, rtts, port): asyncio.run(_asyncio_clients(n, rtts, port))

# ---------------------------------------------------------------- 5. GIL convoy: I/O wake-up latency next to CPU threads
def case_convoy(cpu_threads, samples):
    stop = threading.Event()
    def burn():
        while not stop.is_set(): sum(range(10_000))
    burners = [threading.Thread(target=burn, daemon=True) for _ in range(cpu_threads)]
    for b in burners: b.start()
    lat = []
    for _ in range(samples):
        t0 = time.perf_counter(); time.sleep(0.001); lat.append((time.perf_counter()-t0-0.001)*1e3)
    stop.set()
    print(json.dumps(dict(cpu_threads=cpu_threads, p50_ms=statistics.median(lat),
                          p99_ms=sorted(lat)[int(0.99*len(lat))], max_ms=max(lat))))

# ---------------------------------------------------------------- driver
def run(case, *args):
    try:
        out = subprocess.run([sys.executable, __file__, case, *map(str, args)], capture_output=True, text=True, timeout=150)
    except subprocess.TimeoutExpired:
        return dict(error="timeout >150 s")
    if out.returncode or not out.stdout.strip():
        return dict(error=(out.stderr.strip().splitlines() or ["?"])[-1])
    return json.loads(out.stdout.strip().splitlines()[-1])

def fmt(r, extra=""):
    if "error" in r: return f"ERROR: {r['error']} {extra}"
    return (f"wall {r['wall_ms']:8.1f} ms | cpu {r['cpu_ms']:8.1f} ms (usr {r['usr_ms']:7.1f} sys {r['sys_ms']:7.1f})"
            f" | ctx vol {r['vcs']:>8} invol {r['ics']:>6} | rssΔ {r['rss_mb']:6.1f} MB peak {r['peak_mb']:6.1f} MB {extra}")

def driver():
    print(f"Python {sys.version.split()[0]}, {os.cpu_count()} CPUs, GIL={sys._is_gil_enabled()}, switchinterval={sys.getswitchinterval()}\n")
    print("== 1. footprint at rest (parked waiters) ==")
    for n in (1000, 5000, 10000):
        r = run("thread_footprint", n); print(f"threads n={n:<6}", fmt(r, f"| per thread {r.get('per_unit_us',0):6.1f} µs {r.get('per_unit_kb',0):6.1f} KB" if 'error' not in r else f"created={r.get('created')}"))
    for n in (1000, 10000, 100000):
        r = run("task_footprint", n);   print(f"tasks   n={n:<6}", fmt(r, f"| per task   {r['per_unit_us']:6.1f} µs {r['per_unit_kb']:6.1f} KB"))
    print("\n== 2. context switch cost (ping-pong, 2 switches per round) ==")
    for case in ("thread_pingpong", "task_pingpong", "task_sleep0"):
        r = run(case, 50000); print(f"{case:<16}", fmt(r, f"| per switch {r['per_switch_us']:5.2f} µs"))
    print("\n== 3. concurrency sweep: N waiters x 10 x 10 ms simulated I/O (ideal 100 ms) ==")
    for n in (10, 100, 1000, 5000, 9000, 10000):
        for case in ("thread_sweep", "asyncio_sweep"):
            r = run(case, n, 10, 10); print(f"{case:<14} n={n:<6}", fmt(r))
    print("\n== 4. real TCP echo: N connections x 20 round trips (server = asyncio, separate process) ==")
    srv = subprocess.Popen([sys.executable, __file__, "echo_server"], stdout=subprocess.PIPE, text=True)
    port = int(srv.stdout.readline())
    try:
        for n in (100, 1000, 4000):
            for case in ("thread_clients", "asyncio_clients"):
                r = run(case, n, 20, port); print(f"{case:<16} n={n:<5}", fmt(r))
    finally:
        srv.terminate()
    print("\n== 5. GIL convoy: wake-up latency of a 1 ms sleep in an I/O thread, with K CPU-bound threads alongside ==")
    for k in (0, 1, 2):
        r = run("convoy", k, 300); print(f"cpu_threads={k}  p50 {r['p50_ms']:.2f} ms  p99 {r['p99_ms']:.2f} ms  max {r['max_ms']:.2f} ms")

def extra():
    print("== A. switch throughput vs number of switching pairs (200k switches total) ==")
    for pairs in (1, 4, 16, 64, 256):
        r = run("thread_pairs", pairs, 100000 // pairs); print(f"thread_pairs P={pairs:<4}", fmt(r, f"| per switch {r.get('per_switch_us', float('nan')):6.2f} µs" if 'error' not in r else ""))
    for pairs in (1, 16, 256, 4096):
        r = run("task_pairs", pairs, 100000 // pairs);   print(f"task_pairs   P={pairs:<4}", fmt(r, f"| per switch {r['per_switch_us']:6.2f} µs"))
    print("\n== B. thread sweep, intermediate N (10 x 10 ms) ==")
    for n in (2000, 3000):
        r = run("thread_sweep", n, 10, 10); print(f"thread_sweep   n={n:<6}", fmt(r))
    print("\n== C. is it thread COUNT or WAKE-UP RATE? (same total sleep budget per waiter) ==")
    for case, n, iters, ms in (("thread_sweep", 5000, 1, 100), ("asyncio_sweep", 5000, 1, 100),
                               ("thread_sweep", 100, 100, 1), ("asyncio_sweep", 100, 100, 1),
                               ("thread_sweep", 1000, 100, 1), ("asyncio_sweep", 1000, 100, 1)):
        r = run(case, n, iters, ms); print(f"{case:<14} n={n:<5} {iters:>3} x {ms:>3} ms (= {n*1000//ms:>8} wakeups/s)", fmt(r))

def linux():
    import platform; print(platform.platform(), "|", f"Python {sys.version.split()[0]}, {os.cpu_count()} CPUs")
    print("== footprint ==")
    for n in (1000, 5000):
        r = run("thread_footprint", n); print(f"threads n={n:<6}", fmt(r, f"| per thread {r.get('per_unit_us',0):6.1f} µs {r.get('per_unit_kb',0):6.1f} KB" if 'error' not in r else ""))
    r = run("task_footprint", 10000); print(f"tasks   n=10000 ", fmt(r, f"| per task {r['per_unit_us']:6.1f} µs {r['per_unit_kb']:6.1f} KB"))
    print("== switch cost ==")
    for pairs in (1, 16, 256):
        r = run("thread_pairs", pairs, 100000 // pairs); print(f"thread_pairs P={pairs:<4}", fmt(r, f"| per switch {r.get('per_switch_us', float('nan')):6.2f} µs"))
        r = run("task_pairs", pairs, 100000 // pairs);   print(f"task_pairs   P={pairs:<4}", fmt(r, f"| per switch {r['per_switch_us']:6.2f} µs"))
    print("== sweep 10 x 10 ms ==")
    for n in (100, 1000, 2000, 5000, 10000):
        for case in ("thread_sweep", "asyncio_sweep"):
            r = run(case, n, 10, 10); print(f"{case:<14} n={n:<6}", fmt(r))
    print("== convoy ==")
    for k in (0, 1):
        r = run("convoy", k, 300); print(f"cpu_threads={k}  p50 {r['p50_ms']:.2f} ms  p99 {r['p99_ms']:.2f} ms")

if __name__ == "__main__":
    if len(sys.argv) == 1: driver()
    elif sys.argv[1] == "linux": linux()
    elif sys.argv[1] == "extra": extra()
    else:
        case, *args = sys.argv[1:]
        globals()["case_" + case](*map(int, args))
