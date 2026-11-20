"""Convert device cycles at the configured frequency; host times remain nanoseconds."""
DEFAULT_CLOCK_HZ = 1_000_000_000


def clock_hz(source):
    return source.get("hardware", source).get("clock_hz", DEFAULT_CLOCK_HZ)


def cycles_ns(cycles, source):
    hz = clock_hz(source)
    return cycles if hz == DEFAULT_CLOCK_HZ else cycles * (1e9 / hz)
