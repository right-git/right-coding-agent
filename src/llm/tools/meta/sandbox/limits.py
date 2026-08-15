"""Resource budgets for one sandboxed run.

Exceeding any of these raises `SandboxError` and aborts the whole run; the
numbers are also quoted in the `run_tools` docstring shown to the model, so
keep the two in sync.
"""

MAX_OPS = 100_000  # interpreter step budget
MAX_TOTAL_SLEEP = 600  # seconds of cumulative sleep() allowed
MAX_WALL_TIME = 900  # hard wall-clock timeout for one run_tools call
MAX_SLEEP_CALL = 300  # max single sleep() duration
MAX_PARALLEL = 32  # max branches in one parallel(...)
MAX_MEMORY_BYTES = 32 * 1024 * 1024  # 32 MB total for all sandbox data
MAX_SEQ_LEN = 1_000_000  # max length of any str/list produced by * or +
MAX_RANGE = 1_000_000  # max length of range()
MAX_INT_BITS = 256 * 1024  # ~78k decimal digits; one multiply stays fast
MEM_CHECK_EVERY = 256  # deep-size scope audit every N ops
