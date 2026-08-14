"""
Sandboxed Python-subset interpreter for LLM tool orchestration ("run_tools" engine).

Design goals:
- Syntax IS Python (LLMs already know it perfectly) -> parsed with `ast`, then
  executed by a tree-walking interpreter. No exec/eval, no compile of user code.
- Whitelist everything: node types, builtins, methods. No imports, no os/sys,
  no dunder access, no attribute traversal escapes.
- Async under the hood: tool calls are awaited server-side, `sleep()` costs
  ZERO tokens, `parallel(...)` is a special form that runs tool calls
  concurrently (asyncio.gather) without eager evaluation.
- Resource limits: op budget, wall-clock timeout, total-sleep cap.

NOTE: This is a strong application-level sandbox, but for truly adversarial
input run the whole interpreter inside an isolated worker (gVisor / Firecracker
/ WASM) as defense in depth.
"""

from __future__ import annotations

import ast
import asyncio
import random as _random
import sys
import time
from typing import Any, Callable, Awaitable


# ---------------------------------------------------------------- exceptions

class SandboxError(Exception):
    """FATAL: policy violation / resource budget. Not catchable by LLM code."""


class ExecError(Exception):
    """Runtime error (tool failure, bad key, etc). Catchable by LLM try/except."""


class _ReturnSignal(Exception):
    def __init__(self, value):
        self.value = value


class _BreakSignal(Exception):
    pass


class _ContinueSignal(Exception):
    pass


# ---------------------------------------------------------------- policy

ALLOWED_NODES = {
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign,
    ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass,
    ast.Try, ast.ExceptHandler, ast.Return, ast.Raise,
    ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.IfExp,
    ast.Call, ast.Name, ast.Constant, ast.Attribute, ast.Subscript,
    ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Slice,
    ast.JoinedStr, ast.FormattedValue,
    ast.ListComp, ast.DictComp, ast.comprehension,
    ast.Load, ast.Store, ast.Del,
    ast.And, ast.Or, ast.Not,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot,
    ast.Starred, ast.keyword,
}

# Safe methods reachable via attribute access (obj.method). No dunders ever.
SAFE_METHODS = {
    str: {"lower", "upper", "strip", "lstrip", "rstrip", "split", "rsplit",
          "join", "replace", "startswith", "endswith", "find", "count",
          "title", "capitalize", "zfill", "isdigit", "isalpha"},
    list: {"append", "extend", "pop", "insert", "remove", "index", "count",
           "sort", "reverse", "copy", "clear"},
    dict: {"get", "keys", "values", "items", "pop", "update", "copy", "clear"},
    set: {"add", "remove", "discard", "union", "intersection", "difference"},
    tuple: {"index", "count"},
}

MAX_OPS = 100_000          # interpreter step budget
MAX_TOTAL_SLEEP = 600      # seconds of cumulative sleep() allowed
MAX_WALL_TIME = 900        # hard wall-clock timeout for one run_tools call
MAX_SLEEP_CALL = 300       # max single sleep() duration
MAX_PARALLEL = 32          # max branches in one parallel(...)
MAX_MEMORY_BYTES = 32 * 1024 * 1024  # 32 MB total for all sandbox data
MAX_SEQ_LEN = 1_000_000    # max length of any str/list produced by * or +
MAX_RANGE = 1_000_000      # max length of range()
MAX_INT_BITS = 256 * 1024  # ~78k decimal digits; one multiply stays fast
MEM_CHECK_EVERY = 256      # deep-size scope audit every N ops


def _deep_size(roots: list, limit: int) -> int:
    """Approximate recursive size of all live sandbox data. Short-circuits
    as soon as the limit is exceeded, so cost is bounded."""
    seen: set[int] = set()
    stack = list(roots)
    total = 0
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        total += sys.getsizeof(o)
        if total > limit:
            return total
        if isinstance(o, dict):
            stack.extend(o.keys())
            stack.extend(o.values())
        elif isinstance(o, (list, tuple, set, frozenset)):
            stack.extend(o)
    return total


def _safe_range(*args):
    r = range(*args)
    if len(r) > MAX_RANGE:
        raise SandboxError(f"range() longer than {MAX_RANGE}")
    return r


# ---- guarded versions of amplifying methods -----------------------------
# These check the PROJECTED output size before allocating, because a single
# op can allocate arbitrarily large output that a periodic audit never sees.

def _g_zfill(s):
    def zfill(width):
        if not isinstance(width, int) or width > MAX_SEQ_LEN:
            raise SandboxError(f"zfill width exceeds {MAX_SEQ_LEN}")
        return s.zfill(width)
    return zfill

def _g_replace(s):
    def replace(old, new, *rest):
        old, new = str(old), str(new)
        n = s.count(old) if old else 0
        projected = len(s) + n * (len(new) - len(old))
        if projected > MAX_SEQ_LEN:
            raise SandboxError(f"replace() would produce > {MAX_SEQ_LEN} chars")
        return s.replace(old, new, *rest)
    return replace

def _g_join(s):
    def join(iterable):
        items = list(iterable)  # may be large but bounded by existing data
        if len(items) > MAX_SEQ_LEN:
            raise SandboxError(f"join() over > {MAX_SEQ_LEN} items")
        total = sum(len(str(x)) for x in items) + len(s) * max(len(items) - 1, 0)
        if total > MAX_SEQ_LEN:
            raise SandboxError(f"join() would produce > {MAX_SEQ_LEN} chars")
        return s.join(items)
    return join

def _g_extend(lst):
    def extend(other):
        other = list(other)
        if len(lst) + len(other) > MAX_SEQ_LEN:
            raise SandboxError(f"extend() would exceed {MAX_SEQ_LEN} items")
        return lst.extend(other)
    return extend

# method name -> factory that binds `obj` and returns a guarded callable
GUARDED_METHODS = {
    (str, "zfill"): _g_zfill,
    (str, "replace"): _g_replace,
    (str, "join"): _g_join,
    (list, "extend"): _g_extend,
}


def _make_builtins(interp: "Interpreter") -> dict[str, Any]:
    async def _sleep(seconds):
        seconds = float(seconds)
        if seconds < 0 or seconds > MAX_SLEEP_CALL:
            raise SandboxError(f"sleep() must be 0..{MAX_SLEEP_CALL}s")
        interp.total_sleep += seconds
        if interp.total_sleep > MAX_TOTAL_SLEEP:
            raise SandboxError(f"total sleep budget exceeded ({MAX_TOTAL_SLEEP}s)")
        await asyncio.sleep(seconds)

    return {
        # async primitives
        "sleep": _sleep,
        # numeric / misc
        "random": _random.random,
        "randint": _random.randint,
        "choice": _random.choice,
        "now": time.time,
        # pure builtins
        "len": len, "range": _safe_range, "abs": abs, "round": round,
        "min": min, "max": max, "sum": sum,
        "sorted": sorted, "reversed": lambda x: list(reversed(x)),
        "enumerate": enumerate, "zip": zip,
        "any": any, "all": all,
        "isinstance": isinstance,
        "type_name": lambda x: type(x).__name__,
        "str": str, "int": int, "float": float, "bool": bool,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
        "print": None,  # replaced per-run: appends to interp.logs
        # exception class allowed in `except`
        "Exception": Exception,
    }


# ---------------------------------------------------------------- interpreter

class Interpreter:
    def __init__(self, tools: dict[str, Callable[..., Awaitable[Any]]]):
        """
        tools: name -> async function. These are your real tool implementations
        (HTTP calls, MCP proxies, job queue clients, etc).
        """
        self.tools = tools
        self.ops = 0
        self.total_sleep = 0.0
        self.logs: list[str] = []
        self._mem_roots: list = []  # scope + in-flight containers

    # -- public entrypoint ----------------------------------------------------

    async def run(self, code: str) -> dict[str, Any]:
        """Execute LLM-generated code. Returns {"result", "logs", "error"}."""
        self.ops = 0
        self.total_sleep = 0.0
        self.logs = []

        # Wrap in a function so top-level `return` is valid (LLMs love writing it)
        indented = "\n".join("    " + line for line in code.splitlines())
        wrapped = f"def __sandbox_main__():\n{indented}\n"
        try:
            outer = ast.parse(wrapped, mode="exec")
            tree = outer.body[0]  # FunctionDef; we interpret its .body
        except SyntaxError as e:
            return {"result": None, "logs": [], "error": f"SyntaxError: {e}"}

        try:
            for stmt in tree.body:
                self._validate(stmt)
        except SandboxError as e:
            return {"result": None, "logs": [], "error": f"PolicyError: {e}"}

        env = _make_builtins(self)
        env["print"] = lambda *a: self.logs.append(" ".join(str(x) for x in a))
        scope: dict[str, Any] = {}
        self._mem_roots = [scope]

        async def _exec():
            try:
                for stmt in tree.body:
                    await self._exec_stmt(stmt, scope, env)
            except _ReturnSignal as r:
                return r.value
            return scope.get("_", None)

        try:
            result = await asyncio.wait_for(_exec(), timeout=MAX_WALL_TIME)
            return {"result": result, "logs": self.logs, "error": None}
        except SandboxError as e:
            return {"result": None, "logs": self.logs, "error": f"SandboxError: {e}"}
        except asyncio.TimeoutError:
            return {"result": None, "logs": self.logs,
                    "error": f"Timeout: exceeded {MAX_WALL_TIME}s wall clock"}
        except MemoryError:
            return {"result": None, "logs": self.logs,
                    "error": "SandboxError: memory exhausted"}
        except RecursionError:
            return {"result": None, "logs": self.logs,
                    "error": "SandboxError: recursion limit exceeded"}
        except Exception as e:
            return {"result": None, "logs": self.logs,
                    "error": f"{type(e).__name__}: {e}"}

    # -- static validation ----------------------------------------------------

    def _validate(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if type(node) not in ALLOWED_NODES:
                raise SandboxError(f"'{type(node).__name__}' is not allowed")
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("_"):
                    raise SandboxError(f"attribute '{node.attr}' is forbidden")
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                raise SandboxError(f"name '{node.id}' is forbidden")

    def _tick(self):
        self.ops += 1
        if self.ops > MAX_OPS:
            raise SandboxError(f"op budget exceeded ({MAX_OPS})")
        if self.ops % MEM_CHECK_EVERY == 0:
            size = _deep_size(self._mem_roots + [self.logs], MAX_MEMORY_BYTES)
            if size > MAX_MEMORY_BYTES:
                raise SandboxError(
                    f"memory budget exceeded (~{size // 1024} KB > "
                    f"{MAX_MEMORY_BYTES // 1024} KB)"
                )

    # -- statements -----------------------------------------------------------

    async def _exec_stmt(self, node, scope, env):
        self._tick()
        t = type(node)

        if t is ast.Expr:
            val = await self._eval(node.value, scope, env)
            scope["_"] = val

        elif t is ast.Assign:
            val = await self._eval(node.value, scope, env)
            for target in node.targets:
                self._assign(target, val, scope, env)

        elif t is ast.AnnAssign:
            if node.value is not None:
                val = await self._eval(node.value, scope, env)
                self._assign(node.target, val, scope, env)

        elif t is ast.AugAssign:
            cur = await self._eval(
                ast.Name(id=node.target.id, ctx=ast.Load()), scope, env
            ) if isinstance(node.target, ast.Name) else await self._eval(node.target, scope, env)
            rhs = await self._eval(node.value, scope, env)
            val = self._binop(node.op, cur, rhs)
            self._assign(node.target, val, scope, env)

        elif t is ast.If:
            branch = node.body if await self._eval(node.test, scope, env) else node.orelse
            for s in branch:
                await self._exec_stmt(s, scope, env)

        elif t is ast.While:
            while await self._eval(node.test, scope, env):
                self._tick()
                try:
                    for s in node.body:
                        await self._exec_stmt(s, scope, env)
                except _BreakSignal:
                    break
                except _ContinueSignal:
                    continue

        elif t is ast.For:
            iterable = await self._eval(node.iter, scope, env)
            for item in iterable:
                self._tick()
                self._assign(node.target, item, scope, env)
                try:
                    for s in node.body:
                        await self._exec_stmt(s, scope, env)
                except _BreakSignal:
                    break
                except _ContinueSignal:
                    continue

        elif t is ast.Try:
            try:
                for s in node.body:
                    await self._exec_stmt(s, scope, env)
            except (SandboxError, _ReturnSignal, _BreakSignal, _ContinueSignal):
                raise
            except Exception as exc:
                handled = False
                for handler in node.handlers:
                    if handler.name:
                        scope[handler.name] = exc
                    handled = True
                    for s in handler.body:
                        await self._exec_stmt(s, scope, env)
                    break
                if not handled:
                    raise
            for s in node.finalbody:
                await self._exec_stmt(s, scope, env)

        elif t is ast.Return:
            value = await self._eval(node.value, scope, env) if node.value else None
            raise _ReturnSignal(value)

        elif t is ast.Raise:
            msg = await self._eval(node.exc, scope, env) if node.exc else "raised"
            raise SandboxError(str(msg))

        elif t is ast.Break:
            raise _BreakSignal()

        elif t is ast.Continue:
            raise _ContinueSignal()

        elif t is ast.Pass:
            pass

        else:
            raise SandboxError(f"statement '{t.__name__}' not supported")

    def _assign(self, target, val, scope, env):
        if isinstance(target, ast.Name):
            scope[target.id] = val
        elif isinstance(target, (ast.Tuple, ast.List)):
            vals = list(val)
            if len(vals) != len(target.elts):
                raise SandboxError("unpacking length mismatch")
            for t_, v_ in zip(target.elts, vals):
                self._assign(t_, v_, scope, env)
        elif isinstance(target, ast.Subscript):
            raise SandboxError("subscript assignment: use dict.update / list.append")
        else:
            raise SandboxError("unsupported assignment target")

    # -- expressions ----------------------------------------------------------

    async def _eval(self, node, scope, env):
        self._tick()
        t = type(node)

        if t is ast.Constant:
            return node.value

        if t is ast.Name:
            if node.id in scope:
                return scope[node.id]
            if node.id in env:
                return env[node.id]
            if node.id in self.tools:
                return self.tools[node.id]
            raise SandboxError(f"name '{node.id}' is not defined")

        if t is ast.List:
            return [await self._eval(e, scope, env) for e in node.elts]
        if t is ast.Tuple:
            return tuple([await self._eval(e, scope, env) for e in node.elts])
        if t is ast.Set:
            return {await self._eval(e, scope, env) for e in node.elts}
        if t is ast.Dict:
            return {
                await self._eval(k, scope, env): await self._eval(v, scope, env)
                for k, v in zip(node.keys, node.values)
            }

        if t is ast.JoinedStr:
            parts = []
            for v in node.values:
                if isinstance(v, ast.FormattedValue):
                    parts.append(str(await self._eval(v.value, scope, env)))
                else:
                    parts.append(str(v.value))
            return "".join(parts)

        if t is ast.BoolOp:
            if isinstance(node.op, ast.And):
                result = True
                for v in node.values:
                    result = await self._eval(v, scope, env)
                    if not result:
                        return result
                return result
            else:  # Or
                result = False
                for v in node.values:
                    result = await self._eval(v, scope, env)
                    if result:
                        return result
                return result

        if t is ast.UnaryOp:
            operand = await self._eval(node.operand, scope, env)
            if isinstance(node.op, ast.Not):
                return not operand
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand

        if t is ast.BinOp:
            left = await self._eval(node.left, scope, env)
            right = await self._eval(node.right, scope, env)
            return self._binop(node.op, left, right)

        if t is ast.Compare:
            left = await self._eval(node.left, scope, env)
            for op, comparator in zip(node.ops, node.comparators):
                right = await self._eval(comparator, scope, env)
                ok = self._compare(op, left, right)
                if not ok:
                    return False
                left = right
            return True

        if t is ast.IfExp:
            cond = await self._eval(node.test, scope, env)
            return await self._eval(node.body if cond else node.orelse, scope, env)

        if t is ast.Subscript:
            obj = await self._eval(node.value, scope, env)
            if isinstance(node.slice, ast.Slice):
                lo = await self._eval(node.slice.lower, scope, env) if node.slice.lower else None
                hi = await self._eval(node.slice.upper, scope, env) if node.slice.upper else None
                st = await self._eval(node.slice.step, scope, env) if node.slice.step else None
                return obj[lo:hi:st]
            key = await self._eval(node.slice, scope, env)
            try:
                return obj[key]
            except (KeyError, IndexError, TypeError) as e:
                raise ExecError(f"subscript error: {e}")

        if t is ast.Attribute:
            obj = await self._eval(node.value, scope, env)
            for typ, methods in SAFE_METHODS.items():
                if isinstance(obj, typ) and node.attr in methods:
                    factory = GUARDED_METHODS.get((typ, node.attr))
                    if factory is not None:
                        return factory(obj)
                    return getattr(obj, node.attr)
            # allow dict-style attr read on plain dicts returned by tools
            if isinstance(obj, dict) and node.attr in obj:
                return obj[node.attr]
            raise SandboxError(
                f"attribute '{node.attr}' not allowed on {type(obj).__name__}"
            )

        if t is ast.ListComp:
            return await self._comprehension(node, scope, env, kind="list")
        if t is ast.DictComp:
            return await self._comprehension(node, scope, env, kind="dict")

        if t is ast.Call:
            return await self._call(node, scope, env)

        raise SandboxError(f"expression '{t.__name__}' not supported")

    async def _comprehension(self, node, scope, env, kind):
        if len(node.generators) != 1:
            raise SandboxError("only single-generator comprehensions allowed")
        gen = node.generators[0]
        iterable = await self._eval(gen.iter, scope, env)
        out_list, out_dict = [], {}
        local = dict(scope)
        self._mem_roots.append(out_list if kind == "list" else out_dict)
        try:
            for item in iterable:
                self._tick()
                self._assign(gen.target, item, local, env)
                skip = False
                for cond in gen.ifs:
                    if not await self._eval(cond, local, env):
                        skip = True
                        break
                if skip:
                    continue
                if kind == "list":
                    out_list.append(await self._eval(node.elt, local, env))
                else:
                    k = await self._eval(node.key, local, env)
                    v = await self._eval(node.value, local, env)
                    out_dict[k] = v
        finally:
            self._mem_roots.pop()
        return out_list if kind == "list" else out_dict

    # -- calls (incl. `parallel` special form) --------------------------------

    async def _call(self, node: ast.Call, scope, env):
        # SPECIAL FORM: parallel(tool_a(...), tool_b(...), ...)
        # Args are NOT eagerly awaited; they run concurrently via gather.
        if isinstance(node.func, ast.Name) and node.func.id in ("parallel", "run_functions_in_parallel"):
            if len(node.args) > MAX_PARALLEL:
                raise SandboxError(f"parallel(): max {MAX_PARALLEL} branches")
            coros = []
            for arg in node.args:
                if not isinstance(arg, ast.Call):
                    raise SandboxError("parallel() accepts only direct tool calls")
                coros.append(self._call(arg, scope, env))
            return list(await asyncio.gather(*coros, return_exceptions=False))

        func = await self._eval(node.func, scope, env)
        if func is None or not callable(func):
            raise SandboxError("attempt to call a non-callable")

        args = [await self._eval(a, scope, env) for a in node.args]
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise SandboxError("**kwargs unpacking not allowed")
            kwargs[kw.arg] = await self._eval(kw.value, scope, env)

        try:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except (SandboxError, ExecError, _ReturnSignal, _BreakSignal, _ContinueSignal):
            raise
        except MemoryError:
            raise SandboxError("memory exhausted during tool/method call")
        except RecursionError:
            raise SandboxError("recursion limit hit during call")
        except Exception as e:
            raise ExecError(f"tool call failed: {type(e).__name__}: {e}")

    # -- operators ------------------------------------------------------------

    def _binop(self, op, left, right):
        if isinstance(op, ast.Add):
            # s = s + s doubles exponentially -> would OOM between mem audits
            if isinstance(left, (str, list, tuple)) and isinstance(right, (str, list, tuple)):
                if len(left) + len(right) > MAX_SEQ_LEN:
                    raise SandboxError(f"sequence longer than {MAX_SEQ_LEN} via '+'")
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            # int * int: unbounded growth AND a single huge multiply is a CPU DoS
            if isinstance(left, int) and isinstance(right, int):
                if left.bit_length() + right.bit_length() > MAX_INT_BITS:
                    raise SandboxError("integer result too large")
                return left * right
            for seq, n in ((left, right), (right, left)):
                if isinstance(seq, (str, list, tuple)) and isinstance(n, int):
                    if len(seq) * max(n, 0) > MAX_SEQ_LEN:
                        raise SandboxError(f"sequence longer than {MAX_SEQ_LEN} via '*'")
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.FloorDiv):
            return left // right
        if isinstance(op, ast.Mod):
            return left % right
        if isinstance(op, ast.Pow):
            if isinstance(right, (int, float)) and abs(right) > 128:
                raise SandboxError("exponent too large")
            if isinstance(left, int) and isinstance(right, int) and right > 0:
                if left.bit_length() * right > MAX_INT_BITS:
                    raise SandboxError("integer result too large")
            return left ** right
        raise SandboxError("operator not allowed")

    def _compare(self, op, left, right):
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
        if isinstance(op, ast.In):
            return left in right
        if isinstance(op, ast.NotIn):
            return left not in right
        if isinstance(op, ast.Is):
            return left is right
        if isinstance(op, ast.IsNot):
            return left is not right
        raise SandboxError("comparison not allowed")


# ---------------------------------------------------------------- demo

if __name__ == "__main__":
    # Mock tool implementations (in production: HTTP/MCP/queue clients)
    JOBS: dict[str, dict] = {}

    async def add_task(url: str) -> str:
        job_id = f"job_{_random.randint(1000, 9999)}"
        JOBS[job_id] = {"status": "running", "started": time.time(), "url": url}
        return job_id

    async def get_job_status(job_id: str) -> str:
        job = JOBS.get(job_id)
        if not job:
            return "not_found"
        # pretend jobs finish after 0.5s
        if time.time() - job["started"] > 0.5:
            job["status"] = "success"
        return job["status"]

    async def send_notification_to_admin(admin_id: int) -> dict:
        await asyncio.sleep(0.3)  # simulate network latency
        return {"admin_id": admin_id, "delivered": True}

    TOOLS = {
        "add_task": add_task,
        "get_job_status": get_job_status,
        "send_notification_to_admin": send_notification_to_admin,
    }

    LLM_CODE = """
try:
    job_id = add_task("https://www.example.com")
    functions_responses = None
    if job_id:
        while get_job_status(job_id) == "running":
            sleep(0.2)
        job_status = get_job_status(job_id)
        print(f"job {job_id} finished: {job_status}")
        if job_status == "success":
            functions_responses = parallel(
                send_notification_to_admin(1),
                send_notification_to_admin(2),
                send_notification_to_admin(3),
                send_notification_to_admin(4),
            )
    return functions_responses
except Exception as e:
    return str(e)
"""

    BLOCKED_CODE = """
import os
os.system("rm -rf /")
"""

    async def main():
        interp = Interpreter(TOOLS)

        t0 = time.time()
        out = await interp.run(LLM_CODE)
        elapsed = time.time() - t0
        print("== main scenario ==")
        print("result:", out["result"])
        print("logs:", out["logs"])
        print("error:", out["error"])
        print(f"elapsed: {elapsed:.2f}s "
              f"(4 notifications x 0.3s ran in parallel, not 1.2s serial)")

        print("\n== blocked scenario ==")
        out2 = await interp.run(BLOCKED_CODE)
        print("error:", out2["error"])

    asyncio.run(main())