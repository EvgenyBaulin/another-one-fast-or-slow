#!/usr/bin/env python3
"""Measure train/test ONNX rows in crash-isolated, supervised ORT workers.

Example: .venv/bin/python scripts/measure.py --provider cuda --gpus 0,1
Resume uses final ids in <out>.jsonl, including failures/timeouts. Missing models
are never journaled. Use a new output path to retry previously failed rows or
to change provider/thread settings. --limit is applied BEFORE resume filtering.
CSV is an atomic snapshot of the journal; JSONL is the source of truth.
"""

import argparse
import collections
import csv
import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import re
import signal
import statistics
import sys
import tempfile
import time


COLUMNS = (
    "id", "file", "batch", "shape", "provider", "ms_median", "ms_min",
    "ms_first", "create_s", "n_runs", "measured_batch", "status", "error",
)
FINAL = {"ok", "ok_native_batch", "ok_reduced_batch", "failed", "timeout"}
TIMED_CAP = 20.0


def short_error(exc):
    return " ".join(str(exc).split())[:400]


def empty_result(row, provider):
    result = dict.fromkeys(COLUMNS, None)
    result.update({key: row[key] for key in ("id", "file", "batch", "shape")})
    result.update(provider=provider, n_runs=0, status="failed", error="")
    return result


def parse_shape(spec):
    overrides = {}
    if spec != "default":
        for part in re.split(r"[;,]", spec):
            name, dims = part.rsplit(":", 1)
            overrides[name] = [int(dim) for dim in dims.split("x")]
            if any(dim <= 0 for dim in overrides[name]):
                raise ValueError(f"Nonpositive shape override: {part}")
    return overrides


def real_inputs(model):
    initializers = {value.name for value in model.graph.initializer}
    initializers.update(value.values.name for value in model.graph.sparse_initializer)
    return [value for value in model.graph.input if value.name not in initializers]


def make_feeds(model, batch, overrides, np, onnx):
    """Mirror features.set_inputs, retaining scalars and real input dtypes."""
    rng = np.random.default_rng(0)
    feeds = {}
    inputs = real_inputs(model)
    unknown = overrides.keys() - {value.name for value in inputs}
    if unknown:
        raise ValueError(f"Unknown shape override inputs: {sorted(unknown)}")
    for value in inputs:
        if not value.type.HasField("tensor_type"):
            raise ValueError(f"Unsupported non-tensor input: {value.name}")
        tensor = value.type.tensor_type
        if not tensor.HasField("shape"):
            raise ValueError(f"Unknown input rank: {value.name}")
        dims = [
            dim.dim_value if dim.HasField("dim_value") and dim.dim_value > 0 else 128
            for dim in tensor.shape.dim
        ]
        if dims:
            dims = list(overrides.get(value.name, dims))
            if not dims:
                raise ValueError(f"Empty shape override: {value.name}")
            dims[0] = batch
        dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type))
        if np.issubdtype(dtype, np.floating):
            data = (rng.standard_normal(dims).astype(dtype) if dims
                    else np.asarray(0.5, dtype=dtype))
        elif np.issubdtype(dtype, np.integer):
            data = np.zeros(dims, dtype=dtype) if dims else np.asarray(100, dtype=dtype)
        elif dtype == np.dtype(bool):
            data = np.ones(dims, dtype=dtype)
        else:
            raise ValueError(f"Unsupported input dtype {dtype}: {value.name}")
        feeds[value.name] = data
        tensor.shape.ClearField("dim")
        for dim in dims:
            tensor.shape.dim.add().dim_value = dim
    # These inferred annotations can describe the old batch and mislead ORT's
    # optimizer. Weights and operator constants remain untouched.
    del model.graph.value_info[:]
    for value in model.graph.output:
        if value.type.HasField("tensor_type"):
            value.type.tensor_type.ClearField("shape")
    return feeds


def measure(row, config, connection):
    import numpy as np
    import onnx
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    if config["provider"] == "cuda" and hasattr(ort, "preload_dlls"):
        # CUDA/cuDNN come from pip wheels (nvidia-*), not from the system
        try:
            ort.preload_dlls()
        except Exception:
            pass
    result = empty_result(row, config["provider"])
    model = onnx.load(str(Path(config["data"]) / "data" / row["file"]))
    requested = int(float(row["batch"]))
    overrides = parse_shape(row["shape"])
    native = next((value.type.tensor_type.shape.dim[0].dim_value
                   for value in real_inputs(model)
                   if value.type.HasField("tensor_type")
                   and value.type.tensor_type.shape.dim
                   and value.type.tensor_type.shape.dim[0].HasField("dim_value")
                   and value.type.tensor_type.shape.dim[0].dim_value > 0), None)
    # Save original input metadata so retries never inherit a failed batch.
    original_inputs = [value.SerializeToString() for value in model.graph.input]
    candidates = [(requested, "ok")]
    if native is not None and native != requested:
        candidates.append((native, "ok_native_batch"))
    reduced = requested // 2
    while reduced >= 1:
        if reduced not in {batch for batch, _ in candidates}:
            candidates.append((reduced, "ok_reduced_batch"))
        reduced //= 2
    errors = []

    def report_phase(deadline):
        metrics = {key: result[key] for key in
                   ("measured_batch", "n_runs", "create_s", "ms_first")}
        connection.send(("phase", (deadline, metrics)))

    for batch, status in candidates:
        session = None
        feeds = None
        result.update(measured_batch=batch, n_runs=0, create_s=None, ms_first=None)
        report_phase(None)
        try:
            for value, serialized in zip(model.graph.input, original_inputs):
                value.ParseFromString(serialized)
            feeds = make_feeds(model, batch, overrides, np, onnx)
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.log_severity_level = 3
            options.intra_op_num_threads = config["threads"]
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            if config["provider"] == "cuda":
                providers = [("CUDAExecutionProvider", {"device_id": config["gpu"]}),
                             "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
            serialized = model.SerializeToString()
            start = time.perf_counter()
            session = ort.InferenceSession(serialized, sess_options=options, providers=providers)
            result["create_s"] = time.perf_counter() - start
            del serialized
            session.disable_fallback()
            if (config["provider"] == "cuda"
                    and "CUDAExecutionProvider" not in session.get_providers()):
                raise RuntimeError("CUDAExecutionProvider failed to initialize; refusing CPU-only timing")
            start = time.perf_counter()
            session.run(None, feeds)
            result["ms_first"] = (time.perf_counter() - start) * 1000
            session.run(None, feeds)
            times = []
            timing_start = time.perf_counter()
            # The parent also enforces this cap if a single sess.run hangs.
            report_phase(timing_start + TIMED_CAP)
            while len(times) < 30:
                if time.perf_counter() - timing_start >= TIMED_CAP:
                    break
                start = time.perf_counter()
                session.run(None, feeds)
                times.append((time.perf_counter() - start) * 1000)
                result["n_runs"] = len(times)
                report_phase(timing_start + TIMED_CAP)
                if len(times) >= 5 and sum(times) >= 1000:
                    break
            report_phase(None)
            if not times:
                raise RuntimeError("No timed runs completed")
            result.update(ms_median=statistics.median(times), ms_min=min(times),
                          status=status, error="; ".join(errors)[:700])
            return result
        except Exception as exc:
            errors.append(f"batch {batch}: {short_error(exc)}")
        finally:
            # Release GPU/CPU allocations before constructing the next session.
            session = None
            feeds = None
    result["error"] = "; ".join(errors)[:700]
    return result


def worker(connection, config):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Import-time ORT warnings precede its logger configuration. The parent
    # reports bounded exception messages, including errors from native crashes.
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 2)
    while True:
        try:
            row = connection.recv()
        except EOFError:
            break
        if row is None:
            break
        try:
            result = measure(row, config, connection)
        except Exception as exc:
            result = empty_result(row, config["provider"])
            result["error"] = short_error(exc)
        connection.send(("result", result))
    connection.close()


def load_journal(path, provider):
    records = {}
    if not path.exists():
        return records
    with path.open("r+b") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                record = json.loads(line)
            except (ValueError, UnicodeError):
                if not line.endswith(b"\n") and not stream.read(1):
                    stream.seek(offset)
                    stream.truncate()
                    print("Recovered incomplete last JSONL record", flush=True)
                    break
                raise ValueError(f"Invalid JSONL record at byte {offset}: {path}") from None
            if record.get("provider") != provider:
                raise ValueError(f"Journal provider mismatch: {path}")
            if record.get("status") in FINAL:
                records[record["id"]] = record
            if not line.endswith(b"\n"):
                stream.seek(0, os.SEEK_END)
                stream.write(b"\n")
                break
    return records


def write_csv(path, records):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records.values())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_rows(args):
    rows = {}
    for split in ("train", "test"):
        with (args.data / f"{split}.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                batch = float(row["batch"])
                if not math.isfinite(batch) or batch < 1 or not batch.is_integer():
                    raise ValueError(f"Invalid batch for {row['id']}")
                if row["id"] in rows:
                    raise ValueError(f"Duplicate row id: {row['id']}")
                rows[row["id"]] = {key: row[key] for key in ("id", "file", "batch", "shape")}
    if args.ids:
        selected = set(args.ids.split(","))
        if selected - rows.keys():
            raise ValueError(f"Unknown ids: {sorted(selected - rows.keys())}")
        rows = {key: row for key, row in rows.items() if key in selected}
    available = []
    sizes = {}
    missing = 0
    for row in rows.values():
        path = args.data / "data" / row["file"]
        try:
            sizes[row["file"]] = path.stat().st_size
        except FileNotFoundError:
            missing += 1
            continue
        available.append(row)
    available.sort(key=lambda row: (sizes[row["file"]], row["file"], row["id"]))
    if args.limit is not None:
        available = available[:args.limit]
    return available, missing


def stop_worker(slot):
    process = slot["process"]
    if process.is_alive():
        process.terminate()
        process.join(timeout=0.5)
    if process.is_alive():
        process.kill()
    process.join(timeout=0.5)
    slot["connection"].close()
    process.close()


def run(args):
    args.data = args.data.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    journal_path = Path(str(args.out) + ".jsonl")
    records = load_journal(journal_path, args.provider)
    rows, missing = load_rows(args)
    pending = [row for row in rows if row["id"] not in records]
    print(f"Selected {len(rows)} rows; resumed {len(rows) - len(pending)}; "
          f"pending {len(pending)}; missing files {missing}", flush=True)
    if not pending:
        write_csv(args.out, records)
        return 0
    if args.provider == "cuda":
        import onnxruntime as ort
        ort.set_default_logger_severity(3)
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise ValueError("CUDAExecutionProvider unavailable; install onnxruntime-gpu on the server")
    # Avoid hidden BLAS thread pools multiplying CPU concurrency.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    context = mp.get_context("spawn")
    # A shared file-group deque preserves global size ordering. A worker that
    # finishes a row retains its file until that group's rows are exhausted.
    groups = collections.deque()
    for row in pending:
        if not groups or groups[-1][0]["file"] != row["file"]:
            groups.append(collections.deque())
        groups[-1].append(row)
    slots = []
    start = time.perf_counter()
    completed = 0
    counts = collections.Counter()
    interrupted = False

    def spawn(index, group=None):
        parent, child = context.Pipe()
        config = dict(data=str(args.data), provider=args.provider, threads=args.threads,
                      gpu=args.gpus[index % len(args.gpus)])
        process = context.Process(target=worker, args=(child, config), daemon=True)
        process.start()
        child.close()
        return dict(process=process, connection=parent, row=None, group=group,
                    started=None, timed_deadline=None, index=index, recycle=False,
                    metrics={})

    def dispatch(slot):
        if not slot["group"]:
            slot["group"] = groups.popleft() if groups else None
        if slot["group"]:
            slot["row"] = slot["group"].popleft()
            slot["started"] = time.perf_counter()
            slot["timed_deadline"] = None
            slot["metrics"] = {}
            try:
                slot["connection"].send(slot["row"])
            except (ConnectionError, OSError):
                # A process can exit between is_alive() and dispatch. Let the
                # supervisor finalize this row and replace the process.
                slot["recycle"] = True

    with journal_path.open("a", encoding="utf-8") as journal:
        def finish(slot, result):
            nonlocal completed
            journal.write(json.dumps(result, allow_nan=False) + "\n")
            journal.flush()
            records[result["id"]] = result
            counts[result["status"]] += 1
            completed += 1
            ms = result["ms_median"]
            display = f"{ms:.3f}ms" if ms is not None else "-"
            print(f"[{completed}/{len(pending)}] {time.perf_counter() - start:.1f}s "
                  f"{result['status']} {result['id']} {display}", flush=True)
            slot["row"] = None
            slot["recycle"] = result["status"] == "failed"

        def receive(slot):
            try:
                while slot["connection"].poll():
                    kind, payload = slot["connection"].recv()
                    if kind == "result":
                        finish(slot, payload)
                        return
                    slot["timed_deadline"], slot["metrics"] = payload
            except (EOFError, ConnectionError, OSError):
                pass  # Process exit below produces a final failed record.

        try:
            n_workers = len(args.gpus) if args.provider == "cuda" else args.workers
            for index in range(n_workers):
                slot = spawn(index)
                slots.append(slot)
                dispatch(slot)
            while completed < len(pending):
                connections = [slot["connection"] for slot in slots if slot["row"]]
                ready = wait(connections, timeout=0.1)
                for slot in slots:
                    if slot["connection"] in ready:
                        receive(slot)
                    now = time.perf_counter()
                    dead = not slot["process"].is_alive()
                    timed_out = slot["row"] is not None and (
                        now - slot["started"] >= args.timeout
                        or (slot["timed_deadline"] is not None
                            and now >= slot["timed_deadline"]))
                    if dead or timed_out or slot["recycle"]:
                        if slot["row"]:
                            result = empty_result(slot["row"], args.provider)
                            result.update(slot["metrics"])
                            result["status"] = "timeout" if timed_out else "failed"
                            result["error"] = (
                                f"Timed inference exceeded {TIMED_CAP:g}s cap"
                                if timed_out and slot["timed_deadline"] is not None
                                and now >= slot["timed_deadline"] else
                                f"Row exceeded {args.timeout:g}s timeout" if timed_out else
                                f"Worker exited with code {slot['process'].exitcode} (crash/OOM)")
                            finish(slot, result)
                        group, index = slot["group"], slot["index"]
                        stop_worker(slot)
                        replacement = spawn(index, group)
                        slot.clear()
                        slot.update(replacement)
                    if slot["row"] is None:
                        dispatch(slot)
        except KeyboardInterrupt:
            interrupted = True
            # Save already completed responses; active rows remain resumable.
            for slot in slots:
                if slot["row"]:
                    receive(slot)
            print("Interrupted; active rows will be retried on resume", flush=True)
        finally:
            for slot in slots:
                stop_worker(slot)
            write_csv(args.out, records)
    print(f"Finished {completed}: {dict(counts)}; CSV: {args.out}", flush=True)
    return 130 if interrupted else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpus", default="0,1", help="One CUDA worker per unique device id")
    parser.add_argument("--workers", type=int, default=4, help="CPU workers (ignored for CUDA)")
    parser.add_argument("--threads", type=int, default=1, help="Intra-op threads per worker")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ids", help="Comma-separated row ids")
    args = parser.parse_args()
    try:
        args.gpus = [int(value) for value in args.gpus.split(",")]
    except ValueError:
        parser.error("--gpus must be comma-separated integer device ids")
    if not args.gpus or min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must contain unique nonnegative device ids")
    if args.workers < 1 or args.threads < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--workers, --threads and --timeout must be positive")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.out is None:
        args.out = args.data / f"measure_{args.provider}.csv"
    try:
        return run(args)
    except (ValueError, OSError) as exc:
        print(f"Error: {short_error(exc)}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
