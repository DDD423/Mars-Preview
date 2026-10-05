"""Non-interactive processes, bounded streams, and owned process trees."""
import codecs
import ctypes
import os
import queue
import signal
import subprocess
import threading
import time
from .registry import WorkError

OUTPUT_LIMIT = 4 * 1024 * 1024


if os.name == "nt":
    from ctypes import wintypes as W
    K = ctypes.WinDLL("kernel32", use_last_error=True)

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", W.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", W.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", W.DWORD), ("SchedulingClass", W.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class THREADENTRY(ctypes.Structure):
        _fields_ = [("dwSize", W.DWORD), ("cntUsage", W.DWORD), ("th32ThreadID", W.DWORD),
                    ("th32OwnerProcessID", W.DWORD), ("tpBasePri", W.LONG), ("tpDeltaPri", W.LONG), ("dwFlags", W.DWORD)]

    for name, restype, argtypes in [
        ("CreateJobObjectW", W.HANDLE, [ctypes.c_void_p, W.LPCWSTR]),
        ("SetInformationJobObject", W.BOOL, [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD]),
        ("AssignProcessToJobObject", W.BOOL, [W.HANDLE, W.HANDLE]),
        ("TerminateJobObject", W.BOOL, [W.HANDLE, W.UINT]),
        ("CloseHandle", W.BOOL, [W.HANDLE]),
        ("CreateToolhelp32Snapshot", W.HANDLE, [W.DWORD, W.DWORD]),
        ("Thread32First", W.BOOL, [W.HANDLE, ctypes.POINTER(THREADENTRY)]),
        ("Thread32Next", W.BOOL, [W.HANDLE, ctypes.POINTER(THREADENTRY)]),
        ("OpenThread", W.HANDLE, [W.DWORD, W.BOOL, W.DWORD]),
        ("ResumeThread", W.DWORD, [W.HANDLE]),
    ]:
        fn = getattr(K, name)
        fn.restype, fn.argtypes = restype, argtypes


class ProcessTree:
    def __init__(self, argv, cwd, env):
        self.job = None
        self.proc = None
        try:
            if os.name == "nt":
                self.job = K.CreateJobObjectW(None, None)
                if not self.job:
                    raise ctypes.WinError(ctypes.get_last_error())
                limit = EXTENDED()
                limit.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
                if not K.SetInformationJobObject(self.job, 9, ctypes.byref(limit), ctypes.sizeof(limit)):
                    raise ctypes.WinError(ctypes.get_last_error())
            self.proc = subprocess.Popen(argv, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
                                         creationflags=(0x08000000 | 0x4) if os.name == "nt" else 0,
                                         start_new_session=os.name != "nt")
            if os.name == "nt":
                if not K.AssignProcessToJobObject(self.job, int(self.proc._handle)):
                    raise ctypes.WinError(ctypes.get_last_error())
                # The initial thread remains suspended until job ownership is established.
                snapshot = K.CreateToolhelp32Snapshot(0x4, 0)
                if snapshot == ctypes.c_void_p(-1).value:
                    raise ctypes.WinError(ctypes.get_last_error())
                resumed = False
                try:
                    entry = THREADENTRY()
                    entry.dwSize = ctypes.sizeof(entry)
                    more = K.Thread32First(snapshot, ctypes.byref(entry))
                    while more:
                        if entry.th32OwnerProcessID == self.proc.pid:
                            thread = K.OpenThread(0x2, False, entry.th32ThreadID)
                            if not thread:
                                raise ctypes.WinError(ctypes.get_last_error())
                            try:
                                if K.ResumeThread(thread) == 0xFFFFFFFF:
                                    raise ctypes.WinError(ctypes.get_last_error())
                                resumed = True
                            finally:
                                K.CloseHandle(thread)
                        more = K.Thread32Next(snapshot, ctypes.byref(entry))
                finally:
                    K.CloseHandle(snapshot)
                if not resumed:
                    raise WorkError("JOB_ERROR", "无法恢复已绑定 Job Object 的进程")
        except Exception:
            if self.proc:
                self.proc.kill()
                self.proc.wait()
            self.close()
            raise

    def kill(self):
        if os.name == "nt" and self.job:
            K.TerminateJobObject(self.job, 1)
        elif self.proc:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def close(self):
        if self.job:
            K.CloseHandle(self.job)
            self.job = None


def run_terminal(args, powershell, context, cancel, emit, limit=OUTPUT_LIMIT):
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("DAVE_SLOT_")}
    env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    for key, binding in context["filter"]["bindings"].items():
        env["DAVE_SLOT_" + key[1:-1]] = binding["value"]
    if powershell:
        # No textual interpolation of user script or bindings.
        prefix = "$ErrorActionPreference = 'Stop'; $OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
        argv = ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", prefix + args["script"]]
    else:
        if os.name == "nt" and os.path.splitext(args["program"])[1].lower() in (".bat", ".cmd"):
            raise WorkError("BATCH_NOT_SUPPORTED", "terminal.exec 不直接运行批处理；请明确使用 PowerShell 脚本工具")
        argv = [args["program"]] + args["argv"]
    timeout = args.get("timeout") or context["settings"]["terminal_timeout"]
    try:
        tree = ProcessTree(argv, context["workspace"], env)
    except (OSError, ValueError) as exc:
        raise WorkError("PROCESS_START_FAILED", "无法启动程序", str(exc))
    proc, chunks = tree.proc, queue.Queue(maxsize=64)

    def reader(pipe, stream):
        try:
            while True:
                data = pipe.read1(8192)
                if not data:
                    break
                chunks.put((stream, data))
        finally:
            pipe.close()
            chunks.put((stream, None))

    threads = [threading.Thread(target=reader, args=(pipe, stream), daemon=True)
               for pipe, stream in ((proc.stdout, "stdout"), (proc.stderr, "stderr"))]
    for thread in threads:
        thread.start()
    decoders = {s: codecs.getincrementaldecoder("utf-8")("replace") for s in ("stdout", "stderr")}
    outputs = {"stdout": [], "stderr": []}
    start, kept, finished, truncated, reason = time.monotonic(), 0, 0, False, None
    try:
        while finished < 2:
            if reason is None and (cancel.is_set() or time.monotonic() - start > timeout):
                reason = "CANCELLED" if cancel.is_set() else "TIMEOUT"
                tree.kill()
            # Parent may exit with children holding pipe handles. Kill its remaining tree.
            if proc.poll() is not None:
                tree.kill()
            try:
                stream, data = chunks.get(timeout=0.05)
            except queue.Empty:
                continue
            if data is None:
                finished += 1
                text = decoders[stream].decode(b"", final=True)
            else:
                room = max(0, limit - kept)
                accepted = data[:room]
                kept += len(accepted)
                if len(accepted) < len(data) and not truncated:
                    truncated = True
                    emit("output_limit", {"message": "终端输出达到上限；继续排空剩余输出", "limit": limit})
                text = decoders[stream].decode(accepted)
            if text:
                outputs[stream].append(text)
                emit(stream, {"text": text})
        code = proc.wait()
        result = {"exit_code": code, "stdout": "".join(outputs["stdout"]),
                  "stderr": "".join(outputs["stderr"]), "truncated": truncated}
        if reason:
            raise WorkError(reason, "任务已停止" if reason == "CANCELLED" else "终端运行超时", result)
        if code:
            raise WorkError("PROCESS_EXIT", "程序退出码不为零", result)
        return result
    finally:
        tree.kill()
        tree.close()
        for thread in threads:
            thread.join(timeout=2)
