"""Store the optional Hugging Face token encrypted for the current Windows user."""
import ctypes
from ctypes import POINTER, Structure, byref, c_byte, c_uint32, c_void_p
from pathlib import Path


class _Blob(Structure):
    _fields_ = [("cbData", c_uint32), ("pbData", POINTER(c_byte))]


def _crypt(data, protect):
    if not hasattr(ctypes, "WinDLL"):
        raise OSError("權杖安全保存功能目前僅支援 Windows")
    crypt = ctypes.WinDLL("Crypt32.dll", use_last_error=True)
    kernel = ctypes.WinDLL("Kernel32.dll", use_last_error=True)
    source = ctypes.create_string_buffer(data)
    in_blob = _Blob(len(data), ctypes.cast(source, POINTER(c_byte)))
    out_blob = _Blob()
    if protect:
        ok = crypt.CryptProtectData(byref(in_blob), "mp3-transcriber Hugging Face token",
                                    None, None, None, 1, byref(out_blob))
    else:
        ok = crypt.CryptUnprotectData(byref(in_blob), None, None, None, None, 1, byref(out_blob))
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        kernel.LocalFree(c_void_p(ctypes.cast(out_blob.pbData, c_void_p).value))


def save_token(path, token):
    Path(path).write_bytes(_crypt(token.encode("utf-8"), True))


def load_token(path):
    target = Path(path)
    if not target.is_file():
        return ""
    return _crypt(target.read_bytes(), False).decode("utf-8")


def delete_token(path):
    Path(path).unlink(missing_ok=True)
