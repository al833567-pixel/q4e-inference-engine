"""pytest fixtures（沙箱對策）。

tmp_path：本環境沙箱拒 pytest 預設 temp 機制（系統 temp 與 --basetemp 的
建後 scandir／rmtree 皆 PermissionError；實測見 03A §四）。改用 repo 內
_testtmp/ 直接建目錄：同進程建＋用＋刪＝實測通過。
"""
import os
import shutil
import time

import pytest

_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_testtmp")


@pytest.fixture()
def tmp_path(request):
    name = "".join(c if c.isalnum() else "_" for c in request.node.name)[:40]
    d = os.path.join(_BASE, f"{os.getpid()}_{time.time_ns()}_{name}")
    os.makedirs(d, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
