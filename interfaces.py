"""介面合約（v2：spec → q4e 核心）。

這是 spec 層（Python，本 repo）與 **q4e 核心**（C++）之間的**契約**：
- `InferenceEngine`：q4e-server 實作（載 GGUF + 推理 + 暴露**真實** router 選擇）。
- `CacheManager`：本 repo 實作**決策邏輯**（LRU + 確定性預取）；q4e 核心接**真實 placement**（GPU offload）。
- `NgramTable`：本 repo 實作**解析**（CPU-only）；q4e 核心接 **offload**（host/SSD + 非同步預取）。

實作時：照下面簽章填方法即可，簽章不改。
本 repo 的邏輯（CacheManager/NgramTable）用 fake placement 測（tests/），不碰 GPU。
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------- #
# 共用型別
# --------------------------------------------------------------------------- #
TokenId = int
ExpertId = int
LayerId = int

# 快取層別：vram=L1, ram=L2, ssd=L3
Tier = str  # "vram" | "ram" | "ssd"
TIERS: tuple[Tier, ...] = ("vram", "ram", "ssd")

# 一次 touch 的命中統計：{ "vram": 8, "ram": 2, "ssd": 0 }
HitBreakdown = dict[str, int]


@dataclass
class CacheConfig:
    """兩層快取參數（A 定，B 用）。"""
    n_layers: int = 48
    n_experts_per_layer: int = 512        # routed 專家數
    active_routed: int = 10               # 每 token 激活 routed
    has_shared: bool = True               # 每層 1 個 shared（常駐熱）
    vram_capacity_per_layer: int = 20     # L1：2×激活（20 routed + shared）
    ram_capacity_per_layer: int = 142     # L2：N×一輪未命中（≈18 GB 預算）
    ram_mult: int = 1                     # N 倍數
    ssd_path: str = "D:\\"                # L3：冷專家 + N-gram
    ng_on_ssd: bool = True
    # 快取過濾：只預取進 ram（不進 vram），保護熱層
    prefetch_into: Tier = "ram"


@dataclass
class EngineConfig:
    """推理引擎參數（B 用）。"""
    gguf_dir: str = r"D:\AI_Model\Qwen3.8-Flash-Next"
    ngl: int | str = "auto"               # VRAM 層（B 依 GPU 空間定）
    max_ctx: int = 2048
    n_batch: int = 512
    n_threads: int = 0                    # 0 = 自動


@dataclass
class ServerConfig:
    """API 層參數（A 用）。"""
    host: str = "127.0.0.1"
    port: int = 8080
    api_keys: list[str] = field(default_factory=list)   # 空 = 讀 env DSH_API_KEY
    model_name: str = "qwen3.8-flash-next"
    stream_default: bool = False
    engine: str = "stub"        # "stub"（A，不載模型）| "llama"（B，lazy load GGUF）


# --------------------------------------------------------------------------- #
# InferenceEngine（B 實作）
# --------------------------------------------------------------------------- #
class InferenceEngine(abc.ABC):
    """GGUF 載入 + 逐 token 推理 + 暴露每層 router 選擇。

    B（略笨 AI，有 GPU）實作。A 只依賴這些簽章（用 stub/fake 測）。
    """

    @abc.abstractmethod
    def load(self, cfg: EngineConfig) -> None:
        """載入 GGUF（多片自動拼接），設定 VRAM 層(-ngl)/ctx。"""

    @abc.abstractmethod
    def decode(self, token_ids: list[TokenId]) -> TokenId:
        """給一組 token（或一輪上下文），回下一個 token id。"""

    @abc.abstractmethod
    def router_experts(self, layer: LayerId, token: TokenId) -> list[ExpertId]:
        """回第 layer 層、該 token 選到的 routed 專家 id（top-10）。

        B 從引擎的 router logits 取 top-k；A 用 fake 版回固定/隨機 id 測快取邏輯。
        """

    @abc.abstractmethod
    def free(self) -> None:
        """釋放模型/記憶體。"""

    # -- 選用（B 若支援就填） ------------------------------------------------ #
    def expert_weight_bytes(self, layer: LayerId, expert: ExpertId) -> int:
        """該專家權重位元組數（供快取大小計算）。預設回 0（B 可覆寫）。"""
        return 0


# --------------------------------------------------------------------------- #
# CacheManager（A 寫邏輯，B 接 placement）
# --------------------------------------------------------------------------- #
class CacheManager(abc.ABC):
    """兩層快取決策：LRU 取代 + 需求分頁 + 確定性預取 + 命中統計。

    A（聰明 AI）實作**決策邏輯**（純 Python，fake placement 測）。
    B（略笨 AI）實作/覆寫 `place()` 把「逐出/提升」接到真實 GPU offload。
    """

    def __init__(self, cfg: CacheConfig) -> None:
        self.cfg = cfg

    @abc.abstractmethod
    def touch(self, layer: LayerId, expert_ids: list[ExpertId]) -> HitBreakdown:
        """一個 token 在第 layer 層用到這些專家 → 回各層命中統計 {vram,ram,ssd}。

        A 實作：查 L1→L2→L3（LRU），未命中從下一層載入並提升。
        """

    @abc.abstractmethod
    def prefetch(self, layer: LayerId, hint_ids: list[ExpertId]) -> None:
        """投機預取（確定性：上一 token 該層的選擇）。可失敗、可被 LRU 逐出。

        A 實作：把 hint 放進 `cfg.prefetch_into`（預設 ram），不污染 vram。
        """

    @abc.abstractmethod
    def stats(self) -> dict[str, Any]:
        """回命中統計 + 重疊度（供基準/除錯）。"""

    # -- B 覆寫：把「載入/逐出」接真實 placement ----------------------------- #
    def place(self, layer: LayerId, expert: ExpertId, to_tier: Tier) -> None:
        """把某專家搬到某層（vram/ram/ssd）。A 的 fake 版只記狀態；B 覆寫做真實 H2D/D2H。"""
        raise NotImplementedError("B 應覆寫 place() 接真實 placement")

    def evict(self, layer: LayerId, expert: ExpertId, to_tier: Tier) -> None:
        """把某專家逐出到某層（LRU 滿時）。A 的 fake 版只記狀態；B 覆寫做真實 D2H/寫 SSD。"""
        raise NotImplementedError("B 應覆寫 evict() 接真實 placement")


# --------------------------------------------------------------------------- #
# NgramTable（A 寫解析，B 接 offload）
# --------------------------------------------------------------------------- #
class NgramTable(abc.ABC):
    """N-gram embedding 表（per_layer_token_embd，51.2B，4.5 bit/value，28.8 GB）。

    A（聰明 AI）實作**解析**（讀 GGUF 位元組，CPU-only）+ 解碼 + 隨機讀 bench。
    B（略笨 AI）實作/覆寫 `offload()` 把表離到 host/SSD + 非同步預取。
    """

    @abc.abstractmethod
    def parse(self, gguf_path: str) -> None:
        """解析 N-gram 張量結構（shape/dtype/block 佈局）。"""

    @abc.abstractmethod
    def lookup(self, token_ids: list[TokenId]) -> list[float]:
        """layer 2 查 16 行（8 bigram + 8 trigram hash）→ 回 16×embd 值。"""

    @abc.abstractmethod
    def bench_random_reads(self, n: int = 1000) -> dict[str, float]:
        """隨機讀 n 次（每次 16 行）→ 回 {mean_ms, p50_ms, p99_ms, throughput_mbps}。"""

    # -- B 覆寫：offload 到 host/SSD ---------------------------------------- #
    def offload(self, path: str) -> None:
        """把表離到 host/SSD（mmap / NO_BUFFERING）。A 的 fake 版只記路徑；B 覆寫做真實離載。"""
        raise NotImplementedError("B 應覆寫 offload() 做真實離載")

    def prefetch(self, token_ids: list[TokenId]) -> None:
        """非同步預取下一 token 的 16 行（與計算重疊）。B 覆寫做真實非同步讀。"""
        raise NotImplementedError("B 應覆寫 prefetch() 做真實非同步預取")


# --------------------------------------------------------------------------- #
# 組裝（B 用）
# --------------------------------------------------------------------------- #
@dataclass
class Stack:
    """A/B 共用的物件組合：engine + cache + ngram。"""
    engine: InferenceEngine
    cache: CacheManager
    ngram: NgramTable
    cfg: dict[str, Any] = field(default_factory=dict)
