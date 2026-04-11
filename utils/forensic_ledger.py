"""
Blockchain Forensic Ledger — SHA-256 Tamper-Proof Media Provenance

MY DESIGN ADDITION: Immutable forensic chain for media verification.

Each analyzed media file gets a Block containing:
  - SHA-256 hash of the raw media bytes
  - SHA-256 hash of the detection result
  - Timestamp
  - Previous block hash (chaining)

Any post-analysis tampering of the media changes its hash, breaking the chain.
This provides court-admissible forensic evidence of what was analyzed and when.

Storage: local JSON ledger (swap out the _persist / _load calls to use
         IPFS, Ethereum, or a DB for production deployments).
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path


# ─────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────

@dataclass
class Block:
    index:           int
    timestamp:       float
    media_hash:      str        # SHA-256 of raw media bytes
    result_hash:     str        # SHA-256 of JSON detection result
    verdict:         str        # "REAL" | "FAKE" | "UNCERTAIN"
    confidence:      float      # 0.0 – 1.0
    module_scores:   Dict[str, float] = field(default_factory=dict)
    metadata:        Dict[str, Any]   = field(default_factory=dict)
    prev_hash:       str = "0" * 64
    block_hash:      str = ""

    def compute_hash(self) -> str:
        """Deterministically hash this block's contents."""
        payload = json.dumps({
            'index':        self.index,
            'timestamp':    self.timestamp,
            'media_hash':   self.media_hash,
            'result_hash':  self.result_hash,
            'verdict':      self.verdict,
            'confidence':   self.confidence,
            'prev_hash':    self.prev_hash,
        }, sort_keys=True).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()

    def seal(self):
        """Compute and store this block's hash. Call once after creation."""
        self.block_hash = self.compute_hash()

    def verify(self) -> bool:
        """Returns True if block_hash matches current content (not tampered)."""
        return self.block_hash == self.compute_hash()


# ─────────────────────────────────────────────────────────────────
# Media hashing utilities
# ─────────────────────────────────────────────────────────────────

def hash_file(path: str, chunk_size: int = 65536) -> str:
    """
    Compute SHA-256 of a file's raw bytes.
    Streams in chunks to handle large video files without loading all into RAM.
    """
    sha = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest()


def hash_bytes(data: bytes) -> str:
    """SHA-256 of an arbitrary byte string."""
    return hashlib.sha256(data).hexdigest()


def hash_result(result: dict) -> str:
    """Deterministic hash of a detection result dict."""
    serialized = json.dumps(result, sort_keys=True, default=str).encode('utf-8')
    return hashlib.sha256(serialized).hexdigest()


# ─────────────────────────────────────────────────────────────────
# Blockchain ledger
# ─────────────────────────────────────────────────────────────────

class ForensicLedger:
    """
    Local append-only blockchain for deepfake detection records.

    Each call to record_analysis() adds a new block linking back to
    the previous one. The chain can be verified at any point via verify_chain().

    Usage:
        ledger = ForensicLedger(storage_path='./ledger.json')
        block  = ledger.record_analysis(
            media_path='video.mp4',
            result=detection_result,
            verdict='FAKE',
            confidence=0.91,
        )
        print(block.block_hash)  # forensic receipt
        ledger.verify_chain()    # True = untampered
    """

    GENESIS_HASH = "0" * 64

    def __init__(self, storage_path: str = './deepshield_ledger.json'):
        self.storage_path = Path(storage_path)
        self.chain: List[Block] = []
        self._load()

    # ── Chain operations ─────────────────────────────────────────

    def _genesis(self) -> Block:
        """Create the genesis (first) block."""
        genesis = Block(
            index=0,
            timestamp=time.time(),
            media_hash=self.GENESIS_HASH,
            result_hash=self.GENESIS_HASH,
            verdict='GENESIS',
            confidence=1.0,
            prev_hash=self.GENESIS_HASH,
            metadata={'note': 'DeepShield genesis block'},
        )
        genesis.seal()
        return genesis

    @property
    def last_hash(self) -> str:
        return self.chain[-1].block_hash if self.chain else self.GENESIS_HASH

    def record_analysis(self,
                        media_path: str,
                        result: dict,
                        verdict: str,
                        confidence: float,
                        module_scores: Optional[Dict[str, float]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Block:
        """
        Add a new block recording the analysis of a media file.

        Args:
            media_path:     Path to the video/audio file that was analyzed.
            result:         Full detection result dict (will be hashed).
            verdict:        'REAL', 'FAKE', or 'UNCERTAIN'.
            confidence:     Detection confidence (0–1).
            module_scores:  Per-module scores (gaze, lip, voice, etc.).
            metadata:       Additional metadata (filename, duration, etc.).

        Returns:
            The newly created and sealed Block.
        """
        m_hash = hash_file(media_path) if os.path.exists(media_path) else hash_bytes(media_path.encode())
        r_hash = hash_result(result)

        block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            media_hash=m_hash,
            result_hash=r_hash,
            verdict=verdict.upper(),
            confidence=float(confidence),
            module_scores=module_scores or {},
            metadata={
                'filename':  os.path.basename(media_path),
                **(metadata or {}),
            },
            prev_hash=self.last_hash,
        )
        block.seal()
        self.chain.append(block)
        self._persist()
        return block

    def record_raw(self,
                   media_bytes: bytes,
                   result: dict,
                   verdict: str,
                   confidence: float,
                   module_scores: Optional[Dict[str, float]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Block:
        """Same as record_analysis but takes raw bytes instead of a path."""
        m_hash = hash_bytes(media_bytes)
        r_hash = hash_result(result)

        block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            media_hash=m_hash,
            result_hash=r_hash,
            verdict=verdict.upper(),
            confidence=float(confidence),
            module_scores=module_scores or {},
            metadata=metadata or {},
            prev_hash=self.last_hash,
        )
        block.seal()
        self.chain.append(block)
        self._persist()
        return block

    # ── Verification ─────────────────────────────────────────────

    def verify_chain(self) -> bool:
        """
        Validate every block in the chain:
          1. Each block's hash matches its recomputed hash (content integrity).
          2. Each block's prev_hash matches the previous block's hash (linkage).
        Returns True if chain is fully intact, False if any tampering detected.
        """
        if not self.chain:
            return True
        # Skip genesis block (index 0)
        for i, block in enumerate(self.chain):
            if not block.verify():
                print(f"[TAMPER] Block {i} content hash mismatch!")
                return False
            if i > 0:
                expected_prev = self.chain[i - 1].block_hash
                if block.prev_hash != expected_prev:
                    print(f"[TAMPER] Block {i} prev_hash mismatch!")
                    return False
        return True

    def verify_media(self, media_path: str) -> Optional[Block]:
        """
        Check whether a given media file matches any block's media_hash.
        Returns the matching Block if found (provenance confirmed), else None.
        """
        if not os.path.exists(media_path):
            return None
        file_hash = hash_file(media_path)
        for block in self.chain:
            if block.media_hash == file_hash:
                return block
        return None

    def get_forensic_report(self, block_index: Optional[int] = None) -> str:
        """
        Generate a human-readable forensic report for a block (or all).
        """
        if block_index is not None:
            blocks = [self.chain[block_index]]
        else:
            blocks = self.chain

        lines = ["=" * 60, "DEEPSHIELD FORENSIC LEDGER REPORT", "=" * 60]
        for b in blocks:
            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(b.timestamp))
            lines += [
                f"\nBlock #{b.index}",
                f"  Timestamp:   {ts}",
                f"  File:        {b.metadata.get('filename', 'N/A')}",
                f"  Media hash:  {b.media_hash}",
                f"  Result hash: {b.result_hash}",
                f"  Verdict:     {b.verdict}",
                f"  Confidence:  {b.confidence:.2%}",
            ]
            if b.module_scores:
                lines.append("  Module scores:")
                for mod, score in b.module_scores.items():
                    lines.append(f"    {mod:20s}: {score:.3f}")
            lines.append(f"  Block hash:  {b.block_hash}")
            lines.append(f"  Prev hash:   {b.prev_hash}")
            lines.append(f"  Integrity:   {'✓ VALID' if b.verify() else '✗ TAMPERED'}")

        chain_ok = self.verify_chain()
        lines += [
            "",
            "=" * 60,
            f"Chain integrity: {'✓ INTACT ({} blocks)'.format(len(self.chain)) if chain_ok else '✗ CHAIN BROKEN'}",
            "=" * 60,
        ]
        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────────

    def _persist(self):
        """Serialize chain to JSON."""
        data = [asdict(block) for block in self.chain]
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load chain from JSON if it exists."""
        if not self.storage_path.exists():
            self.chain = [self._genesis()]
            self._persist()
            return
        with open(self.storage_path, 'r') as f:
            data = json.load(f)
        self.chain = [Block(**d) for d in data]
        if not self.verify_chain():
            print("[WARNING] Loaded ledger failed integrity check. "
                  "It may have been tampered with.")

    def export_json(self, path: str):
        """Export full chain to a JSON file for external audit."""
        data = [asdict(block) for block in self.chain]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Ledger exported to {path} ({len(self.chain)} blocks)")
