"""
SSH Runner for remote Trn2 execution.
Handles file transfer, cache clearing, compilation, and benchmarking.
"""
import subprocess
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RemoteResult:
    stdout: str = ""
    stderr: str = ""
    returncode: int = -1
    success: bool = False
    duration: float = 0.0


class SSHRunner:
    def __init__(self, host: str, key_path: str = "~/.ssh/id_rsa",
                 remote_dir: str = "~/nki-moe", timeout: int = 1800):
        self.host = host
        self.key_path = str(Path(key_path).expanduser())
        self.remote_dir = remote_dir
        self.timeout = timeout
        self._ssh_base = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-i", self.key_path,
            self.host
        ]
        self._scp_base = [
            "scp", "-o", "StrictHostKeyChecking=no",
            "-i", self.key_path
        ]

    def run_remote(self, cmd: str, timeout: int = None) -> RemoteResult:
        """Execute command on Trn2 via SSH"""
        t = timeout or self.timeout
        full_cmd = self._ssh_base + [f"cd {self.remote_dir} && {cmd}"]
        start = time.time()
        try:
            result = subprocess.run(
                full_cmd, capture_output=True, text=True, timeout=t
            )
            elapsed = time.time() - start
            return RemoteResult(
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
                success=result.returncode == 0,
                duration=elapsed
            )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            logger.error(f"SSH command timed out after {t}s: {cmd[:100]}...")
            return RemoteResult(stderr=f"Timeout after {t}s", duration=elapsed)
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"SSH error: {e}")
            return RemoteResult(stderr=str(e), duration=elapsed)

    def upload_file(self, local_path: str, remote_path: str = None) -> bool:
        """SCP file to Trn2"""
        remote = remote_path or f"{self.remote_dir}/{Path(local_path).name}"
        cmd = self._scp_base + [local_path, f"{self.host}:{remote}"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                logger.error(f"SCP failed: {result.stderr}")
            return result.returncode == 0
        except Exception as e:
            logger.error(f"SCP error: {e}")
            return False

    def download_file(self, remote_path: str, local_path: str) -> bool:
        """SCP file from Trn2"""
        cmd = self._scp_base + [f"{self.host}:{remote_path}", local_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"SCP download error: {e}")
            return False

    def clear_compile_cache(self) -> bool:
        """Clear Neuron compile cache — required when NKI kernel changes"""
        result = self.run_remote(
            "rm -rf /var/tmp/neuron-compile-cache/* 2>/dev/null; "
            "echo 'cache cleared'",
            timeout=30
        )
        return result.success

    def clear_compiled_model(self, compiled_path: str) -> bool:
        """Remove traced model to force recompilation"""
        result = self.run_remote(f"rm -rf {compiled_path} 2>/dev/null; echo 'model cleared'", timeout=30)
        return result.success

    def check_connection(self) -> bool:
        """Test SSH connectivity"""
        result = self.run_remote("echo 'connected'", timeout=10)
        return result.success and "connected" in result.stdout

    def run_benchmark(self, model_path: str, compiled_path: str,
                      enable_nki: bool = True) -> RemoteResult:
        """Run main.py --mode benchmark"""
        nki_flag = "--enable-nki" if enable_nki else ""
        cmd = (
            f"python3 main.py --mode benchmark {nki_flag} "
            f"--model-path {model_path} "
            f"--compiled-model-path {compiled_path}"
        )
        return self.run_remote(cmd)

    def run_generate(self, model_path: str, compiled_path: str,
                     prompt: str = "What is the capital of France?",
                     enable_nki: bool = True) -> RemoteResult:
        """Run main.py --mode generate for accuracy check"""
        nki_flag = "--enable-nki" if enable_nki else ""
        cmd = (
            f"python3 main.py --mode generate {nki_flag} "
            f"--model-path {model_path} "
            f"--compiled-model-path {compiled_path} "
            f'--prompt "{prompt}"'
        )
        return self.run_remote(cmd)
