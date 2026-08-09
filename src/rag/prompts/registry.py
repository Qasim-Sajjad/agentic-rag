"""Prompt registry. No prompt string appears inline anywhere in the codebase.

Two payoffs from hashing the file content: a prompt change produces a new
config hash in the eval harness so the regression suite runs, and a wrong
answer in production can be traced to the exact prompt that produced it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).parent
REGISTRY_FILE = PROMPTS_DIR / "registry.yaml"


class PromptNotFoundError(RuntimeError):
    """No file for that role and version. A deploy time error, not a runtime one."""


@dataclass(frozen=True)
class Prompt:
    role: str
    version: str
    text: str
    content_hash: str

    @property
    def identifier(self) -> str:
        """What goes in a trace step and in the eval config hash."""
        return f"{self.role}/{self.version}"


class PromptRegistry:
    def __init__(self, directory: Path = PROMPTS_DIR) -> None:
        self._directory = directory
        self._active = self._load_registry()

    def _load_registry(self) -> dict[str, str]:
        raw = yaml.safe_load((self._directory / "registry.yaml").read_text("utf-8"))
        return {str(role): str(version) for role, version in (raw or {}).items()}

    def active_version(self, role: str) -> str:
        version = self._active.get(role)
        if version is None:
            raise PromptNotFoundError(f"no active version for role {role!r}")
        return version

    def get(self, role: str, version: str | None = None) -> Prompt:
        resolved = version or self.active_version(role)
        path = self._directory / role / f"{resolved}.md"
        if not path.is_file():
            raise PromptNotFoundError(f"no prompt file at {path}")
        text = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        return Prompt(role=role, version=resolved, text=text, content_hash=digest)

    def versions(self, role: str) -> list[str]:
        """Superseded versions stay in the repo. The diff is the evidence."""
        return sorted(path.stem for path in (self._directory / role).glob("*.md"))


@lru_cache(maxsize=1)
def get_registry() -> PromptRegistry:
    return PromptRegistry()
