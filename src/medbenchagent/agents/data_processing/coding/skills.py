"""Run-owned copies of explicitly enabled, packaged Coding skills."""
import hashlib
from importlib.resources import files
from pathlib import Path, PurePosixPath

from pydantic import Field, model_validator
from medbenchagent.compilation.contracts import Contract, Identifier

SKILL_ROOT = "resources/coding-skills"


class FrozenSkills(Contract):
    names: list[Identifier] = Field(default_factory=list)
    files: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def identities(self):
        if len(self.names) != len(set(self.names)):
            raise ValueError("Duplicate enabled skill")
        for name, digest in self.files.items():
            path = PurePosixPath(name)
            if (not path.parts or path.parts[0] not in self.names or path.is_absolute()
                    or ".." in path.parts or "\\" in name or ":" in name
                    or path.as_posix() != name or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("Invalid frozen skill resource")
        if any(f"{name}/SKILL.md" not in self.files for name in self.names):
            raise ValueError("Enabled skill is missing SKILL.md")
        return self


def freeze_skills(directory: Path, names: list[str]) -> FrozenSkills:
    source = files(__package__).joinpath("skills")
    available = {entry.name: entry for entry in source.iterdir() if entry.is_dir()}
    if len(names) != len(set(names)) or set(names) - available.keys():
        raise ValueError("Coding skills must select distinct packaged skill names")
    root = directory / SKILL_ROOT
    contents = {}

    def collect(node, prefix):
        for entry in sorted(node.iterdir(), key=lambda p: p.name):
            relative = f"{prefix}/{entry.name}"
            if entry.is_dir():
                collect(entry, relative)
            else:
                contents[relative] = entry.read_bytes()

    for name in names:
        collect(available[name], name)
    snapshot = FrozenSkills(names=names, files={key: hashlib.sha256(value).hexdigest()
                                              for key, value in contents.items()})
    for name, content in contents.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != content:
                raise ValueError("Cannot overwrite a frozen skill resource")
        else:
            with target.open("xb") as stream:
                stream.write(content)
    verify_skills(directory, snapshot)
    return snapshot


def verify_skills(directory: Path, snapshot: FrozenSkills) -> Path:
    directory = Path(directory).resolve()
    root = directory / SKILL_ROOT
    if not root.resolve().is_relative_to(directory):
        raise ValueError("Frozen skill directory escapes run")
    actual = {}
    for path in root.rglob("*"):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Frozen skills cannot contain links or escaping resources")
        if path.is_file():
            actual[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != snapshot.files:
        raise ValueError("Frozen skill resources changed or are missing")
    return root


def load_native_skills(directory: Path, snapshot: FrozenSkills):
    """Only the Coding environment imports the native SDK skill parser."""
    root = verify_skills(directory, snapshot)
    if not snapshot.names:
        return []
    from openhands.sdk.skills import load_skills_from_dir
    repo, knowledge, native = load_skills_from_dir(root, strict=True, root=root)
    if repo or knowledge or set(native) != set(snapshot.names):
        raise ValueError("Native SDK did not load exactly the frozen AgentSkills selection")
    return [native[name] for name in snapshot.names]
