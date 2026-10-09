"""Assemble compiler artifacts and their precomputed content identity.

The digest describes installed content, including native compiler/binding
libraries. It is computed once by Bazel, never when importing a model.
"""

import argparse
import hashlib
import json
import shutil
import tomllib
from pathlib import Path


def content_identity(package):
    digest = hashlib.sha256()
    for path in sorted((package / "iree").rglob("*")):
        if not path.is_file() or path.name == "_build_provenance.json":
            continue
        relative = path.relative_to(package).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def assemble(manifest, metadata, output):
    output.mkdir(parents=True, exist_ok=True)
    destinations = set()
    for entry in json.loads(manifest.read_text()):
        relative = Path(entry["destination"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid package destination: {relative}")
        if relative in destinations:
            raise ValueError(f"Duplicate package destination: {relative}")
        destinations.add(relative)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry["source"], target)
        # Preserve executability while normalizing modes for stable packaging.
        target.chmod(0o755 if Path(entry["source"]).stat().st_mode & 0o111 else 0o644)
    project = tomllib.loads(metadata.read_text())["project"]
    # Match the version module provided by IREE's standard compiler package.
    # Mark the revision as a content identity rather than claiming a release
    # commit for a build that may include local dependency patches.
    (output / "iree/compiler/version.py").write_text(
        "from json import loads as _loads\n"
        "from pathlib import Path as _Path\n"
        f"VERSION = {project['version']!r}\n"
        'PACKAGE_SUFFIX = ""\n'
        'REVISIONS = {"IREE": "source-build:" + _loads(\n'
        '    _Path(__file__).with_name("_build_provenance.json").read_text()\n'
        ')["build_id"]}\n'
    )
    provenance = {"schema_version": 1, "build_id": content_identity(output)}
    (output / "iree/compiler/_build_provenance.json").write_text(
        json.dumps(provenance, sort_keys=True) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.manifest, args.metadata, args.output)


if __name__ == "__main__":
    main()
