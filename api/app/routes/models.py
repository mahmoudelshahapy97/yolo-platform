"""Model registry API: list, inspect, upload a version, deploy (gated), roll back."""

import hashlib
import json
import re
import shutil
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from ..deps import require, state
from ..services.auth import Principal
from ..services.model_registry import GateFailed

router = APIRouter(prefix="/v1/models", tags=["models"])
NAME = re.compile(r"^[a-z0-9_]{1,64}$")


def _check_name(name: str) -> None:
    if not NAME.match(name):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown model '{name}'")


@router.get("")
async def list_models(request: Request, p: Principal = Depends(require("infer"))):
    core = state(request)
    versions = await core.registry.list_versions()
    out = {}
    for v in versions:
        if not p.may_use(v["model_name"]):
            continue
        m = out.setdefault(v["model_name"], {"name": v["model_name"], "task": v["task"], "active_version": None,
                                             "versions": []})
        if v["status"] == "active":
            m["active_version"] = v["version"]
        m["versions"].append({"version": v["version"], "status": v["status"]})
    return {"models": list(out.values())}


@router.get("/{name}")
async def get_model(name: str, request: Request, p: Principal = Depends(require("infer"))):
    _check_name(name)
    if not p.may_use(name):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown model '{name}'")
    core = state(request)
    versions = await core.registry.list_versions(name)
    if not versions:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown model '{name}'")
    active = await core.registry.active(name)
    names = active.names if active else {}
    return {
        "name": name,
        "task": versions[-1]["task"],
        "imgsz": versions[-1]["imgsz"],
        "max_batch": versions[-1]["max_batch"],
        "active_version": active.version if active else None,
        "triton_model": active.serve_name if active else None,
        "ready": await core.triton.model_ready(active.serve_name) if active else False,
        "classes": names,
        "versions": versions,
    }


@router.post("/{name}/versions/{version}", status_code=201)
async def upload_version(
    name: str, version: int, request: Request,
    plan: UploadFile = File(..., description="TensorRT engine built by inference/export/build_plan.sh"),
    golden: UploadFile = File(..., description="golden.npz from inference/validation/parity.py --write-golden"),
    meta: Optional[UploadFile] = File(None, description="model_meta.json if classes or sizes changed"),
    p: Principal = Depends(require("admin")),
):
    """Place a new version in the model repository and register it (it does not serve until deployed)."""
    _check_name(name)
    core = state(request)
    model_dir = core.settings.model_repository / name
    if not (model_dir / "config.pbtxt").is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown model '{name}'; new models go through inference/export")
    target = model_dir / str(version)
    if target.exists():
        raise HTTPException(status.HTTP_409_CONFLICT, f"{name} v{version} already exists; versions are immutable")
    tmp = model_dir / f".upload-{version}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        digest = hashlib.sha256()
        with (tmp / "model.plan").open("wb") as out:
            while chunk := await plan.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        (tmp / "model.plan.sha256").write_text(digest.hexdigest() + "\n")
        with (tmp / "golden.npz").open("wb") as out:
            while chunk := await golden.read(1 << 20):
                out.write(chunk)
        if meta is not None:
            parsed = json.loads(await meta.read())
            for k in ("task", "imgsz", "max_batch", "names"):
                if k not in parsed:
                    raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"model_meta.json lacks '{k}'")
            (tmp / "model_meta.json").write_text(json.dumps(parsed))
        tmp.rename(target)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    await core.registry.sync()
    return {"model": name, "version": version, "status": "registered", "sha256": digest.hexdigest()}


@router.post("/{name}/versions/{version}/deploy")
async def deploy_version(name: str, version: int, request: Request, p: Principal = Depends(require("admin"))):
    """Load the version next to the active one, run the gate, switch traffic, then drain the old one."""
    _check_name(name)
    core = state(request)
    await core.registry.sync()
    try:
        return await core.registry.activate(name, version, reason=f"deploy by key {p.key_id}")
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    except GateFailed as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, {"message": "gate failed; not deployed", "gate": exc.report})


@router.post("/{name}/rollback")
async def rollback(name: str, request: Request, p: Principal = Depends(require("admin"))):
    _check_name(name)
    try:
        return await state(request).registry.rollback(name)
    except LookupError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc))
    except GateFailed as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, {"message": "gate failed", "gate": exc.report})
