"""GET /jobs/{job_id}/dicomdir/download：DICOMDIR 文件集构建。

覆盖：
* 成功路径：DICOMDIR + 稳定短路径（File ID 合规）、四级层次、引用可解析、
  内容为匿名文件重编码且不含原始路径/身份字段、重复构建路径稳定、临时文件清理；
* 404/409：任务不存在 / 未完成 / 无可读输出；
* 409 problems：匿名输出不可读、缺标签、同一匿名 SOP UID 多份输出。
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.crypto import derive_uid
from tests.sample_data import _write_bytes
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.fileset import FileSet, is_conformant_file_id
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

SECRET = "dicomdir-secret-123456"


def _ct_bytes(
    sop_uid: str,
    series_uid: str,
    study_uid: str,
    patient_id: str = "P-1",
    patient_name: str = "Zhang^San",
    modality: str = "CT",
    with_optional: bool = True,
) -> bytes:
    """最小合法 CT DICOM；with_optional=False 时省略 InstanceNumber 等可选标签。"""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.PatientID = patient_id
    ds.PatientName = patient_name
    ds.Modality = modality
    ds.PatientBirthDate = "19800101"
    ds.InstitutionName = "Secret Hospital"
    ds.AccessionNumber = "ACC1"
    ds.StudyID = "STUDY1"
    if with_optional:
        ds.InstanceNumber = int(sop_uid.rsplit(".", 1)[-1])
        ds.SeriesNumber = 1
    return _write_bytes(ds)


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in files.items():
            zf.writestr(name, blob)
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DICOM_ANON_WORK_DIR", str(tmp_path / "jobs"))
    import importlib
    import app.config
    import app.jobs
    import app.main
    importlib.reload(app.config)
    importlib.reload(app.jobs)
    importlib.reload(app.main)
    with TestClient(app.main.app) as c:
        yield c


def _upload(client, files: dict[str, bytes]):
    r = client.post(
        "/jobs",
        files={"file": ("studies.zip", _zip_bytes(files), "application/zip")},
        data={"secret": SECRET, "shift_dates": "false"},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _download(client, job_id: str):
    return client.get(f"/jobs/{job_id}/dicomdir/download")


def _extract_zip(content: bytes, dest: Path) -> FileSet:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        zf.extractall(dest)
    return FileSet(str(dest / "DICOMDIR"))


def _registry():
    from app.main import registry
    return registry


# --------------------------------------------------------------------------- #
# 成功路径
# --------------------------------------------------------------------------- #
def test_dicomdir_builds_four_level_hierarchy_with_short_file_ids(client, tmp_path):
    files = {
        # 患者 A / 检查 1 / 系列 1：两个实例
        "orig/path/a/CT0001.dcm": _ct_bytes("1.1.1", "2.1.1", "3.1", "P-1", "Zhang^San"),
        "orig/path/a/CT0002.dcm": _ct_bytes("1.1.2", "2.1.1", "3.1", "P-1", "Zhang^San"),
        # 患者 B / 检查 2 / 系列 2
        "other/deep/dir/CT0003.dcm": _ct_bytes("1.2.1", "2.2.1", "3.2", "P-2", "Li^Si"),
    }
    job = _upload(client, files)
    r = _download(client, job["id"])
    assert r.status_code == 200, r.text
    assert r.headers["x-dicomdir-file-count"] == "3"

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = zf.namelist()
    assert names[0] == "DICOMDIR"
    # 所有实例路径均为符合 File ID 约束的稳定短路径（无原始路径、无扁平 UID 长名）
    instance_names = [n for n in names if n != "DICOMDIR"]
    assert len(instance_names) == 3
    for n in instance_names:
        parts = n.split("/")
        assert len(parts) <= 8
        assert all(len(p) <= 8 for p in parts)
        assert is_conformant_file_id(Path(n))
        # 不得带入原始路径成分
        assert "orig" not in parts and "other" not in parts and "deep" not in parts
        assert not n.endswith(".dcm")

    fs = _extract_zip(r.content, tmp_path / "fs")
    assert len(fs) == 3
    # 每个叶子记录引用都能在解压目录中解析到实际文件
    for inst in fs:
        assert Path(str(inst.path)).is_file()
        # 引用的就是匿名 SOP UID 对应的实例
        assert inst.ReferencedSOPInstanceUIDInFile == inst.SOPInstanceUID

    # 直接解析 DICOMDIR 记录序列，核对四级层次与匿名 UID
    from pydicom import dcmread
    dd = dcmread(tmp_path / "fs" / "DICOMDIR")
    types = [rec.DirectoryRecordType for rec in dd.DirectoryRecordSequence]
    assert types.count("PATIENT") == 2
    assert types.count("STUDY") == 2
    assert types.count("SERIES") == 2
    assert types.count("IMAGE") == 3

    # PATIENT 记录使用患者伪名（取自匿名输出），且不含原始姓名
    patient_ids = {
        str(rec.PatientID)
        for rec in dd.DirectoryRecordSequence
        if rec.DirectoryRecordType == "PATIENT"
    }
    assert patient_ids == {
        job["patients"][i]["new_patient_id"] for i in range(2)
    }
    blob = r.content
    assert b"Zhang" not in blob and b"Li^Si" not in blob
    assert b"Secret Hospital" not in blob


def test_dicomdir_references_match_zip_entries_and_anonymous_uids(client, tmp_path):
    files = {
        "a1.dcm": _ct_bytes("1.1.1", "2.1.1", "3.1"),
        "a2.dcm": _ct_bytes("1.1.2", "2.1.1", "3.1"),
    }
    job = _upload(client, files)
    r = _download(client, job["id"])
    assert r.status_code == 200

    fs = _extract_zip(r.content, tmp_path / "fs")
    sops = {derive_uid(SECRET, s) for s in ("1.1.1", "1.1.2")}
    assert {inst.SOPInstanceUID for inst in fs} == sops
    # ReferencedFileID 与 ZIP 条目一一对应
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        zip_names = set(zf.namelist())
    for inst in fs:
        rel = "/".join(str(p) for p in inst.ReferencedFileID)
        assert rel in zip_names


def test_dicomdir_file_ids_are_stable_across_builds(client):
    files = {
        "a1.dcm": _ct_bytes("1.1.1", "2.1.1", "3.1", "P-1"),
        "a2.dcm": _ct_bytes("1.1.2", "2.1.1", "3.1", "P-1"),
        "b1.dcm": _ct_bytes("1.2.1", "2.2.1", "3.2", "P-2"),
    }
    job = _upload(client, files)
    r1 = _download(client, job["id"])
    r2 = _download(client, job["id"])
    assert r1.status_code == r2.status_code == 200
    n1 = sorted(zipfile.ZipFile(io.BytesIO(r1.content)).namelist())
    n2 = sorted(zipfile.ZipFile(io.BytesIO(r2.content)).namelist())
    assert n1 == n2


def test_dicomdir_instances_are_anonymized_outputs(client, tmp_path):
    """文件集内实例是匿名输出的内容：伪名/匿名 UID 一致，无 PatientID=原值。"""
    files = {"a1.dcm": _ct_bytes("1.1.1", "2.1.1", "3.1", patient_id="P-1")}
    job = _upload(client, files)
    r = _download(client, job["id"])
    assert r.status_code == 200

    fs = _extract_zip(r.content, tmp_path / "fs")
    from pydicom import dcmread
    for inst in fs:
        ds = dcmread(str(inst.path))
        assert str(ds.PatientID) != "P-1"
        assert str(ds.PatientID) == job["patients"][0]["new_patient_id"]
        assert str(ds.SOPInstanceUID) == derive_uid(SECRET, "1.1.1")
        assert str(ds.StudyInstanceUID) == derive_uid(SECRET, "3.1")
        # 机构身份标签已被匿名化清空（元素保留为空值）
        assert not str(getattr(ds, "InstitutionName", "") or "").strip()
        # 文件元信息 SOP UID 与叶子记录一致
        assert str(ds.file_meta.MediaStorageSOPInstanceUID) == inst.SOPInstanceUID


def test_dicomdir_works_without_optional_instance_tags(client, tmp_path):
    """缺 InstanceNumber/SeriesNumber（叶子类型可选键）不阻断构建。"""
    files = {"a1.dcm": _ct_bytes("1.1.1", "2.1.1", "3.1", with_optional=False)}
    job = _upload(client, files)
    r = _download(client, job["id"])
    assert r.status_code == 200, r.text
    fs = _extract_zip(r.content, tmp_path / "fs")
    assert len(fs) == 1


def test_dicomdir_temp_zip_cleaned_after_response(client):
    files = {"a1.dcm": _ct_bytes("1.1.1", "2.1.1", "3.1")}
    job = _upload(client, files)
    r = _download(client, job["id"])
    assert r.status_code == 200
    job_obj = _registry().get(job["id"])
    base = os.path.dirname(job_obj.result.output_dir)
    leftovers = [
        n for n in os.listdir(base)
        if n.startswith("dicomdir-") and n.endswith(".zip")
    ]
    assert leftovers == []


def test_dicomdir_content_disposition_filename(client):
    files = {"a1.dcm": _ct_bytes("1.1.1", "2.1.1", "3.1")}
    job = _upload(client, files)
    r = _download(client, job["id"])
    assert r.status_code == 200
    cd = r.headers["content-disposition"]
    assert cd.startswith('attachment; filename="dicomdir-')
    assert cd.endswith('.zip"')


# --------------------------------------------------------------------------- #
# 错误路径
# --------------------------------------------------------------------------- #
def test_dicomdir_job_not_found_404(client):
    r = _download(client, "deadbeef")
    assert r.status_code == 404


def test_dicomdir_failed_job_409(client):
    r = client.post(
        "/jobs",
        files={"file": ("broken.zip", b"PK\x03\x04 garbage", "application/zip")},
        data={"secret": SECRET},
    )
    job = r.json()
    assert job["status"] == "failed"
    resp = _download(client, job["id"])
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "JOB_NOT_READY"


def test_dicomdir_no_usable_output_409(client):
    # 全部文件非法：任务 ready（引擎层面把文件标为 error），但无成功输出
    files = {
        "notes/not-a-dicom.txt": b"hello, this is not dicom",
        "broken/bad.dcm": b"\x00\x01\x02not dicom at all\xff\xfe",
    }
    job = _upload(client, files)
    assert job["output_file_count"] == 0
    r = _download(client, job["id"])
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "DICOMDIR_NO_USABLE_OUTPUT"


def test_dicomdir_unreadable_output_409(client):
    files = {
        "ok.dcm": _ct_bytes("1.1.1", "2.1.1", "3.1"),
        "bad.dcm": _ct_bytes("1.1.2", "2.1.1", "3.1"),
    }
    job = _upload(client, files)
    assert job["output_file_count"] == 2

    # 破坏其中一份匿名输出
    job_obj = _registry().get(job["id"])
    targets = [
        os.path.join(job_obj.result.output_dir, f["output_name"])
        for f in job["files"]
        if derive_uid(SECRET, "1.1.2") == f["sop_instance_uid"]
    ]
    assert len(targets) == 1
    with open(targets[0], "wb") as fh:
        fh.write(b"corrupted-not-dicom")

    r = _download(client, job["id"])
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["code"] == "DICOMDIR_BUILD_INVALID"
    assert len(detail["problems"]) == 1
    problem = detail["problems"][0]
    assert problem["reason"] == "unreadable"
    assert problem["sop_instance_uid"] == derive_uid(SECRET, "1.1.2")
    assert problem["output_file"].endswith(".dcm")
    assert problem["message"]


def test_dicomdir_missing_required_tag_409(client):
    files = {"a1.dcm": _ct_bytes("1.1.1", "2.1.1", "3.1")}
    job = _upload(client, files)

    # 从匿名输出中删除 Modality（SERIES 记录必需）
    job_obj = _registry().get(job["id"])
    from pydicom import dcmread
    report = job_obj.result.files[0]
    path = os.path.join(job_obj.result.output_dir, report.output_name)
    ds = dcmread(path)
    del ds.Modality
    ds.save_as(path, enforce_file_format=True)

    r = _download(client, job["id"])
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["code"] == "DICOMDIR_BUILD_INVALID"
    problem = detail["problems"][0]
    assert problem["reason"] == "missing_required_tag"
    assert any("Modality" in t for t in problem["missing_tags"])
    assert problem["sop_instance_uid"] == derive_uid(SECRET, "1.1.1")
    assert problem["output_file"] == report.output_name


def test_dicomdir_ambiguous_sop_409(client):
    dup_sop = "1.5.5.777"
    files = {
        "dup_a.dcm": _ct_bytes(dup_sop, "2.5.5", "3.55"),
        "dup_b.dcm": _ct_bytes(dup_sop, "2.5.5", "3.55"),
        "other.dcm": _ct_bytes("1.6.6.1", "2.6.6", "3.66"),
    }
    job = _upload(client, files)
    assert job["output_file_count"] == 3

    r = _download(client, job["id"])
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["code"] == "DICOMDIR_BUILD_INVALID"
    problem = next(
        p for p in detail["problems"]
        if p["sop_instance_uid"] == derive_uid(SECRET, dup_sop)
    )
    assert problem["reason"] == "ambiguous_sop"
    new_sop = derive_uid(SECRET, dup_sop)
    assert problem["output_files"] == sorted(
        [f"{new_sop}.dcm", f"{new_sop}-2.dcm"]
    )

    # 全量下载行为不受影响
    full = client.get(f"/jobs/{job['id']}/download")
    assert full.status_code == 200
    assert len(zipfile.ZipFile(io.BytesIO(full.content)).namelist()) == 3


def test_dicomdir_collects_multiple_problems_at_once(client):
    """多个文件同时出问题时，一次性在 problems[] 中全部列明。"""
    files = {
        "a1.dcm": _ct_bytes("1.1.1", "2.1.1", "3.1"),
        "a2.dcm": _ct_bytes("1.1.2", "2.1.1", "3.1"),
        "a3.dcm": _ct_bytes("1.1.3", "2.1.1", "3.1"),
    }
    job = _upload(client, files)
    job_obj = _registry().get(job["id"])
    by_sop = {f["sop_instance_uid"]: f["output_name"] for f in job["files"]}

    # a2：破坏内容；a3：删除 Modality
    p2 = os.path.join(job_obj.result.output_dir,
                      by_sop[derive_uid(SECRET, "1.1.2")])
    with open(p2, "wb") as fh:
        fh.write(b"deadbeef")
    p3 = os.path.join(job_obj.result.output_dir,
                      by_sop[derive_uid(SECRET, "1.1.3")])
    from pydicom import dcmread
    ds3 = dcmread(p3)
    del ds3.Modality
    ds3.save_as(p3, enforce_file_format=True)

    r = _download(client, job["id"])
    assert r.status_code == 409
    problems = r.json()["detail"]["problems"]
    reasons = {(p["sop_instance_uid"]): p["reason"] for p in problems}
    assert reasons[derive_uid(SECRET, "1.1.2")] == "unreadable"
    assert reasons[derive_uid(SECRET, "1.1.3")] == "missing_required_tag"
    # 正常文件不出现在问题列表中
    assert derive_uid(SECRET, "1.1.1") not in reasons
