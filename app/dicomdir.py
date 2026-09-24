"""为已完成任务生成供离线阅片软件浏览的 DICOMDIR 文件集 ZIP。

数据来源严格限定为“已完成任务 output_dir 中的匿名 DICOM 文件”，
**不重新解压或处理原始 ZIP**：

1. 从匿名输出中读取患者伪名（PatientID/PatientName）与匿名
   Study/Series/SOP UID，重建 PATIENT → STUDY → SERIES → 实例（叶子记录）
   四级目录层次；
2. 实例通过 pydicom FileSet 映射为**符合 DICOM File ID 约束**（每段 ≤8 字符、
   至多 8 段、仅 0-9/A-Z/_）的稳定短路径，如
   ``PT000000/ST000000/SE000000/IM000000``；映射键为
   （伪名, Study/Series/SOP UID）排序序号，同一批匿名输出重复构建结果一致；
3. DICOMDIR 叶子记录的 ReferencedFileID 由 FileSet 写入并**逐个核对**，
   必须指向 ZIP 内实际打包的文件；目录记录只引用匿名文件中的标签，
   不写入原始路径、原始文件名或任何身份字段。

任一匿名文件不可读、缺少建目录必需标签，或同一匿名 SOP UID 对应多份输出时，
收集全部问题后抛出 DicomDirError（路由层映射为 409），**不产生残缺目录包**。
"""
from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset
from pydicom.fileset import (
    DIRECTORY_RECORDERS,
    FileSet,
    RecordNode,
    _four_level_record_type,
    is_conformant_file_id,
)

from .anonymizer import FileReport, JobResult
from .study_subset import _successful_outputs

# 问题原因（detail.problems[].reason）
REASON_UNREADABLE = "unreadable"
REASON_MISSING_TAG = "missing_required_tag"
REASON_AMBIGUOUS_SOP = "ambiguous_sop"

# 建目录必需的标签：层级定位（PATIENT/STUDY/SERIES/实例）与叶子记录引用
# 所依赖的标签。叶子记录类型专属的说明性标签（如 IMAGE 的 InstanceNumber、
# RT STRUCTURE SET 的 StructureSetLabel）多为 Type 2 可空/可选，缺失时不阻断
# 建目录，只在记录中省略。
# PatientID/PatientName 由匿名引擎保证写入伪名，这里仍做防御性非空检查。
REQUIRED_TOP_LEVEL_TAGS = (
    "PatientID",
    "PatientName",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "SOPClassUID",
    "Modality",
)
REQUIRED_FILE_META_TAGS = ("MediaStorageSOPClassUID", "TransferSyntaxUID")

# DICOM 标签号与名称，用于错误信息
_TAG_DESCRIPTIONS = {
    "PatientID": "(0010,0020) PatientID",
    "PatientName": "(0010,0010) PatientName",
    "StudyInstanceUID": "(0020,000D) StudyInstanceUID",
    "SeriesInstanceUID": "(0020,000E) SeriesInstanceUID",
    "SOPInstanceUID": "(0008,0018) SOPInstanceUID",
    "SOPClassUID": "(0008,0016) SOPClassUID",
    "Modality": "(0008,0060) Modality",
    "MediaStorageSOPClassUID": "(0002,0002) MediaStorageSOPClassUID",
    "TransferSyntaxUID": "(0002,0010) TransferSyntaxUID",
}


class DicomDirError(Exception):
    """DICOMDIR 文件集构建失败；http_status 由路由层映射为 HTTPException。"""

    def __init__(
        self,
        code: str,
        http_status: int,
        message: str,
        problems: list[dict] | None = None,
        **extra,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.detail: dict = {"code": code, "message": message}
        if problems:
            self.detail["problems"] = problems
        self.detail.update(extra)


@dataclass
class _Instance:
    """通过校验、待加入 FileSet 的一个匿名输出实例。"""

    report: FileReport
    dataset: Dataset

    @property
    def output_name(self) -> str:
        return self.report.output_name


def _output_path(result: JobResult, report: FileReport) -> str:
    return os.path.join(result.output_dir, report.output_name)


# --------------------------------------------------------------------------- #
# 校验：逐文件读取 + 必需标签 + SOP 唯一性
# --------------------------------------------------------------------------- #
def _is_empty(container, keyword: str) -> bool:
    """元素缺失 / 零 VM / 空串均视为空。"""
    if keyword not in container:
        return True
    elem = container[keyword]
    if not hasattr(elem, "VM"):  # 防御：原始/非标准取值
        return elem is None or not str(elem).strip()
    return elem.VM == 0 or not str(elem.value or "").strip()


def _validate_outputs(result: JobResult) -> list[_Instance]:
    """读取并校验全部成功匿名输出，返回实例列表；收集到任何问题即抛错。"""
    reports = _successful_outputs(result)
    if not reports:
        raise DicomDirError(
            "DICOMDIR_NO_USABLE_OUTPUT",
            409,
            "任务没有任何可读的成功匿名输出，无法构建 DICOMDIR 文件集",
        )

    problems: list[dict] = []
    instances: list[_Instance] = []
    sop_buckets: dict[str, list[FileReport]] = {}

    for report in reports:
        path = _output_path(result, report)
        try:
            ds = pydicom.dcmread(path)
            missing: list[str] = []
            for keyword in REQUIRED_TOP_LEVEL_TAGS:
                if _is_empty(ds, keyword):
                    missing.append(_TAG_DESCRIPTIONS[keyword])
            file_meta = getattr(ds, "file_meta", None)
            for keyword in REQUIRED_FILE_META_TAGS:
                if file_meta is None or _is_empty(file_meta, keyword):
                    missing.append(_TAG_DESCRIPTIONS[keyword])
        except Exception as exc:  # 截断 / 编码损坏 / 标签无法解析：逐文件列明
            problems.append({
                "output_file": report.output_name,
                "sop_instance_uid": report.sop_instance_uid or "",
                "reason": REASON_UNREADABLE,
                "missing_tags": [],
                "message": f"匿名输出文件无法读取：{exc}",
            })
            continue

        if missing:
            problems.append({
                "output_file": report.output_name,
                "sop_instance_uid": report.sop_instance_uid or "",
                "reason": REASON_MISSING_TAG,
                "missing_tags": missing,
                "message": "匿名输出缺少构建 DICOMDIR 必需的标签：" + "、".join(missing),
            })
            continue

        inst = _Instance(report=report, dataset=ds)
        instances.append(inst)
        sop_buckets.setdefault(str(ds.SOPInstanceUID).strip(), []).append(report)

    # 同一匿名 SOP UID 对应多份输出（输入 ZIP 内重复实例的改名输出）：
    # 无法唯一决定目录记录指向哪一份，拒绝构建
    for sop_uid in sorted(sop_buckets):
        bucket = sop_buckets[sop_uid]
        if len(bucket) > 1:
            problems.append({
                "output_file": "",
                "sop_instance_uid": sop_uid,
                "reason": REASON_AMBIGUOUS_SOP,
                "missing_tags": [],
                "output_files": sorted(r.output_name for r in bucket),
                "message": (
                    "同一匿名 SOPInstanceUID 对应多份输出文件，"
                    "无法唯一确定 DICOMDIR 记录引用"
                ),
            })

    if problems:
        problems.sort(key=lambda p: (p["reason"], p["output_file"],
                                     p["sop_instance_uid"]))
        raise DicomDirError(
            "DICOMDIR_BUILD_INVALID",
            409,
            "匿名输出存在不可读文件、缺少建目录必需标签或 SOP UID 歧义，"
            "已拒绝输出 DICOMDIR 文件集 ZIP",
            problems=problems,
        )

    return instances


# --------------------------------------------------------------------------- #
# 目录记录与 FileSet 构建
# --------------------------------------------------------------------------- #
def _build_leaf_record(ds: Dataset) -> tuple[str, Dataset]:
    """构建叶子目录记录（类型由 SOP Class 决定）。

    优先使用 pydicom 的标准记录构建器（带完整的类型专属说明字段）；构建器
    按其内置严格校验拒绝时（如 InstanceNumber 等 Type 2 键缺失），回退为
    仅含通用可选键的最小叶子记录——这些键不参与层级定位，缺失不影响目录浏览。
    """
    leaf_type = _four_level_record_type(ds)
    try:
        leaf = DIRECTORY_RECORDERS[leaf_type](ds)
    except ValueError:
        leaf = Dataset()
        if "InstanceNumber" in ds and ds["InstanceNumber"].VM:
            leaf.InstanceNumber = ds.InstanceNumber
    leaf.DirectoryRecordType = leaf_type
    # add_custom 会统一补齐 ReferencedFileID / ReferencedSOP*InFile；
    # 这里显式赋值保证 RecordNode 校验顺序稳定
    leaf.ReferencedFileID = None
    leaf.ReferencedSOPClassUIDInFile = ds.SOPClassUID
    leaf.ReferencedSOPInstanceUIDInFile = ds.SOPInstanceUID
    leaf.ReferencedTransferSyntaxUIDInFile = ds.file_meta.TransferSyntaxUID
    return leaf_type, leaf


def _build_records(ds: Dataset) -> RecordNode:
    """按匿名数据集构建 PATIENT/STUDY/SERIES/叶子四级目录记录链。

    上层三级手工构建（匿名化会清空 StudyID 等标签，pydicom 默认 STUDY
    构建器强制非空会误报）；所有字段均取自匿名输出，不引入原始路径或身份字段。
    """
    _, leaf = _build_leaf_record(ds)

    series_rec = Dataset()
    series_rec.DirectoryRecordType = "SERIES"
    series_rec.Modality = ds.Modality
    series_rec.SeriesInstanceUID = ds.SeriesInstanceUID
    if "SpecificCharacterSet" in ds:
        series_rec.SpecificCharacterSet = ds.SpecificCharacterSet
    # SeriesNumber 为可选（Type 2），有值才写入
    if "SeriesNumber" in ds and ds["SeriesNumber"].VM:
        series_rec.SeriesNumber = ds.SeriesNumber

    study_rec = Dataset()
    study_rec.DirectoryRecordType = "STUDY"
    study_rec.StudyInstanceUID = ds.StudyInstanceUID
    if "SpecificCharacterSet" in ds:
        study_rec.SpecificCharacterSet = ds.SpecificCharacterSet
    # StudyDate/StudyTime/StudyID/AccessionNumber/StudyDescription 均为
    # Type 2/3 可选；匿名输出中已平移/清空/删除：存在即复制（保留空值元素），
    # 已删除的不补造
    for keyword in ("StudyDate", "StudyTime", "StudyID",
                    "AccessionNumber", "StudyDescription"):
        if keyword in ds:
            setattr(study_rec, keyword, ds[keyword].value)

    patient_rec = Dataset()
    patient_rec.DirectoryRecordType = "PATIENT"
    patient_rec.PatientName = ds.PatientName
    patient_rec.PatientID = ds.PatientID
    if "SpecificCharacterSet" in ds:
        patient_rec.SpecificCharacterSet = ds.SpecificCharacterSet

    nodes = [
        RecordNode(patient_rec),
        RecordNode(study_rec),
        RecordNode(series_rec),
        RecordNode(leaf),
    ]
    for parent, child in zip(nodes, nodes[1:]):
        child.parent = parent
    return nodes[-1]


def build_dicomdir_fileset(instances: list[_Instance], dest_dir: str) -> FileSet:
    """把实例按稳定顺序加入 FileSet 并写出到 dest_dir，返回 FileSet。"""
    # 稳定排序决定 File ID 序号：同一批匿名输出重复构建路径完全一致
    instances = sorted(
        instances,
        key=lambda it: (
            str(it.dataset.PatientID),
            str(it.dataset.StudyInstanceUID),
            str(it.dataset.SeriesInstanceUID),
            str(it.dataset.SOPInstanceUID),
            it.output_name,
        ),
    )
    fileset = FileSet()
    for inst in instances:
        leaf = _build_records(inst.dataset)
        fileset.add_custom(inst.dataset, leaf)
    fileset.write(dest_dir)
    return fileset


def _expected_zip_entries(fileset: FileSet) -> dict[str, str]:
    """返回 {ZIP 内路径（/ 分隔）: 磁盘绝对路径}，并核对 File ID 合规性。"""
    entries: dict[str, str] = {}
    root = Path(fileset.root_path)
    for instance in fileset:
        file_id = [str(part) for part in instance.ReferencedFileID]
        if not file_id or not is_conformant_file_id(Path(*file_id)):
            raise DicomDirError(
                "DICOMDIR_FILE_ID_INVALID",
                500,
                f"生成的 DICOM File ID 不符合规范：{file_id}",
            )
        rel = "/".join(file_id)
        disk_path = root.joinpath(*file_id)
        if not disk_path.is_file():
            raise DicomDirError(
                "DICOMDIR_REFERENCED_FILE_MISSING",
                500,
                f"DICOMDIR 引用的文件在文件集中不存在：{rel}",
            )
        entries[rel] = str(disk_path)
    return entries


def build_dicomdir_zip(fileset: FileSet, dest_zip: str) -> int:
    """将文件集根目录（含 DICOMDIR 与短路径实例）打包，返回实例文件数。"""
    entries = _expected_zip_entries(fileset)
    root = Path(fileset.root_path)
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(str(root / "DICOMDIR"), arcname="DICOMDIR")
        for rel in sorted(entries):
            zf.write(entries[rel], arcname=rel)
    return len(entries)


def write_dicomdir_zip(job, work_dir: str) -> tuple[str, int]:
    """构建 DICOMDIR 文件集 ZIP 到工作目录临时文件，返回 (路径, 实例数)。"""
    result: JobResult = job.result
    instances = _validate_outputs(result)

    fd, tmp_zip = tempfile.mkstemp(
        prefix=f"dicomdir-{job.id[:8]}-",
        suffix=".zip",
        dir=work_dir,
    )
    os.close(fd)
    staging = tempfile.mkdtemp(
        prefix=f"dicomdir-fs-{job.id[:8]}-",
        dir=work_dir,
    )
    try:
        fileset = build_dicomdir_fileset(instances, staging)
        count = build_dicomdir_zip(fileset, tmp_zip)
    except Exception:
        try:
            os.unlink(tmp_zip)
        except OSError:
            pass
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return tmp_zip, count
