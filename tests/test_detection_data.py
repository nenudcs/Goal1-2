"""Run only inside the competition container: python -m unittest discover -s tests."""
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from openpyxl import Workbook

from src.detection.common import require_container
from src.detection.data import (_components, _make_splits, _pair_rows, _table, _text_id,
                                discover_cases, load_sequence)
from src.data.paths import PathResolver


def setUpModule():
    require_container()


class DataTests(unittest.TestCase):
    def test_numeric_excel_ids_rejected_and_text_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.xlsx"
            book = Workbook()
            book.active.append(["AccessionNumber", "SeriesUid"])
            book.active.append([123456789012345678, "0002"])
            book.save(path)
            row = _table(path)[0]
            with self.assertRaises(ValueError):
                _text_id(row["accessionnumber"], "test")
            self.assertEqual(_text_id(row["seriesuid"], "test"), "0002")

    def test_blank_pair_label_remains_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.csv"
            path.write_text("src_img,desc_img,label\nA,B,\nA,C,0\nB,C,1\n")
            self.assertEqual(_pair_rows(path, 1), [{"a": "A", "b": "C", "label": 0}, {"a": "B", "b": "C", "label": 1}])
            path.write_text("src_img,desc_img\nA,B\n")
            self.assertEqual(_pair_rows(path, 1)[0]["label"], 1)

    def test_components_and_official_split_leakage(self):
        pairs = [{"a": "A", "b": "B", "label": 1}, {"a": "C", "b": "D", "label": 0}]
        patients = {"A": "P1", "B": None, "C": "P1", "D": "P2"}
        groups = _components(list(patients), pairs, patients)
        self.assertIn(["A", "B", "C"], groups)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.csv"
            path.write_text("AccessionNumber,split\nA,train\nB,val\nC,train\nD,calibration\n")
            with self.assertRaisesRegex(ValueError, "leaks"):
                _make_splits([{"accession": a, "label": 0} for a in patients], pairs, {"split_file": str(path)}, patients)

    def test_alias_nifti_geometry_and_ambiguous_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "Composition" / "0001" / "S1" / "S1.nii.gz"
            path.parent.mkdir(parents=True)
            arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            nib.save(nib.Nifti1Image(arr, np.diag([-2., 3., 4., 1.])), path)
            records = discover_cases(root, {})
            volume, meta = load_sequence(records[0])
            self.assertEqual(volume.shape, (4, 3, 2))
            self.assertEqual(meta["orientation"], ["R", "A", "S"])
            np.testing.assert_array_equal(volume, arr[::-1].transpose(2, 1, 0))
            self.assertEqual(PathResolver(root).series_path("0001", "S1", "compositing"), path)
            alternate = root / "fake" / "0001" / "S2" / "S2.nii.gz"
            alternate.parent.mkdir(parents=True)
            nib.save(nib.Nifti1Image(arr, np.eye(4)), alternate)
            with self.assertRaisesRegex(ValueError, "multiple source"):
                discover_cases(root, {})

    def test_dicom_spatial_order_and_nifti_equivalence(self):
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            series_uid = generate_uid()
            for z in (2, 0, 1):
                meta = FileMetaDataset()
                meta.TransferSyntaxUID = ExplicitVRLittleEndian
                meta.MediaStorageSOPClassUID = MRImageStorage
                meta.MediaStorageSOPInstanceUID = generate_uid()
                dataset = FileDataset(str(root / f"{2-z}.dcm"), {}, file_meta=meta, preamble=b"\0"*128)
                dataset.SOPClassUID = MRImageStorage
                dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
                dataset.SeriesInstanceUID = series_uid
                dataset.AccessionNumber = "0001"
                dataset.PatientID = "P1"
                dataset.Modality = "MR"
                dataset.Rows = dataset.Columns = 2
                dataset.SamplesPerPixel = 1
                dataset.PhotometricInterpretation = "MONOCHROME2"
                dataset.BitsAllocated = dataset.BitsStored = 16
                dataset.HighBit, dataset.PixelRepresentation = 15, 0
                dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
                dataset.ImagePositionPatient = [0, 0, z*2]
                dataset.PixelSpacing = [1, 1]
                dataset.PixelData = np.full((2, 2), z+1, dtype=np.uint16).tobytes()
                dataset.save_as(dataset.filename, enforce_file_format=True)
            records = discover_cases(root, {})
            volume, meta = load_sequence(records[0])
            np.testing.assert_array_equal(volume[:, 0, 0], [1, 2, 3])
            self.assertEqual(meta["patient_id"], "P1")
            reference = root / "reference" / "N1" / "N1.nii.gz"
            reference.parent.mkdir(parents=True)
            nib.save(nib.Nifti1Image(volume.transpose(2, 1, 0), np.asarray(meta["affine_ras_xyz"])), reference)
            second, _ = load_sequence({"kind": "nifti", "series_uid": "N1", "path": str(reference)})
            np.testing.assert_array_equal(volume, second)


if __name__ == "__main__":
    unittest.main()
