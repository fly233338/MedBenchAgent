"""Tiny BrEaST-shaped sources shared by conversion and collaboration checks."""
from openpyxl import Workbook
from PIL import Image


def create_breast_source(root):
    root.mkdir(parents=True)
    workbook = root / "clinical.xlsx"
    book = Workbook()
    sheet = book.active
    sheet.title = "BrEaST-Lesions-USG clinical dat"
    sheet.append(["CaseID", "Image_filename", "Mask_tumor_filename", "Mask_other_filename", "Pathology"])
    sheet.append([1, "one-a.png & one-b.png", "one-mask.png", None, "benign"])
    sheet.append([2, "two.png", "two-mask.png", None, "malignant"])
    sheet.append([3, "three.png", None, None, "benign"])
    sheet.append([4, "four.png", None, None, None])
    dictionary = book.create_sheet("Data Dictionary")
    dictionary.append(["Field", "Description"])
    dictionary.append(["Pathology", "Recorded lesion label; blank means unknown"])
    book.save(workbook)
    book.close()
    for name in ("one-a", "one-b", "two", "three", "four", "one-mask", "two-mask"):
        image = Image.new("L", (8, 8), 0 if "mask" in name else 100)
        if "mask" in name:
            image.paste(255, (2, 2, 6, 6))
        image.save(root / f"{name}.png")
    return workbook
