# load và định dạng tất cả data thành dạng dataframe dễ sử dụng

import pandas as pd
import zipfile
import os
import ssl  # Thêm thư viện để xử lý chứng chỉ bảo mật
from pathlib import Path

# Cấu hình bypass lỗi SSL khi tải dữ liệu từ internet
ssl._create_default_https_context = ssl._create_unverified_context

DATA_DIR = Path(__file__).resolve().parent

# 1. TẢI DỮ LIỆU GERMAN CREDIT
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
columns = [
    'Status', 'Duration', 'History', 'Purpose', 'Amount', 'Savings',
    'Employment', 'InstallmentRate', 'PersonalStatus', 'OtherDebtors',
    'ResidenceSince', 'Property', 'Age', 'OtherPlans', 'Housing',
    'ExistingCredits', 'Job', 'Liable', 'Telephone', 'ForeignWorker', 'Class'  # Giữ nguyên tên cột là Class theo ý bạn
]

gcredit = pd.read_csv(url, sep=' ', header=None, names=columns)

# Giữ nguyên ánh xạ nhãn: 1 -> 1 (Đồng ý), 2 -> 0 (Từ chối)
gcredit['Class'] = gcredit['Class'].map({1: 1, 2: 0})

# áp mapping để dữ liệu dễ đọc khi EDA
# Mapping codebook + bảng metadata cột (chi tiết tiếng Việt)
code_maps = {
    'Status': {
        'A11': '< 0 DM',
        'A12': '0 <= ... < 200 DM',
        'A13': '>= 200 DM / salary assignment >= 1 year',
        'A14': 'no checking account'
    },
    'History': {
        'A30': 'no credits taken / all paid back duly',
        'A31': 'all credits at this bank paid back duly',
        'A32': 'existing credits paid back duly till now',
        'A33': 'delay in paying off in the past',
        'A34': 'critical account / other credits existing'
    },
    'Purpose': {
        'A40': 'car (new)', 'A41': 'car (used)', 'A42': 'furniture/equipment',
        'A43': 'radio/television', 'A44': 'domestic appliances', 'A45': 'repairs',
        'A46': 'education', 'A47': 'vacation', 'A48': 'retraining',
        'A49': 'business', 'A410': 'others'
    },
    'Savings': {
        'A61': '< 100 DM', 'A62': '100 <= ... < 500 DM',
        'A63': '500 <= ... < 1000 DM', 'A64': '>= 1000 DM',
        'A65': 'unknown / no savings account'
    },
    'Employment': {
        'A71': 'unemployed', 'A72': '< 1 year',
        'A73': '1 <= ... < 4 years', 'A74': '4 <= ... < 7 years',
        'A75': '>= 7 years'
    },
    'PersonalStatus': {
        'A91': 'male: divorced/separated',
        'A92': 'female: divorced/separated/married',
        'A93': 'male: single',
        'A94': 'male: married/widowed',
        'A95': 'female: single'
    },
    'OtherDebtors': {'A101': 'none', 'A102': 'co-applicant', 'A103': 'guarantor'},
    'Property': {
        'A121': 'real estate',
        'A122': 'building society savings/life insurance',
        'A123': 'car or other',
        'A124': 'unknown / no property'
    },
    'OtherPlans': {'A141': 'bank', 'A142': 'stores', 'A143': 'none'},
    'Housing': {'A151': 'rent', 'A152': 'own', 'A153': 'for free'},
    'Job': {
        'A171': 'unemployed / unskilled non-resident',
        'A172': 'unskilled resident',
        'A173': 'skilled employee / official',
        'A174': 'management / self-employed / highly qualified'
    },
    'Telephone': {'A191': 'none', 'A192': 'yes (registered)'},
    'ForeignWorker': {'A201': 'yes', 'A202': 'no'},
}

gcredit_mapped = gcredit.copy()
for col, mapper in code_maps.items():
    if col in gcredit_mapped.columns:
        gcredit_mapped[col] = gcredit_mapped[col].map(mapper).fillna(gcredit_mapped[col])

gcredit = gcredit_mapped.copy()


# 2. LOAD DỮ LIỆU LENDING CLUB 
lending_file_path = DATA_DIR / "lendingclub.csv"

if lending_file_path.exists():
    lending_club = pd.read_csv(lending_file_path)
    print(f"Đã load xong {len(lending_club)} dòng dữ liệu Lending Club.")
else:
    print("Cảnh báo: Không tìm thấy file lendingclub.csv trong thư mục data")

# 3. LOAD DỮ LIỆU GMSC ĐÃ XỬ LÝ
gmsc_file_path = DATA_DIR / "gmsc.csv"

if gmsc_file_path.exists():
    gmsc_cleaned = pd.read_csv(gmsc_file_path)
    print(f"Đã nạp dữ liệu GMSC sạch: {gmsc_cleaned.shape}")
else:
    print("Cảnh báo: Chưa tìm thấy file gmsc.csv sạch trong thư mục data.")
    
# 4. LƯU VÀO THƯ MỤC DATA
gcredit.to_csv(DATA_DIR / "german_credit.csv", index=False)

if 'lending_club' in locals():
    lending_club.to_csv(DATA_DIR / "lending.csv", index=False)