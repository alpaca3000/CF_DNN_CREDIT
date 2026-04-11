from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def load_data(dataset_name: str) -> pd.DataFrame:
	"""Load dữ liệu từ data/ theo dataset_name đã chuẩn xác."""
	file_map = {
		"german_credit": "german_credit.csv",
		"gmsc": "gmsc.csv",
		"lending_club": "lending_club_50k.csv",
	}

	if dataset_name not in file_map:
		raise ValueError("dataset_name phải là: 'german_credit', 'gmsc', hoặc 'lending_club'.")

	path = DATA_DIR / file_map[dataset_name]
	if not path.exists():
		raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")

	return pd.read_csv(path)


def split_data(
	df: pd.DataFrame,
	dataset_name: str,
	target_col: str | None = None,
	random_state: int = 42,
) -> Union[
	tuple[pd.DataFrame, pd.DataFrame],
	tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
]:
	"""
	Chia dữ liệu theo quy tắc:
	- Nếu dataset_name == 'german_credit'  -> Train/Test = 80/20
	- Ngược lại                     -> Train/Valid/Test = 70/15/15

	Nếu truyền `target_col`, sẽ stratify theo cột này.

	Returns:
	- german: (train_df, test_df)
	- others: (train_df, valid_df, test_df)
	"""
	stratify_y = df[target_col] if (target_col is not None and target_col in df.columns) else None

	if dataset_name == "german":
		train_df, test_df = train_test_split(
			df,
			test_size=0.2,
			random_state=random_state,
			stratify=stratify_y,
		)
		return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

	# 70/15/15: tách 70/30 trước, rồi 30 -> 15/15
	train_df, temp_df = train_test_split(
		df,
		test_size=0.3,
		random_state=random_state,
		stratify=stratify_y,
	)

	temp_stratify = (
		temp_df[target_col]
		if (target_col is not None and target_col in temp_df.columns)
		else None
	)
	valid_df, test_df = train_test_split(
		temp_df,
		test_size=0.5,
		random_state=random_state,
		stratify=temp_stratify,
	)

	return (
		train_df.reset_index(drop=True),
		valid_df.reset_index(drop=True),
		test_df.reset_index(drop=True),
	)
