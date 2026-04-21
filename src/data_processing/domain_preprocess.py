import pandas as pd
import numpy as np

class GermanCreditDomainPreprocessor:
    """
    Xử lý đặc thù (Domain-specific) cho bộ dữ liệu German Credit.
    Chỉ thực hiện các phép biến đổi toán học thay đổi phân phối và logic nghiệp vụ.
    """
    def __init__(self):
        pass

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        X = X_raw.copy()

        # 1. Xử lý lệch phân phối (Log-transform)
        # Cộng thêm 1 (log1p) để tránh lỗi log(0) nếu có
        if 'Amount' in X.columns:
            X['Amount'] = np.log1p(X['Amount'])
        if 'Age' in X.columns:
            X['Age'] = np.log1p(X['Age'])

        # # 2. Ép kiểu biến Rời rạc bị mất cân bằng thành Phân loại (Categorical)
        # # ExistingCredits: Gộp giá trị >= 3 thành nhóm '3+'
        # if 'ExistingCredits' in X.columns:
        #     X['ExistingCredits'] = X['ExistingCredits'].apply(
        #         lambda x: '3+' if pd.notna(x) and int(x) >= 3 else str(int(x))
        #     )

        # # 3. Ép kiểu các biến Thứ bậc (Ordinal) về chuỗi để Generic Preprocessor nhận diện là Categorical
        # ordinal_cols = ['InstallmentRate', 'ResidenceSince']
        # for col in ordinal_cols:
        #     if col in X.columns:
        #         X[col] = X[col].astype(str)

        # # 5. Xử lý biến nhị phân Liable (chuyển 1, 2 thành 0, 1)
        # if 'Liable' in X.columns:
        #     X['Liable'] = X['Liable'].map({1: 0, 2: 1})

        return X

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Đảo ngược các phép biến đổi toán học để phục vụ hiển thị Counterfactual Explanations.
        """
        X = df.copy()

        # 1. Đảo ngược Log-transform bằng Exponential (expm1)
        if 'Amount' in X.columns:
            X['Amount'] = np.expm1(X['Amount'])
            X['Amount'] = np.round(X['Amount'], 2) # Làm tròn tiền tệ

        if 'Age' in X.columns:
            X['Age'] = np.expm1(X['Age'])
            X['Age'] = np.round(X['Age']).astype(int) # Tuổi phải là số nguyên

        # # 2. Đảo ngược biến nhị phân (Nếu cần hiển thị lại giao diện gốc)
        # if 'Liable' in X.columns:
        #     X['Liable'] = X['Liable'].map({0: 1, 1: 2})
            
        # Lưu ý: Các cột Categorical (ExistingCredits, InstallmentRate) 
        # sẽ giữ nguyên dạng String ('1', '2', '3+') để hiển thị trực tiếp cho User.

        return X