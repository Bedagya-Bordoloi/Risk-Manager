import pandas as pd


def load_user_file(file_path):

    try:

        file_path = file_path.lower()

        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)

        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)

        else:
            raise ValueError("Unsupported file format. Use CSV or XLSX.")

        return df

    except Exception as e:
        raise ValueError(f"Error loading file: {e}")
