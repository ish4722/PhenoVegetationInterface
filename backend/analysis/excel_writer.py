import pandas as pd

def create_excel(data, output_path):
    """Save vegetation indices to an Excel file."""
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)
