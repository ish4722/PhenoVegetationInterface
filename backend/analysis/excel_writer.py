import pandas as pd

def create_excel(data, output_path):
    """
    Save vegetation indices to an Excel file.

    Parameters:
        data: List of dictionaries or structured data containing vegetation indices.
        output_path: Path where the Excel file will be saved.
    """
    # Flatten and process the data if it's a list of dictionaries
    if isinstance(data, list) and all(isinstance(d, dict) for d in data):
        rows = []
        for idx, entry in enumerate(data, start=1):
            for key, values in entry.items():
                for i, value in enumerate(values):
                    rows.append({
                        "Image": idx,
                        "Metric": key,
                        "Pixel_Index": i + 1,
                        "Value": value
                    })
        df = pd.DataFrame(rows)
    else:
        # Directly convert to DataFrame if the structure is simple
        df = pd.DataFrame(data)

    # Save DataFrame to Excel
    df.to_excel(output_path, index=False)
