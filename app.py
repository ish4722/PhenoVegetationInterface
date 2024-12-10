from flask import Flask, request, jsonify, send_file
import os
from backend.preprocessing.time_filter import apply_time_filter
from backend.preprocessing.filters import apply_filters
from backend.segmentation.model import predict_masks
from backend.analysis.vegetation_indices import calculate_indices
from backend.analysis.excel_writer import create_excel
from backend.segmentation.clustering import perform_clustering
from backend.segmentation.roi_processing import ROI_Points, ROI_processing
import uuid  # To generate unique filenames
from config import *


app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return "Pheno AI Flask Server is running!"

@app.route("/process", methods=["POST"])
def process():
    try:
        # Step 1: Handle Uploaded Files
        if 'images' not in request.files:
            return jsonify({"error": "No images uploaded"}), 400

        images = request.files.getlist("images")
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")
        filters = request.form.getlist("filters")

        # Save uploaded images to the upload folder
        image_paths = []
        for image in images:
            filename = f"{uuid.uuid4()}_{image.filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            image.save(filepath)
            image_paths.append(filepath)

        # Step 2: Apply Time Filters
        filtered_images = apply_time_filter(image_paths, start_date, end_date)

        # Step 3: Apply Additional Filters (Blurry, Snowy, Foggy)
        filtered_images = apply_filters(filtered_images, filters)

        # Step 4: Predict Segmentation with Model
        MASKS = predict_masks(MODEL_PATH,filtered_images)

        n_ROIs, points, ROI_area, scaled_height, scaled_width = ROI_Points(MASKS, filtered_images)

        centroids, labels = perform_clustering(points, ROI_area, n_ROIs)

        red, green, blue = ROI_processing(labels, points, ROI_area, scaled_height, scaled_width, filtered_images)

        # Step 5: Calculate Vegetation Indices
        indices_list = []
        for red, green, blue in zip(red, green, blue):
            indices = calculate_indices(red, green, blue)
            indices_list.append(indices)

        # Step 6: Generate Excel Report
        excel_path = os.path.join(OUTPUT_FOLDER, "vegetation_report.xlsx")
        create_excel(indices_list, excel_path)

        # Step 7: Generate Graph
        graph_path = os.path.join(OUTPUT_FOLDER, "vegetation_graph.png")
        # indices_list.plot_graph(output_path=graph_path)  # Assuming `plot_graph` is implemented in `vegetation_data`

        # Step 8: Return Files to User
        return jsonify({
            "excel_path": f"/download/excel/{os.path.basename(excel_path)}",
            # "graph_path": f"/download/graph/{os.path.basename(graph_path)}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download/excel/<filename>")
def download_excel(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)

# @app.route("/download/graph/<filename>")
# def download_graph(filename):
#     return send_file(os.path.join(OUTPUT_FOLDER, filename), mimetype="image/png").

if __name__ == "__main__":
    app.run(debug=True)
