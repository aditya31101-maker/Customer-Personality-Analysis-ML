from pathlib import Path

from flask import Blueprint, current_app, render_template, request
from werkzeug.utils import secure_filename

from .services.analysis_service import AnalysisService


main_bp = Blueprint("main", __name__)


@main_bp.route("/", methods=["GET", "POST"])
def index():
    service = AnalysisService(
        upload_folder=Path(current_app.config["UPLOAD_FOLDER"]),
        default_dataset=Path(current_app.config["DEFAULT_DATASET_PATH"]),
        output_folder=Path(current_app.config["OUTPUT_FOLDER"]),
    )

    context = {
        "result": None,
        "error": None,
        "default_dataset_exists": service.default_dataset.exists(),
    }

    if request.method == "POST":
        use_default = request.form.get("dataset_source", "default") == "default"

        try:
            if use_default:
                dataset_path = service.default_dataset
            else:
                uploaded_file = request.files.get("dataset_file")
                if not uploaded_file or not uploaded_file.filename:
                    raise ValueError("Please upload a CSV or TSV dataset file.")

                filename = secure_filename(uploaded_file.filename)
                dataset_path = service.upload_folder / filename
                service.upload_folder.mkdir(parents=True, exist_ok=True)
                uploaded_file.save(dataset_path)

            context["result"] = service.run_analysis(dataset_path)
        except Exception as exc:
            context["error"] = str(exc)

    return render_template("index.html", **context)
