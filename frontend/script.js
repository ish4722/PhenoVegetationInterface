// Image Slider Logic
const imageLinks = [
                "/Users/ishan/PhenoVegetationInterface-1/assets/1.JPG",
                "/Users/ishan/PhenoVegetationInterface-1/assets/2.JPG",
                "/Users/ishan/PhenoVegetationInterface-1/assets/.JPG",
                "/Users/ishan/PhenoVegetationInterface-1/assets/4.JPG"
];

let currentIndex = 0;
const sliderImage = document.getElementById("slider-image");

function changeImage() {
    currentIndex = (currentIndex + 1) % imageLinks.length;
    sliderImage.src = imageLinks[currentIndex];
}

// Change the image every 2 seconds
setInterval(changeImage, 2000);


document.querySelector(".generate-btn").addEventListener("click", async () => {
    const formData = new FormData();
    const uploadInput = document.getElementById("upload-input");
    const startDate = document.querySelector('input[type="date"]:nth-child(1)').value;
    const endDate = document.querySelector('input[type="date"]:nth-child(2)').value;

    // Collect uploaded files
    for (let file of uploadInput.files) {
        formData.append("images", file);
    }

    // Collect start and end dates
    formData.append("start_date", startDate);
    formData.append("end_date", endDate);

    // Collect filters
    document.querySelectorAll(".remove-images input[type=checkbox]").forEach((checkbox) => {
        if (checkbox.checked) {
            formData.append("filters", checkbox.nextSibling.textContent.trim());
        }
    });

    try {
        const response = await fetch("http://127.0.0.1:5000/process", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            alert(`Error: ${error.error}`);
            return;
        }

        const data = await response.json();

        // // Display the graph
        // const graphImg = document.createElement("img");
        // graphImg.src = `http://127.0.0.1:5000${data.graph_path}`;
        // graphImg.alt = "Generated Vegetation Graph";
        // document.body.appendChild(graphImg);

        // Provide Excel download link
        const excelLink = document.createElement("a");
        excelLink.href = `http://127.0.0.1:5000${data.excel_path}`;
        excelLink.textContent = "Download Vegetation Report (Excel)";
        excelLink.download = "vegetation_report.xlsx";
        document.body.appendChild(excelLink);
    } catch (error) {
        console.error("Error:", error);
        alert("An error occurred while processing your request.");
    }
});

