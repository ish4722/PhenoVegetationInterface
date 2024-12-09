// Image Slider Logic
const imageLinks = [
                "1.JPG",
                "2.JPG",
                "3.JPG",
                "4.JPG"
];

let currentIndex = 0;
const sliderImage = document.getElementById("slider-image");

function changeImage() {
    currentIndex = (currentIndex + 1) % imageLinks.length;
    sliderImage.src = imageLinks[currentIndex];
}

// Change the image every 2 seconds
setInterval(changeImage, 2000);


document.addEventListener("DOMContentLoaded", () => {
    // File input handling
    const uploadInput = document.getElementById("upload-input");
    const generateBtn = document.querySelector(".generate-btn");

    // Uploading files
    generateBtn.addEventListener("click", async () => {
        const files = uploadInput.files;

        // Ensure at least one file is selected
        if (files.length === 0) {
            alert("Please upload at least one image.");
            return;
        }

        const formData = new FormData();

        // Append files to FormData
        for (let file of files) {
            formData.append("images", file);
        }

        // Fetch date values
        const dateInputs = document.querySelectorAll(".calendar-box input");
        const startDate = dateInputs[0].value;
        const endDate = dateInputs[1].value;

        if (startDate && endDate) {
            formData.append("start_date", startDate);
            formData.append("end_date", endDate);
        }

        // Fetch checkboxes
        const checkboxes = document.querySelectorAll(".remove-images input[type='checkbox']");
        checkboxes.forEach((checkbox) => {
            if (checkbox.checked) {
                formData.append("filters", checkbox.parentNode.textContent.trim());
            }
        });

        // Send data to the Flask server
        try {
            const response = await fetch("http://127.0.0.1:5000/process", {
                method: "POST",
                body: formData,
            });

            if (response.ok) {
                const data = await response.json();
                alert("Graph generated successfully!");
                console.log("Graph Data:", data);
                // Handle displaying the graph data or any other UI updates
            } else {
                const error = await response.text();
                alert(`Error: ${error}`);
            }
        } catch (error) {
            console.error("Error:", error);
            alert("An error occurred while generating the graph. Please try again.");
        }
    });
});
