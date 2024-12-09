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
