"use strict";

const canvas = document.getElementsByTagName("canvas")[0];
const ctx = canvas.getContext("2d");
let painting = false;

function startPosition(ev) {
    painting = true;
    ctx.beginPath();
    draw(ev);
}

function endPosition(ev) {
    painting = false;
    ctx.closePath();
}

ctx.lineWidth = 10;
ctx.lineCap = "round";
ctx.strokeStyle = "black";

function draw(ev) {
    if (!painting) return;

    ctx.lineTo(ev.clientX - canvas.offsetLeft, ev.clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(ev.clientX - canvas.offsetLeft, ev.clientY - canvas.offsetTop);
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function submitDrawing() {
    const dataURL = canvas.toDataURL();
    console.log(dataURL)
    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": 'application/json'
        },
        body: JSON.stringify({ 'image': dataURL })
    })
    .then(response => response.json())
    .then(data => {
        console.log("Prediction Result:", data);
        document.getElementById('result').innerText = `Prediction: ${data.prediction}`;
    })
    .catch(error => {
        console.error("Prediction Error:", error);
    });
}

canvas.addEventListener("mousedown", startPosition);
canvas.addEventListener("mouseup", endPosition);
canvas.addEventListener("mousemove", draw);
