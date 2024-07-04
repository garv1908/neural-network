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

ctx.lineWidth = 20;
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
    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": 'application/json'
        },
        body: JSON.stringify({ 'image': dataURL })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data)
        const predictions = data.map((percentage, number) => ({ number, percentage }));
        
        predictions.sort((a, b) => b.percentage - a.percentage);
        
        // Display the top 5 predictions
        for (var j = 0; j < 5; j++) {
            document.getElementById(`row${j+1}`).querySelector('td').innerText = predictions[j].number;
            document.getElementById(`percentage${j+1}`).innerText = `${(predictions[j].percentage * 100).toFixed(4)}%`;
        }
    })
    .catch(error => {
        console.error("Prediction Error:", error);
    });
}

function changeToFNN() {
    fetch("/changeModel", {
        method: "POST",
        headers: {
            "Content-Type": 'application/json'
        },
        body: "FNN"
    })
}

function changeToCNN() {
    fetch("/changeModel", {
        method: "POST",
        headers: {
            "Content-Type": 'application/json'
        },
        body: "CNN"
    })
    .then(data => {
        console.log("Changed model.", data)
    })
}


canvas.addEventListener("mousedown", startPosition);
canvas.addEventListener("mouseup", endPosition);
canvas.addEventListener("mousemove", draw);