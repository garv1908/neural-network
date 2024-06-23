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
        for (var j = 0; j < 5; j++) {
            
            var max = data[0];
            var maxIndex = 0;
            
            for (var i = 1; i < data.length; i++) {
                if (data[i] > max) {
                    maxIndex = i;
                    max = data[i];
                }
            }            
            
            document.getElementById(`row${j+1}`).querySelector('td').innerText = maxIndex;
            document.getElementById(`percentage${j+1}`).innerText = `${(max * 100).toFixed(4)}%`;
            data[maxIndex] = 0;
        }
    })
    .catch(error => {
        console.error("Prediction Error:", error);
    });
}

canvas.addEventListener("mousedown", startPosition);
canvas.addEventListener("mouseup", endPosition);
canvas.addEventListener("mousemove", draw);