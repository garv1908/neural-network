"use strict";

const canvas = document.getElementsByTagName("canvas")[0];
const ctx = canvas.getContext('2d');
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
ctx.lineCap = 'round';
ctx.strokeStyle = 'black';

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
    const img = new Image();
    img.src = dataURL;
    img.onload = () => {
        const scaledCanvas = document.createElement('canvas');
        scaledCanvas.width = 28;
        scaledCanvas.height = 28;
        const scaledCtx = scaledCanvas.getContext('2d');
        scaledCtx.drawImage(img, 0, 0 , 28, 28);

        const imageData = scaledCtx.getImageData(0, 0, 28, 28);
        const pixels = imageData.data;
        const greyScaleData = [];

        for (let i = 0; i < pixels.length; i += 4) {
            const grey = (pixels[i] + pixels[i + 1] + pixels[i + 2] / 3)
            greyScaleData.push(grey / 255)
        }
    }
}

canvas.addEventListener("mousedown", startPosition);
canvas.addEventListener("mouseup", endPosition);
canvas.addEventListener("mousemove", draw);
