const tf = require('@tensorflow/tfjs-node');

async function loadModel() {
    return await tf.loadGraphModel('file://./tmp/loaded_models/convolutional_tfjs/model.json');
}

async function predict(model, imageData) {
    let tensor = tf.tensor(imageData);
    tensor = tensor.expandDims(-1);
    tensor = tensor.expandDims(0);

    const prediction = model.predict(tensor);
    const predictionArray = await prediction.array();
    return predictionArray;
}

async function main() {
    const model = await loadModel();
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', async (data) => {
        try {
            const parsedData = JSON.parse(data);
            const image = parsedData.image;
            if (!image) {
                process.stdout.write(JSON.stringify({ error: 'No image data provided' }));
                return;
            }
            const prediction = await predict(model, image);
            process.stdout.write(JSON.stringify({ prediction }));
        } catch (error) {
            process.stdout.write(JSON.stringify({ error: error.message }));
        }
    });
}

main();
