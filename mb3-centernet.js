const fs = require('fs');
const path = require('path');
const process = require('process');
const log = require('@vladmandic/pilogger');
const tf = require('@tensorflow/tfjs-node');
const canvas = require('canvas');
const { labels } = require('./coco-labels'); // note that number of labels *must* match expected output of the model

const modelOptions = {
  modelPath: 'file://model-f16/mb3-centernet.json',
  outputTensors: ['tower_0/detections'], // ['tower_0/detections', 'tower_0/wh', 'tower_0/keypoints']
  minScore: 0.10, // low confidence, but still remove irrelevant
  iouThreshold: 0.40, // percentage when removing overlapped boxes
  maxResults: 20, // high number of results, but likely never reached
};

// save image with processed results
async function saveImage(img, res) {
  // create canvas
  const c = new canvas.Canvas(img.inputShape[0], img.inputShape[1]);
  const ctx = c.getContext('2d');

  // load and draw original image
  const original = await canvas.loadImage(img.fileName);
  ctx.drawImage(original, 0, 0, c.width, c.height);
  // const fontSize = Math.trunc(c.width / 50);
  const fontSize = Math.round(Math.sqrt(c.width) / 1.5);
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'white';
  ctx.font = `${fontSize}px "Segoe UI"`;

  // draw all detected objects
  for (const obj of res) {
    // draw label at center
    ctx.fillStyle = 'black';
    ctx.fillText(`${Math.round(100 * obj.score)}% ${obj.label}`, obj.box[0] + 5, obj.box[1] - 3);
    ctx.fillStyle = 'white';
    ctx.fillText(`${Math.round(100 * obj.score)}% ${obj.label}`, obj.box[0] + 4, obj.box[1] - 4);
    // ctx.fillText(`${Math.round(100 * obj.score)}% [${obj.strideSize}] #${obj.class} ${obj.label}`, obj.box[0] + 4, obj.box[1] - 4);
    // draw rect using x,y,h,w
    ctx.rect(obj.box[0], obj.box[1], obj.box[2] - obj.box[0], obj.box[3] - obj.box[1]);
  }
  ctx.stroke();

  // write canvas to jpeg
  const outImage = `outputs/${path.basename(img.fileName)}`;
  const out = fs.createWriteStream(outImage);
  out.on('finish', () => log.state('Created output image:', outImage));
  out.on('error', (err) => log.error('Error creating image:', outImage, err));
  const stream = c.createJPEGStream({ quality: 0.6, progressive: true, chromaSubsampling: true });
  stream.pipe(out);
}

// load image from file and prepares image tensor that fits the model
async function loadImage(fileName, inputSize) {
  const data = fs.readFileSync(fileName);
  const obj = tf.tidy(() => {
    const buffer = tf.node.decodeImage(data);
    const resize = tf.image.resizeBilinear(buffer, [inputSize, inputSize]);
    const cast = resize.cast('float32');
    const expand = cast.expandDims(0);
    const tensor = expand;
    const img = { fileName, tensor, inputShape: [buffer.shape[1], buffer.shape[0]], outputShape: tensor.shape, size: buffer.size };
    return img;
  });
  return obj;
}

// process model results
async function processResults(res, inputSize, outputShape) {
  const detections = res.arraySync();
  const squeezeT = tf.squeeze(res);
  res.dispose();
  const arr = tf.split(squeezeT, 6, 1); // x1, y1, x2, y2, score, class
  squeezeT.dispose();
  const stackT = tf.stack([arr[1], arr[0], arr[3], arr[2]], 1); // tf.nms expects y, x
  const boxesT = stackT.squeeze();
  const scoresT = arr[4].squeeze();
  const classesT = arr[5].squeeze();
  arr.forEach((t) => t.dispose());
  // @ts-ignore boxesT type is not correctly inferred
  const nmsT = await tf.image.nonMaxSuppressionAsync(boxesT, scoresT, modelOptions.maxResults, modelOptions.iouThreshold, modelOptions.minScore);
  boxesT.dispose();
  scoresT.dispose();
  classesT.dispose();
  const nms = nmsT.dataSync();
  nmsT.dispose();
  const results = [];
  for (const id of nms) {
    const score = detections[0][id][4];
    const classVal = detections[0][id][5];
    const label = labels[classVal].label;
    const boxRaw = [
      detections[0][id][0] / inputSize,
      detections[0][id][1] / inputSize,
      detections[0][id][2] / inputSize,
      detections[0][id][3] / inputSize,
    ];
    const box = [
      Math.trunc(boxRaw[0] * outputShape[0]),
      Math.trunc(boxRaw[1] * outputShape[1]),
      Math.trunc(boxRaw[2] * outputShape[0]),
      Math.trunc(boxRaw[3] * outputShape[1]),
    ];
    results.push({ id, score, class: classVal, label, box, boxRaw });
  }
  return results;
}

async function main() {
  log.header();

  // init tensorflow
  await tf.enableProdMode();
  await tf.setBackend('tensorflow');
  await tf.ENV.set('DEBUG', false);
  await tf.ready();

  // load model
  const model = await tf.loadGraphModel(modelOptions.modelPath);
  log.info('Loaded model', modelOptions, 'tensors:', tf.engine().memory().numTensors, 'bytes:', tf.engine().memory().numBytes);
  // @ts-ignore
  log.info('Model Signature', model.signature);

  // load image and get approprite tensor for it
  const inputSize = Object.values(model.modelSignature['inputs'])[0].tensorShape.dim[2].size;
  const imageFile = process.argv.length > 2 ? process.argv[2] : null;
  if (!imageFile || !fs.existsSync(imageFile)) {
    log.error('Specify a valid image file');
    process.exit();
  }
  const img = await loadImage(imageFile, inputSize);
  log.info('Loaded image:', img.fileName, 'inputShape:', img.inputShape, 'outputShape:', img.outputShape);

  // run actual prediction
  const t0 = process.hrtime.bigint();
  const res = model.execute(img.tensor, modelOptions.outputTensors);
  const t1 = process.hrtime.bigint();
  log.info('Inference time:', Math.round(parseInt((t1 - t0).toString()) / 1000 / 1000), 'ms');

  // process results
  const results = await processResults(res, inputSize, img.inputShape);
  const t2 = process.hrtime.bigint();
  log.info('Processing time:', Math.round(parseInt((t2 - t1).toString()) / 1000 / 1000), 'ms');

  // print results
  log.data('Results:', results);

  // save processed image
  await saveImage(img, results);
}

main();
