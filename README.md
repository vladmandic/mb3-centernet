# MobileNet-v3 with CenterNet Object Detection for TFJS and NodeJS

Models included in </model-f16> and </model-f16> were converted to TFJS Graph model format from the original repository
Models descriptors and signature have been additionally parsed for readability

Actual model parsing implementation in `mb3-centernet.js` does not follow original Pytyhon implementation and is fully custom and optimized for JavaScript execution

Function `processResults()` takes output of `model.execute` and returns array of objects:

- id: internal number of detection box, used only for debugging
- score: value 0..1
- class: coco class number
- label: coco label as string
- box: detection box [x1, y1, x2, y2] normalized to input image dimensions
- boxRaw: detection box [x1, y1, x2, y2] normalized to 0..1

<br>

## Example

![Example Image](outputs/cars.jpg)

<br><hr><br>

## Conversion Notes

Source: <https://github.com/610265158/mobilenetv3_centernet>

```shell
tensorflowjs_converter \
  --input_format tf_frozen_model \
  --output_format tfjs_graph_model \
  --quantize_float16=* \
  --output_node_names="tower_0/detections,tower_0/keypoints,tower_0/wh" model-frozen/detector.pb model-f16
```

```js
2021-05-19 07:12:34 INFO:  nanodet version 0.0.1
2021-05-19 07:12:34 INFO:  User: vlado Platform: linux Arch: x64 Node: v16.0.0
2021-05-19 07:12:34 DATA:  created on: 2021-05-18T21:49:02.930Z
2021-05-19 07:12:34 INFO:  graph model: /home/vlado/dev/mb3-centernet/model-f16/mb3-centernet.json
2021-05-19 07:12:34 INFO:  size: { unreliable: true, numTensors: 267, numDataBuffers: 267, numBytes: 8060260 }
2021-05-19 07:12:34 INFO:  model inputs based on signature
2021-05-19 07:12:34 INFO:  model outputs based on signature
2021-05-19 07:12:34 DATA:  inputs: [ { name: 'tower_0/images', dtype: 'DT_FLOAT', shape: [ 1, 512, 512, 3, [length]: 4 ] }, [length]: 1 ]
2021-05-19 07:12:34 DATA:  outputs: [
  { id: 0, name: 'tower_0/wh', dytpe: 'DT_FLOAT', shape: [ 1, 128, 128, 4, [length]: 4 ] },
  { id: 1, name: 'tower_0/keypoints', dytpe: 'DT_FLOAT', shape: [ 1, 128, 128, 80, [length]: 4 ] },
  { id: 2, name: 'tower_0/detections', dytpe: 'DT_FLOAT', shape: [ 1, 100, 6, [length]: 3 ] },
  [length]: 3
]
```

Where `tower_0/detections` is array of COCO classes * [ x1, y1, x2, y2, score, class ]  
`tower_0/detections` is built in-model from `tower_0/wh` which contains strided heatmap - since it's already processed into detections, we don't need heatmap post-processing  

<br><hr><br>

## Test

```shell
node ./mb3-centernet.js inputs/dog.jpg
```

```js
2021-05-18 19:37:38 INFO:  Loaded model { modelPath: 'file://model/mb3-centernet.json', outputTensors: [ 'tower_0/detections', [length]: 1 ], minScore: 0.1, iouThreshold: 0.4, maxResults: 20 } tensors: 267 bytes: 8060260
2021-05-18 19:37:38 INFO:  Model Signature {
  inputs: { 'tower_0/images': { name: 'tower_0/images', dtype: 'DT_FLOAT', tensorShape: { dim: [ { size: '1' }, { size: '512' }, { size: '512' }, { size: '3' }, [length]: 4 ] } } },
  outputs: { 'tower_0/wh': { name: 'tower_0/wh' }, 'tower_0/keypoints': { name: 'tower_0/keypoints' }, 'tower_0/detections': { name: 'tower_0/detections' } }
}
2021-05-18 19:37:38 INFO:  Loaded image: inputs/dog.jpg inputShape: [ 1536, 2048, [length]: 2 ] outputShape: [ 1, 512, 512, 3, [length]: 4 ]
2021-05-18 19:37:38 INFO:  Inference time: 216 ms
2021-05-18 19:37:38 INFO:  Processing time: 3 ms
2021-05-18 19:37:38 DATA:  Results: [
  {
    id: 0,
    score: 0.44118914008140564,
    class: 0,
    label: 'person',
    box: [ 678, 228, 1516, 1899, [length]: 4 ],
    boxRaw: [ 0.44152459502220154, 0.11151626706123352, 0.9870420694351196, 0.9275288581848145, [length]: 4 ]
  },
  {
    id: 1,
    score: 0.37394979596138,
    class: 16,
    label: 'dog',
    box: [ 4, 566, 826, 1504, [length]: 4 ],
    boxRaw: [ 0.0030441880226135254, 0.27652108669281006, 0.538006067276001, 0.7345627546310425, [length]: 4 ]
  }
]
2021-05-18 19:37:38 STATE: Created output image: outputs/dog.jpg
```
