# MobileNet-v3 with CenterNet Object Detection for TFJS and NodeJS

Models included in </model-f16> and </model-f16> were converted to TFJS Graph model format from the original repository
Models descriptors and signature have been additionally parsed for readability

Actual model parsing implementation in `mb3-centernet.js` does not follow original Pytyhon implementation and is fully custom and optimized for JavaScript execution

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
2021-05-18 17:54:29 DATA:  created on: 2021-05-18T21:49:00.080Z
2021-05-18 17:54:29 INFO:  graph model: /home/vlado/dev/mb3-centernet/graph-f16/model.json
2021-05-18 17:54:29 INFO:  size: { unreliable: true, numTensors: 267, numDataBuffers: 267, numBytes: 8060260 }
2021-05-18 17:54:29 INFO:  model inputs based on executor
2021-05-18 17:54:29 INFO:  model outputs based on executor
2021-05-18 17:54:29 DATA:  inputs: [ { name: 'tower_0/images', dtype: 'float32', shape: [ 1, 512, 512, 3 }
2021-05-18 17:54:29 DATA:  outputs: [
  { id: 0, name: 'tower_0/wh', dtype: 'DT_FLOAT', shape: undefined },
  { id: 1, name: 'tower_0/keypoints', dtype: 'DT_FLOAT', shape: undefined },
  { id: 2, name: 'tower_0/detections', dtype: 'DT_FLOAT', shape: undefined },
]
```

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
