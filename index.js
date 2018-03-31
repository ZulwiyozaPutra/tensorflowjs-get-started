const tensorflow = require('@tensorflow/tfjs');

const model = tensorflow.sequential();
model.add(tensorflow.layers.dense({ units: 1, inputShape: [1]}));

model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

const xs = tensorflow.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tensorflow.tensor2d([1, 3, 5, 7], [4, 1]);

model.fit(xs, ys).then(() => {
	model.predict(tensorflow.tensor2d([5], [1, 1])).print();
});
