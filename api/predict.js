// api/predict.js

const tf = require('@tensorflow/tfjs-node');
const fetch = require('node-fetch');

let model;
async function loadModel() {
  if (!model) {
    model = await tf.loadLayersModel(
      'file://' + process.cwd() + '/model/zone_model_v1.h5'
    );
  }
  return model;
}

module.exports = async (req, res) => {
  try {
    const { address } = req.body;
    if (!address) return res.status(400).json({ error: 'Missing address' });

    // 1) Geocode to lat/lng
    const geoRes = await fetch(
      `https://maps.googleapis.com/maps/api/geocode/json?address=${encodeURIComponent(address)}&key=${process.env.GOOGLE_MAPS_KEY}`
    );
    const geoJson = await geoRes.json();
    if (!geoJson.results || !geoJson.results[0]) {
      return res.status(400).json({ error: 'Invalid address' });
    }
    const { lat, lng } = geoJson.results[0].geometry.location;

    // 2) Fetch the satellite image
    const mapUrl = `https://maps.googleapis.com/maps/api/staticmap?center=${lat},${lng}&zoom=19&size=512x512&maptype=satellite&key=${process.env.GOOGLE_MAPS_KEY}`;
    const mapResp = await fetch(mapUrl);
    const imgBuffer = await mapResp.buffer();

    // 3) Decode & preprocess
    const tfimg = tf.node
      .decodeImage(imgBuffer, 3)
      .resizeNearestNeighbor([512, 512])
      .toFloat()
      .expandDims();

    // 4) Predict
    const m = await loadModel();
    const prediction = m.predict(tfimg);
    const probs = prediction.arraySync()[0];
    const zoneClass = probs[0] > probs[1] ? 6 : 8;
    const confidence = Math.max(...probs);

    // 5) Return
    res.status(200).json({ zones: zoneClass, confidence });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Prediction failed', detail: err.message });
  }
};
