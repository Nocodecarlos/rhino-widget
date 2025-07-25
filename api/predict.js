// api/predict.js

module.exports = async (req, res) => {
  res.status(200).json({ zones: 6, confidence: 0.99, stub: true });
};

