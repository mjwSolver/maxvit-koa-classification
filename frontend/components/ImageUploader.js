
import { useState } from 'react';
import styles from './ImageUploader.module.css'; // Optional: for styling

export default function ImageUploader() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [jsonData, setJsonData] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null); // Reset previous prediction
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please select an X-ray image first.');
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      // For debugging purposes, you can log the JSON response
      setJsonData(data);

      if (!response.ok) {
        throw new Error(data.error || 'Prediction request failed.');
      }

      setPrediction(data.predicted_grade);

    } catch (err) {
      setError(err.message);
      setPrediction(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <h2>Knee Osteoarthritis Severity Classification</h2>
      <p>Upload a knee X-ray image to predict the severity grade (0, 1, or 2).</p>
      
      <input type="file" onChange={handleFileChange} accept="image/png, image/jpeg" className={styles.fileInput} />
      
      {preview && <img src={preview} alt="X-ray preview" className={styles.preview} />}
      
      <button onClick={handlePredict} disabled={isLoading || !selectedFile} className={styles.predictButton}>
        {isLoading ? 'Analyzing...' : 'Predict Severity'}
      </button>

      {error && <p className={styles.error}>Error: {error}</p>}

      {jsonData && <p className={styles.error}>{jsonData}</p>}

      {prediction !== null && (
        <div className={styles.result}>
          <h3>Prediction Result:</h3>
          <p>The predicted Osteoarthritis grade is: <span>{prediction}</span></p>
        </div>
      )}
    </div>
  );
}