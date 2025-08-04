import { useCallback, useEffect, useRef, useState } from 'react';

const MAX_FILE_MB = 10;
const ACCEPTED_TYPES = ['image/jpeg', 'image/png'];

export default function ImageUploader() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [jsonData, setJsonData] = useState(null);
  const [showDebug, setShowDebug] = useState(false);
  const [hint, setHint] = useState('Tip: Use a clear, centered knee X-ray (PNG or JPG).');

  const dropRef = useRef(null);

  const revokePreview = () => {
    if (preview) URL.revokeObjectURL(preview);
  };

  useEffect(() => {
    return () => revokePreview();
  }, [preview]);

  const validateFile = (file) => {
    if (!file) return 'No file selected.';
    if (!ACCEPTED_TYPES.includes(file.type)) {
      return 'Unsupported file type. Please upload a PNG or JPG image.';
    }
    const sizeMb = file.size / (1024 * 1024);
    if (sizeMb > MAX_FILE_MB) {
      return `File is too large (${sizeMb.toFixed(1)}MB). Max allowed is ${MAX_FILE_MB}MB.`;
    }
    return null;
  };

  const handleSetFile = (file) => {
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      setSelectedFile(null);
      setPrediction(null);
      revokePreview();
      setPreview(null);
      return;
    }
    setError(null);
    setSelectedFile(file);
    revokePreview();
    setPreview(URL.createObjectURL(file));
    setPrediction(null);
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    handleSetFile(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files?.[0];
    handleSetFile(file);
    dropRef.current?.classList.remove('ring-2', 'ring-indigo-500', 'bg-indigo-950/30');
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropRef.current?.classList.add('ring-2', 'ring-indigo-500', 'bg-indigo-950/30');
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropRef.current?.classList.remove('ring-2', 'ring-indigo-500', 'bg-indigo-950/30');
  };

  const parseErrorFromResponse = async (response) => {
    try {
      const text = await response.text();
      const parsed = JSON.parse(text);
      return parsed?.error || parsed?.message || `Request failed (${response.status}).`;
    } catch {
      return `Request failed (${response.status}).`;
    }
  };

  const handlePredict = useCallback(async () => {
    if (!selectedFile) {
      setError('Please select an X-ray image first.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setJsonData(null);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      let data = null;
      try {
        data = await response.json();
      } catch {
        // handled below if !ok
      }

      if (!response.ok) {
        const serverMsg = data?.error || data?.message || (await parseErrorFromResponse(response));
        setPrediction(null);
        setJsonData(data || serverMsg);
        throw new Error(serverMsg);
      }

      setJsonData(data);
      setPrediction(data?.predicted_grade ?? null);
      if (data?.predicted_grade == null) {
        setHint('Prediction returned without a grade. Please try another image or contact support.');
      } else {
        setHint('Success! Review the result below. You can try another image.');
      }
    } catch (err) {
      const fallback = err?.message || 'An unexpected error occurred. Please try again.';
      setError(fallback);
      setPrediction(null);
      if (!jsonData) setJsonData({ error: fallback });
    } finally {
      setIsLoading(false);
    }
  }, [selectedFile, jsonData]);

  const handleReset = () => {
    setSelectedFile(null);
    setPrediction(null);
    setIsLoading(false);
    setError(null);
    setJsonData(null);
    setShowDebug(false);
    revokePreview();
    setPreview(null);
    setHint('Tip: Use a clear, centered knee X-ray (PNG or JPG).');
  };

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 py-12 px-4 sm:px-6 lg:px-8 text-slate-100">
      <div className="mx-auto max-w-2xl">
        <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-slate-900/60 shadow-2xl backdrop-blur-xl">
          <div className="pointer-events-none absolute inset-0 rounded-2xl bg-[radial-gradient(1200px_circle_at_20%_-10%,rgba(79,70,229,0.15),transparent_60%),radial-gradient(900px_circle_at_90%_10%,rgba(14,165,233,0.12),transparent_55%)]" />
          <div className="relative p-8 sm:p-10">
            <div className="mb-6">
              <h2 className="text-3xl font-semibold tracking-tight text-white">
                Knee Osteoarthritis Severity Classification
              </h2>
              <p className="mt-2 text-slate-300">
                Upload a knee X-ray image to predict the severity grade (0, 1, or 2).
              </p>
            </div>

            <div
              ref={dropRef}
              className="group relative mt-6 flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-700 p-8 text-center transition-all duration-200 hover:border-indigo-500/70 hover:bg-indigo-950/30"
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              aria-label="Drag and drop area"
            >
              <div className="inline-flex h-12 w-12 items-center justify-center rounded-full bg-indigo-500/10 text-indigo-400 ring-1 ring-inset ring-indigo-500/20">
                <svg className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 15l4-4a2 2 0 012.828 0L15 16m0 0l2-2a2 2 0 012.828 0L21 15m-6-6h.01M3 7h18a2 2 0 012 2v8a2 2 0 01-2 2H3a2 2 0 01-2-2V9a2 2 0 012-2z" />
                </svg>
              </div>

              <p className="mt-4 text-slate-200">Drag & drop an image here, or</p>

              <label className="mt-3 inline-flex cursor-pointer items-center gap-2 rounded-lg bg-gradient-to-r from-indigo-600 to-violet-600 px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-indigo-900/30 transition-transform duration-150 hover:scale-[1.02] focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-slate-900">
                <input
                  type="file"
                  onChange={handleFileChange}
                  accept="image/png, image/jpeg"
                  className="sr-only"
                  aria-label="Select image"
                />
                Browse files
                <span className="pointer-events-none inline-block h-4 w-px bg-white/25" />
                <span className="text-white/80">PNG/JPG</span>
              </label>

              <p className="mt-3 text-sm text-slate-400">{hint}</p>
            </div>

            {preview && (
              <div className="mt-8">
                <div className="relative inline-block">
                  <img
                    src={preview}
                    alt="X-ray preview"
                    className="h-44 w-44 rounded-xl object-cover shadow-xl ring-1 ring-slate-800 transition-transform duration-200 hover:scale-[1.015]"
                  />
                  <div className="pointer-events-none absolute inset-0 rounded-xl ring-1 ring-inset ring-white/5" />
                  <button
                    onClick={handleReset}
                    className="absolute -top-2 -right-2 inline-flex h-8 w-8 items-center justify-center rounded-full bg-slate-900/90 text-slate-200 shadow-sm ring-1 ring-slate-800 transition-colors hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-slate-900"
                    aria-label="Reset selection"
                  >
                    <svg className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 6.75l12.75 12.75M17.25 6.75L4.5 19.5" />
                    </svg>
                  </button>
                </div>
              </div>
            )}

            <div className="mt-8">
              <button
                onClick={handlePredict}
                disabled={isLoading || !selectedFile}
                className={`relative inline-flex items-center justify-center overflow-hidden rounded-lg px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-indigo-900/30 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-slate-900 transition-all
                ${
                  isLoading || !selectedFile
                    ? 'bg-indigo-800/60 cursor-not-allowed text-indigo-200'
                    : 'bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-600'
                }`}
                aria-busy={isLoading}
                aria-disabled={isLoading || !selectedFile}
              >
                {!isLoading && (
                  <span className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-200 [background:radial-gradient(60%_60%_at_50%_-20%,rgba(255,255,255,0.18),rgba(255,255,255,0))] hover:opacity-100" />
                )}
                {isLoading ? 'Analyzing image…' : 'Predict Severity'}
              </button>
              {isLoading && (
                <p className="mt-2 animate-pulse text-sm text-slate-400">
                  Processing may take a few seconds…
                </p>
              )}
            </div>

            {error && (
              <div className="mt-6 rounded-lg border border-red-900/50 bg-red-950/50 p-4 text-sm text-red-200 shadow-sm" role="alert">
                <div className="flex items-start gap-2">
                  <svg className="mt-0.5 h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M18 10A8 8 0 11 2 10a8 8 0 0116 0zm-7-4a1 1 0 10-2 0v4a1 1 0 002 0V6zm-1 8a1.5 1.5 0 100-3 1.5 1.5 0 000 3z" clipRule="evenodd" />
                  </svg>
                  <div>
                    <strong className="font-semibold">Error:</strong> {error}
                  </div>
                </div>
              </div>
            )}

            {prediction !== null && (
              <div className="mt-6 rounded-xl border border-slate-800 bg-slate-900/70 p-5 shadow-sm backdrop-blur-sm" role="status">
                <h3 className="text-lg font-semibold text-white">Prediction Result</h3>
                <p className="mt-1 text-slate-200">
                  The predicted Osteoarthritis grade is:{' '}
                  <span className="inline-flex items-center rounded-md bg-emerald-900/40 px-2 py-0.5 font-semibold text-emerald-300 ring-1 ring-inset ring-emerald-800/60">
                    {prediction}
                  </span>
                </p>
              </div>
            )}

            <div className="mt-8">
              <button
                onClick={() => setShowDebug((s) => !s)}
                className="inline-flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900 px-4 py-2.5 text-sm font-medium text-slate-200 shadow-sm transition-colors hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-slate-900"
                aria-expanded={showDebug}
                aria-controls="debug-json"
              >
                <svg className="h-4 w-4 text-slate-400" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
                </svg>
                {showDebug ? 'Hide debug' : 'Show debug'}
              </button>
              {showDebug && (
                <pre
                  id="debug-json"
                  className="mt-3 max-h-72 overflow-auto rounded-lg bg-slate-950 p-4 text-xs leading-relaxed text-slate-200 ring-1 ring-slate-800"
                >
                  {typeof jsonData === 'string' ? jsonData : JSON.stringify(jsonData, null, 2)}
                </pre>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}