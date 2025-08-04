import '../styles/globals.css'

import React from 'react';
import ImageUploader from '../components/ImageUploader';

function MyApp({ Component, pageProps }) {
  return (
    <div>
      <Component {...pageProps} />
      <ImageUploader />
    </div>
  );
}

export default MyApp;