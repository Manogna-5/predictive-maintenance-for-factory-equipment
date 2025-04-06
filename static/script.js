document.getElementById('uploadForm').addEventListener('submit', async function(e) {
  e.preventDefault();
  const formData = new FormData(this);
  try {
    const response = await fetch('/upload', {
      method: 'POST',
      body: formData
    });
    const result = await response.json();
    document.getElementById('uploadMsg').
::contentReference[oaicite:0]{index=0}