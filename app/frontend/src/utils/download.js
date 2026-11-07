/** Offer `blob` as a file download under `name`. */
export function saveBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export const saveText = (text, name, type = 'text/plain') => saveBlob(new Blob([text], { type: `${type};charset=utf-8` }), name);
