'use client';

import { useState } from 'react';

export function ZoomableImage({ src, alt, caption, width, height }: { src: string; alt: string; caption?: string; width?: number; height?: number }) {
  const [open, setOpen] = useState(false);
  return <>
    <button type="button" className="image-trigger" onClick={() => setOpen(true)} aria-label={`Open image: ${alt}`}>
      <img src={src} alt={alt} width={width} height={height} loading="lazy" decoding="async" />
    </button>
    {open && <div className="image-lightbox" role="dialog" aria-modal="true" aria-label={alt} onClick={() => setOpen(false)}>
      <div className="image-lightbox-panel" onClick={event => event.stopPropagation()}>
        <button type="button" className="image-lightbox-close" onClick={() => setOpen(false)} aria-label="Close image">×</button>
        <img src={src} alt={alt} width={width} height={height} decoding="async" />
        {caption && <p>{caption}</p>}
      </div>
    </div>}
  </>;
}
