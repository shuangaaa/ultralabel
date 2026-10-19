# Video annotation interface

```bash
python -m pip install -e ".[demo]"
uvicorn demo.backend.app.main:app --reload --port 8000
```

In a second terminal:

```bash
cd demo/frontend
npm install
npm run dev
```

Open the local address shown by Vite. Select a video, pause at the reference frame, and click to place foreground, background, or boundary points. Export saves the video filename, dimensions, reference time, and point coordinates as JSON. Video playback stays in the browser; exported sessions contain annotation metadata.
