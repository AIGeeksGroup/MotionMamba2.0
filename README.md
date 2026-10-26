# Motion Mamba 2.0 Project Page

Official project page for **Motion Mamba 2.0: Autoregressive Diffusion Mamba for Long-Horizon Humanoid Control**.

The page is a static website built with HTML, CSS, JavaScript, Three.js, and recorded motion data. The Simulation section replays synchronized Viser and MuJoCo motions directly in the browser.

## Local Preview

The Simulation section uses JavaScript modules and binary motion files, so the page must be served over HTTP. Opening `index.html` directly with a `file://` URL will not load the interactive simulation.

From the project directory, start a local server:

```bash
cd /Users/zeyuzhang/MotionMamba2.0
python3 -m http.server 8097 --bind 127.0.0.1
```

Then open:

```text
http://127.0.0.1:8097/
```

Stop the server with `Ctrl+C` in the terminal.

## Deploy with GitHub Pages

The repository is configured as a static site and does not require a build step.

1. Commit and push the website branch:

   ```bash
   cd /Users/zeyuzhang/MotionMamba2.0
   git add index.html README.md simulation real-world-video
   git commit -m "update project page"
   git push origin website
   ```

2. Open the repository on GitHub:

   [AIGeeksGroup/MotionMamba2.0](https://github.com/AIGeeksGroup/MotionMamba2.0)

3. Go to **Settings → Pages**.

4. Under **Build and deployment**, choose **Deploy from a branch**.

5. Select the `website` branch and the `/(root)` directory, then click **Save**.

After deployment finishes, the page will be available at:

```text
https://aigeeksgroup.github.io/MotionMamba2.0/
```

GitHub Pages may take a few minutes to publish a new commit. Refresh the deployment page after the Pages workflow completes.

## Project Structure

```text
MotionMamba2.0/
├── index.html              # Main project page
├── real-world-video/       # Real-world and Three.js videos
└── simulation/             # Interactive simulation runtime and motion data
    ├── assets/             # Unitree G1 model assets
    ├── motions/            # Viser and MuJoCo motion recordings
    ├── src/                # Simulation player source
    ├── vendor/             # Local Three.js and URDF dependencies
    └── manifest.json       # Motion metadata and prompt timelines
```

All website assets use relative paths, so the page works both on localhost and under the GitHub Pages repository subpath.
