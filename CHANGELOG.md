# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0]

A full rebuild. Version 1 could not start on current Python, and its published
accuracy figures were not measured on held-out data.

### Fixed — correctness

- **Pest accuracy was measured on leaked data.** The shipped dataset held 3,001
  training and 500 test files but only 661 and 443 distinct images, and 390 of
  the 443 distinct test images (88%) were byte-identical copies of training
  images. `ml/prepare_dataset.py` now de-duplicates by content hash, drops
  images filed under more than one label, and writes a clean stratified split
  (490 train / 103 validation / 103 test).
- **Inference used a different pixel scale than training.** Training rescaled by
  `1/255`; the serving path did not, so the deployed model received 0–255 inputs
  it had never seen. Normalisation is now a `Rescaling` layer inside the saved
  model, making the two paths structurally identical.
- **Nitrogen advice was inverted.** "N is high" recommended manure, coffee
  grounds and nitrogen-fixing plants, which *raise* soil nitrogen; "N is low"
  led with sawdust, nitrogen-hungry crops and leaching, which *lower* it. Every
  nitrogen tip has been re-filed under the condition it treats. Phosphorus and
  potassium were already correct.
- **Malathion was recommended for earthworms.** Earthworms are beneficial soil
  organisms; treating them works against the project's goal of reducing soil
  degradation. Earthworm is now flagged beneficial with no pesticide shown.
- **The crop → fertilizer hand-off raised `IndexError`.** The classifier predicts
  `pigeonpeas`; `Crop_NPK.csv` spells it `pigeonpea`. Aliased, with a test
  asserting every predictable crop resolves to NPK data.
- **Features were unscaled.** SVC and KNN saw N spanning 0–140 against rainfall
  spanning 20–300. Scaling now lives inside the saved pipeline.
- **Model selection scored the test set.** `cross_val_score` ran on the held-out
  data, and `"Voting Score % d" % score` integer-formatted a float, printing `0`
  every run. Selection now uses 5-fold CV on the training split; the held-out set
  is scored once.
- **`validation_steps=6500`** against ~16 batches looped the validation generator
  hundreds of times per epoch.
- **Static asset paths contained a space.** `static/img/pesticide/stem borer`
  produced `src=".../stem borer/cartop.jpg"`, an invalid URL. Renamed to
  `stem_borer`.
- **Asset paths broke once the package was installed.** The project root was
  found by walking up from `__file__`, which only holds while the package sits
  in `src/`. Installed into `site-packages` -- the container layout -- the walk
  landed in the virtualenv and startup failed on a missing `static` directory.
  Resolution now prefers an explicit `CROPCRAFT_PROJECT_ROOT`, then the source
  layout, then the working directory.
- **The container image omitted TensorFlow.** The Dockerfile installed base
  dependencies only, so pest identification raised `ImportError` at runtime. The
  image now carries the `ml` extra, and a missing TensorFlow returns a 503 that
  says what to install rather than a traceback.

### Fixed — security

- **Arbitrary file write via upload.** `os.path.join('static/user uploaded',
  file.filename)` trusted a client-supplied filename, so `..\..\app.py` escaped
  the upload directory. There was no type check, no size limit, and uploads were
  retained indefinitely. Uploads are now decoded in memory and never written to
  disk; the type is confirmed by decoding the bytes rather than trusting the
  declared content type; oversized bodies are abandoned mid-stream.
- **`app.run(debug=True)`** exposed the Werkzeug debugger — remote code execution
  on any network-reachable deployment.
- **Raw exception text was returned to the browser** by
  `except Exception as e: return str(e)`.
- **Unvalidated numeric input.** `int(request.form['nitrogen'])` raised
  `ValueError` and returned a 500 with a stack trace. Pydantic now rejects bad
  input with a 422 and a readable page.
- **Autoescaping was disabled** for advisory content pushed through `Markup()`.
  Advice is now structured data rendered by an autoescaping template.
- **jQuery 1.11.2** (CVE-2020-11022, CVE-2019-11358, CVE-2015-9251) and
  **Bootstrap 3.3.5**, both long past end of life, removed.
- **Google Fonts loaded over plain `http://`**, blocked as mixed content on any
  HTTPS deployment. All assets are vendored and same-origin.

### Added

- FastAPI application with a JSON API alongside the HTML forms, and OpenAPI docs
  at `/docs`.
- `GET /api/metrics` publishing held-out accuracy for both models, with a test
  asserting the README does not drift from the artifacts.
- `GET /healthz` reporting `degraded` when a model artifact is missing.
- Security response headers, including a Content-Security-Policy that forbids
  inline script. No template contains an inline `<script>` or event handler.
- Transfer learning (MobileNetV2, ImageNet) for pest identification, replacing a
  three-layer CNN trained from scratch on ~70 images per class.
- Early stopping and learning-rate reduction on validation loss, replacing a
  fixed 100 epochs.
- Metadata saved alongside each model — feature order, class list, metrics and
  library versions — so serving code cannot silently reorder inputs.
- Test suite, GitHub Actions CI that rebuilds both models from the committed
  datasets, and a multi-stage Dockerfile running as a non-root user.
- `LICENSE`, `.env.example` and this changelog.
- A scoped Content-Security-Policy exception for `/docs` and `/redoc`, which
  FastAPI serves from a CDN with a self-bootstrapping inline script; the strict
  application policy would otherwise blank them out.

### Changed

- Python 3.7 → 3.13; Flask → FastAPI; Keras 2 → Keras 3 / TensorFlow 2.21;
  Bootstrap 3 + jQuery → Bootstrap 5 + HTMX and ~70 lines of vanilla JavaScript.
- Every filesystem path derives from the repository root. Version 1 hardcoded
  `E:\extra\crop\` and could run on exactly one machine.
- Ten near-identical pest templates collapsed into one data-driven page.
- The fertilizer crop dropdown is generated from `Crop_NPK.csv` instead of being
  hardcoded in the template, so the two cannot drift apart.
- Routes are paths (`/crop`) rather than `.html` filenames.
- Model artifacts are build outputs and are no longer committed; the datasets are
  committed so both models reproduce from a clean checkout.

### Removed

- `app.py`, `orig`, `cnn_model.py`, `crop_model.py`, `utils/fertilizer.py`,
  `runtime.txt` (pinning Python 3.7.10, end of life June 2023), the committed
  `.idea/` directory, `__pycache__` artifacts, and a `.docx` in `templates/`.

## [1.0.0]

Initial release: Flask application with crop recommendation via a 13-estimator
soft-voting ensemble, dictionary-based fertilizer advice, and pest
identification via a three-layer CNN.
