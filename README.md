# CropCraft

Crop, fertilizer and pesticide recommendations for Indian farms, from soil readings
and pest photographs.

Enter a soil test and get the crop best matched to it. Enter your NPK against a crop
you have chosen and get specific guidance on closing each nutrient gap. Upload a photo
of the insect eating your field and get an identification with registered treatments.

**Stack:** Python 3.13 · FastAPI · scikit-learn · TensorFlow/Keras 3 · Bootstrap 5 · HTMX

---

## Quick start

```bash
uv venv --python 3.13
uv pip install -e ".[ml,dev]"

# Build the model artifacts (they are not committed -- see "Models" below)
python -m ml.train_crop           # a few seconds
python -m ml.prepare_dataset      # de-duplicates the pest images
python -m ml.train_pest           # ~20 min on CPU

python -m cropcraft --reload      # http://127.0.0.1:8000
```

Interactive API docs are at `/docs`.

With Docker. The image copies `models/` rather than training inside the build, so
run the two training commands above first:

```bash
docker build -t cropcraft .
docker run --rm -p 8000:8000 cropcraft
```

The image carries TensorFlow, which the pest endpoint needs at serving time, so it
is large -- about 3.5 GB. Dropping `--extra ml` from the Dockerfile gives a far
smaller image that still serves crop and fertilizer recommendations; pest
identification then returns 503 explaining what to install, rather than failing.

---

## How well does it actually work?

Both numbers below come from data the models never saw during training or selection.
They are written by the training scripts and served live at `GET /api/metrics`, so
they can be checked rather than taken on trust.

### Crop recommendation — 99.6% accuracy

|                   |                                                        |
| ----------------- | ------------------------------------------------------ |
| Held-out accuracy | **0.9955**                                             |
| Held-out macro F1 | 0.9954                                                 |
| Test set          | 440 rows, 22 crops, stratified, `random_state=42`      |
| Selected model    | GaussianNB, chosen by 5-fold CV on the training split  |

This number is high because `crop_recommendation.csv` is a clean, well-separated
dataset with 100 balanced rows per class — the classes barely overlap. Treat it as
evidence the pipeline is wired correctly, not as evidence that crop choice is a solved
problem. Real soil tests are noisier than this dataset.

### Pest identification — 89.3% accuracy

|                         |                                              |
| ----------------------- | -------------------------------------------- |
| Held-out accuracy       | **0.8932**                                   |
| Held-out top-3 accuracy | 1.0000                                       |
| Test set                | 103 images, 10 classes, no training overlap  |
| Backbone                | MobileNetV2 (ImageNet), fine-tuned           |

Per-class recall is uneven, and the app says so on screen when confidence is low:

| Class                                                 | Recall   |
| ----------------------------------------------------- | -------- |
| armyworm, earthworm, grasshopper, mosquito, stem borer | 1.00     |
| bollworm                                              | 0.93     |
| aphids, mites                                         | 0.89     |
| beetle                                                | 0.78     |
| **sawfly**                                            | **0.40** |

Sawfly is the weak class and should not be trusted without confirmation. The
classifier is also closed-set: it only knows these ten pests and will force anything
else into one of them, so a low-confidence result means "photograph it again", not
"it is probably this".

---

## What changed from version 1

Version 1 could not start on current Python: it imported `Markup` from Flask (removed
in Flask 3.0) and `keras.preprocessing.image` (removed in Keras 3), and loaded models
from hardcoded `E:\extra\crop\` paths. Beyond getting it running, these were the
substantive fixes.

### The pest accuracy was not real

The shipped dataset had 3,001 training files and 500 test files, but only **661 and
443 distinct images** — and **390 of the 443 distinct test images (88%) were
byte-identical copies of training images**. Every reported accuracy was measuring
memorisation.

`ml/prepare_dataset.py` now pools all 3,501 files, drops 2,787 exact duplicates and 18
images filed under more than one label, and writes a fresh stratified split:

```
source files            : 3501
distinct images         :  714
dropped as exact dupes  : 2787
dropped as label-clashes:   18
usable images           :  696   ->  train 490 / val 103 / test 103
```

The honest dataset is about 70 images per class, which is why the classifier now uses
transfer learning rather than training a three-layer CNN from scratch.

### Predictions were silently wrong in production

Training rescaled pixels by `1/255`; inference did not. The deployed model received
0–255 values it had never been trained on. Normalisation is now a `Rescaling` layer
**inside** the saved model, so the training and serving paths cannot diverge.

### The nitrogen advice was inverted

`utils/fertilizer.py` filed the advice back to front. "Your N is high" recommended
manure, coffee grounds and nitrogen-fixing plants — all of which *raise* soil nitrogen.
"Your N is low" led with sawdust, nitrogen-hungry crops and leaching, which *lower* it.
A farmer following it would have made the problem worse. Phosphorus and potassium were
correct. Every nitrogen tip is now filed under the condition it actually treats, and a
regression test pins it.

### It recommended killing earthworms

Version 1 offered malathion for earthworms. Earthworms are beneficial — they improve
soil structure, aeration and nutrient cycling — so treating them works directly against
the project's stated goal of reducing soil degradation. Earthworm is now flagged as
beneficial with no pesticide recommendation.

### Uploads were an arbitrary file write

```python
file_path = os.path.join('static/user uploaded', file.filename)   # 1.x
file.save(file_path)
```

The filename came from the client, so `..\..\app.py` escaped the upload directory.
There was no type check, no size limit, and every upload stayed on disk forever.
Uploads are now decoded in memory and never written anywhere; the type is confirmed by
decoding the bytes, not by trusting the client's header; and oversized bodies are
abandoned mid-stream.

### Other fixes

- `app.run(debug=True)` exposed the Werkzeug debugger — remote code execution if it
  ever faced a network.
- Handlers did `int(request.form['nitrogen'])` directly, so any non-numeric input
  returned a 500 with a stack trace. Pydantic now rejects bad input with a 422 and a
  readable page.
- `except Exception as e: return str(e)` returned raw exception text to the browser.
- The crop model predicted `pigeonpeas`, but `Crop_NPK.csv` spells it `pigeonpea`, so
  that hand-off raised `IndexError`. Aliased, with a test asserting every predictable
  crop resolves to NPK data.
- SVC and KNN were fed unscaled features where N spans 0–140 and rainfall 20–300.
  Scaling now lives inside the pipeline.
- Model selection ran `cross_val_score` on the *test* set, and
  `"Voting Score % d" % score` integer-formatted a float accuracy, so it always printed
  `0`.
- `validation_steps=6500` against ~16 batches of validation data looped the generator
  hundreds of times per epoch.
- jQuery 1.11.2 (CVE-2020-11022, CVE-2019-11358, CVE-2015-9251) and Bootstrap 3.3.5,
  both long EOL, replaced with Bootstrap 5 and ~70 lines of vanilla JS.
- Google Fonts loaded over plain `http://`, blocked as mixed content on any HTTPS
  deploy. All assets are now vendored locally.
- Ten near-identical pest templates collapsed into one data-driven page.
- The fertilizer dropdown was hardcoded in the template and could drift from the CSV;
  it is now generated from the CSV.
- Advice was pushed through `Markup()`, disabling Jinja autoescaping. It is now
  structured data.

---

## Security

- Pest photos are decoded in memory and never written to disk. The type is
  confirmed by decoding the bytes, not by trusting the declared content type,
  and oversized bodies are abandoned mid-stream rather than buffered.
- All input is validated by Pydantic before a handler runs; out-of-range or
  non-numeric values get a 422 and a readable page, never a stack trace.
- Every response carries a Content-Security-Policy that forbids inline script,
  plus `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy`,
  `Permissions-Policy` and `Cross-Origin-Opener-Policy`. No template contains an
  inline `<script>` or event handler, so the policy needs no `unsafe-inline`
  escape hatch.
- Every application asset is vendored and same-origin; no page makes a
  third-party request. The exception is `/docs` and `/redoc`, which FastAPI
  serves from a CDN; those two routes get a scoped, looser policy rather than
  the whole app being relaxed for them.
- The container runs as a non-root user.

---

## Project layout

```
src/cropcraft/
  config.py            Settings; every path derived from the repo root
  main.py              App factory, error handlers
  schemas.py           Pydantic request/response models
  templating.py        Jinja environment
  domain/              Pest and nutrient reference data
  routers/             Page routes; HTML form and JSON API handlers
  services/            Crop, fertilizer and pest logic
ml/
  prepare_dataset.py   De-duplicate images, build clean splits
  train_crop.py        Crop pipeline
  train_pest.py        Pest classifier
tests/                 pytest suite
```

## API

| Method | Path              | Purpose                                          |
| ------ | ----------------- | ------------------------------------------------ |
| `POST` | `/api/crop`       | Recommend a crop from soil and weather readings   |
| `POST` | `/api/fertilizer` | Compare measured NPK against a crop's requirement |
| `POST` | `/api/pesticide`  | Identify a pest from an uploaded image            |
| `GET`  | `/api/crops`      | List predictable crops and NPK reference crops    |
| `GET`  | `/api/metrics`    | Held-out accuracy for both models                 |
| `GET`  | `/healthz`        | Liveness probe; `degraded` if a model is missing  |

```bash
curl -X POST localhost:8000/api/crop -H 'content-type: application/json' \
  -d '{"nitrogen":90,"phosphorous":42,"potassium":43,"temperature":20.9,
       "humidity":82,"ph":6.5,"rainfall":202.9}'
```

## Models

Model artifacts are not committed — they are build outputs, and a pickle written by one
scikit-learn version does not reliably load in the next. The datasets *are* committed,
so `python -m ml.train_crop` and `python -m ml.train_pest` reproduce them from a clean
checkout. CI rebuilds both on every run.

## Development

```bash
pytest                          # test suite
ruff check src ml tests         # lint
ruff format src ml tests        # format
```

## Limitations

- The pest classifier knows ten pests and will force any other insect into one of them.
- Sawfly recall is 0.40; treat sawfly identifications as unconfirmed.
- The crop dataset is clean and separates almost perfectly; expect lower accuracy on
  real soil tests.
- NPK requirements exist for 30 crops, and the crop model predicts 22. The overlap is
  tested, but neither list is exhaustive for Indian agriculture.
- Recommendations are advisory. Confirm dosing against the product label and check with
  a local agricultural extension officer.

## Credits

Built by **Nikesh Walia**. Nutrient requirements from The Fertilizer Association of
India and the Indian Institute of Water Management. Pest images scraped and labelled
for the original project.
