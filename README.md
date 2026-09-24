# FootVision

**Turning broadcast football footage into structured spatial data — detection, tracking, unsupervised team identity, and pitch-plane projection, built from four composable computer-vision stages.**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-FFD21E?style=for-the-badge&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Roboflow](https://img.shields.io/badge/Roboflow-6706CE?style=for-the-badge&logo=roboflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

---

## Abstract

Point a fixed broadcast camera at a football pitch and you get 25 frames a second of pixels — no notion of "player," "team," or "position on the pitch" exists in that stream until you impose it. FootVision is a compact pipeline that imposes exactly that structure, in four stages, each solving a narrower and more tractable problem than the one before it:

1. **Detect** — locate the ball, players, goalkeepers, and referees as bounding boxes.
2. **Track** — give each moving object a temporally-consistent identity across frames.
3. **Classify** — assign each player to one of two teams, *without ever being told what the team colors are*.
4. **Project** — map every detected object from pixel space onto a metric, top-down pitch coordinate system, via homography.

The interesting part isn't any single stage — off-the-shelf object detectors and `cv2.findHomography` are not novel. The interesting part is stage 3: how do you separate two teams when you have zero labels, arbitrary kit colors, and a background (grass) that visually contaminates every crop you extract? The naive answer — average the pixel color inside each box — fails for reasons worth understanding precisely, and the fix (semantic embeddings + unsupervised clustering) is the conceptual core of this project.

---

## Pipeline overview

```
raw video
   │
   ▼
┌─────────────────────┐     RF-DETR / YOLO-style detector
│  1. Detection        │  →  boxes: {ball, goalkeeper, player, referee}
└─────────────────────┘
   │
   ▼
┌─────────────────────┐     ByteTrack (motion + IoU association)
│  2. Tracking          │  →  persistent tracker_id per object
└─────────────────────┘
   │
   ▼
┌─────────────────────┐     SigLIP embeddings → UMAP → KMeans(k=2)
│  3. Team classification │ →  class_id ∈ {0, 1} per player
└─────────────────────┘
   │
   ▼
┌─────────────────────┐     Pitch keypoint detector + homography
│  4. Pitch projection  │  →  (x, y) in real-world pitch coordinates
└─────────────────────┘
   │
   ▼
tactical top-down view (Voronoi control diagram)
```

---

## Stage 1 — Object detection

**Model:** `football-players-detection-3zvbc/11` (Roboflow Universe, served via the `inference` SDK's `get_model()`, which downloads and caches ONNX weights locally on first call — subsequent calls run on-device, not over the network).

Four classes, fixed at training time:

| `class_id` | class        |
|-----------:|---------------|
| 0          | ball          |
| 1          | goalkeeper    |
| 2          | player        |
| 3          | referee       |

Inference is run per-frame at `confidence=0.3`, deliberately permissive — the small, fast-moving ball is easy to miss, so the threshold is tuned to favor recall over precision at this stage, with cleanup deferred to NMS.

**Detail worth flagging:** `all_detections.with_nms(threshold=0.5, class_agnostic=True)` is applied to everything *except* the ball — class-agnostic NMS suppresses overlapping boxes regardless of predicted class, which matters when a goalkeeper and a nearby player's boxes overlap heavily. The ball is deliberately excluded from this NMS pass and instead has its box **expanded** by 10px (`sv.pad_boxes`) — a small, fast-moving object benefits from a slightly generous box rather than aggressive suppression.

Visualization is handled by `supervision`'s annotator zoo: `EllipseAnnotator` for players/referees (a ground-anchored ellipse reads better than a box when bodies overlap), `TriangleAnnotator` for the ball (a small marker over a tiny object), `LabelAnnotator` for tracker IDs, all driven off a shared `sv.ColorPalette` indexed by `class_id`.

---

## Stage 2 — Multi-object tracking

**Method:** `sv.ByteTrack()`, `supervision`'s implementation of the [ByteTrack](https://arxiv.org/abs/2110.06864) association algorithm.

Detection alone gives you *where* things are in a single frame; it says nothing about *which* box in frame *t+1* corresponds to which box in frame *t*. ByteTrack solves this by associating detections across frames using both motion (Kalman-filter-predicted position) and appearance overlap (IoU) — critically, it also recovers low-confidence detections that would otherwise be discarded, re-associating them with existing tracks rather than dropping them, which reduces broken tracks during occlusion (e.g. a player briefly hidden behind another).

One deliberate design choice: `all_detections.class_id = all_detections.class_id - 1` before tracking, shifting `{goalkeeper: 1, player: 2, referee: 3}` down to `{0, 1, 2}` — the ball (class 0) is tracked and annotated separately and excluded from this remapping entirely, so the tracker never has to reason about an object an order of magnitude smaller and faster than everything else it's tracking.

---

## Stage 3 — Unsupervised team classification

This is the stage worth slowing down for.

### Why the obvious approach fails

The intuitive method: crop each player's bounding box, average the pixel colors inside it, and cluster or threshold on that average. It fails for three compounding reasons:

1. **Background contamination.** A bounding box is a rectangle; a player's body is not. Every crop contains a variable amount of grass, pitch lines, and adjacent players bleeding in at the edges — and critically, *every team's crops sit on the same green pitch*, which biases both teams' average colors toward green in the same direction, actively compressing the separation between them rather than adding neutral noise.
2. **Pose variance.** The fraction of the box that's actually jersey — versus shorts, skin, shadow — changes with every stride, turn, and occlusion.
3. **Illumination variance.** A player in a shaded region of the pitch and one in direct sun can register the same jersey as visibly different average RGB values.

### The fix: embeddings, not raw color

Instead of averaging pixels, each crop is passed through **SigLIP** (`google/siglip-base-patch16-224`, from *"Sigmoid Loss for Language-Image Pre-Training"*), a vision-language model whose vision tower produces a semantically rich representation — one that's been trained to be robust to exactly the nuisance variation (pose, lighting, partial occlusion) that breaks raw pixel averaging. Concretely:

```python
inputs = processor(images=batch, return_tensors='pt')
outputs = model(**inputs)
embedding = torch.mean(outputs.last_hidden_state, dim=1)  # (batch, 768)
```

Crops are sampled at a **stride of 30 frames** (`extract_crops`) rather than exhaustively — the classifier only needs a representative sample of jersey appearances across the match, not every frame, and this keeps embedding extraction tractable. Each crop yields a single 768-dimensional vector (SigLIP-base's hidden size) via mean-pooling over the patch-token sequence.

### Dimensionality reduction + clustering

- **UMAP** (`n_components=3`) projects the 768-dim embeddings down to 3 dimensions, preserving local neighborhood structure — crops that are semantically similar in the original space stay close together after reduction.
- **KMeans** (`n_clusters=2`) partitions the reduced embeddings into two groups.

No labels are ever provided — the two clusters emerge purely from the geometry of the embedding space. `predict()` on the fitted `KMeans`/`UMAP` pair is what turns a *new* crop into a team assignment at inference time, via `sports.common.team.TeamClassifier`, which wraps this exact embed → reduce → cluster procedure into a `.fit(crops)` / `.predict(crops)` interface.

### Resolving the goalkeeper

Goalkeepers are excluded from the clustering step — their kits are typically visually distinct from *both* outfield teams by design, so clustering them alongside outfield players would corrupt the two-cluster structure. Instead, `resolve_goalkeeper_team_id()` assigns each goalkeeper to whichever team's *spatial centroid* (mean pitch position of that team's outfield players, in image coordinates) is nearer:

```python
dist_0 = ||goalkeeper_xy - team_0_centroid||
dist_1 = ||goalkeeper_xy - team_1_centroid||
team = 0 if dist_0 < dist_1 else 1
```

A simple but effective heuristic: goalkeepers spend the overwhelming majority of match time near their own goal, closer to their own side's average position than the opposition's.

---

## Stage 4 — Pitch keypoint detection & homography

**Model:** `football-field-detection-f07vi/14` — a second Roboflow model, trained to detect pitch **keypoints**: line intersections, penalty box corners, center circle points, and other fixed structural landmarks defined in `sports.configs.soccer.SoccerPitchConfiguration`.

A single broadcast frame only shows a *perspective-distorted* view of the pitch — parallel lines converge, distances near the camera look larger than distances far away. To reason about real spatial relationships (distances, territorial control, positioning), detections need to be re-expressed in a flat, metric, top-down coordinate system — the actual 105×68m pitch.

This is a classic **homography** problem: given ≥4 point correspondences between two planes (here: detected keypoints in the image vs. their known coordinates in `CONFIG.vertices`), a single 3×3 projective transformation matrix maps any point from one plane to the other.

```python
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.m, _ = cv2.findHomography(source.astype(np.float32), target.astype(np.float32))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        points = points.reshape(-1, 1, 2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.m)
        return points.reshape(-1, 2).astype(np.float32)
```

Only keypoints above `confidence > 0.5` are used as correspondences — noisy or occluded keypoint detections are filtered before the homography is computed, since a degenerate or noisy point set produces an unstable transform. The transform is used **bidirectionally**: pitch→camera to overlay reference lines onto the broadcast frame for visual sanity-checking (Cell 38), and camera→pitch (the inverse direction) to actually project every player, goalkeeper, referee, and ball position into pitch coordinates for tactical analysis (Cell 41).

Anchor point matters here: player positions are projected using `sv.Position.BOTTOM_CENTER` — the point where a player's feet meet the pitch — rather than the box centroid, since that's the point that actually lies on the pitch plane the homography was computed for.

---

## Output: tactical territorial control (Voronoi diagram)

With every player's pitch-plane `(x, y)` coordinate and team assignment resolved, `sports.annotators.soccer.draw_pitch_voronoi_diagram` partitions the pitch into regions of nearest-player dominance per team — a Voronoi tessellation seeded by each team's player positions, rendered on the flat pitch diagram. This is the payoff of the whole pipeline: raw pixels → structured, per-team spatial control, visualized on a diagram that has nothing to do with camera angle anymore.

---

## Data

Five broadcast clips (`0bfacc_0.mp4`, `2e57b9_0.mp4`, `08fd33_0.mp4`, `573e61_0.mp4`, `121364_0.mp4`), pulled from a public Google Drive folder via `gdown`. No dataset curation or annotation was performed as part of this notebook — both detection models are pretrained, sourced directly from Roboflow Universe.

---

## Known limitations (documented honestly, not glossed over)

A research-style writeup should be honest about the seams. This notebook has a few:

- **Cell 6 (`annotate_football_video`) is dead code.** The annotators are invoked without ever passing `detections` — the function runs without error but draws nothing. Cell 7 reimplements the same logic correctly; Cell 6 was never fixed or removed.
- **GPU is assumed, not detected, in one place.** Cell 3 hardcodes `ONNXRUNTIME_EXECUTION_PROVIDERS=[CUDAExecutionProvider]`, which will silently fail to provide any speedup (or error, depending on onnxruntime's fallback behavior) on a CPU-only runtime — it isn't conditioned on `torch.cuda.is_available()` the way the SigLIP device selection later in the notebook is.
- **Ball tracking is present but disabled.** Cell 8 and Cell 27 both contain a commented-out line (`ball_detections = tracker.update_with_detections(...)`) — the ball is detected and drawn per-frame but never assigned a persistent tracker identity, meaning no ball trajectory can be reconstructed across frames as written.
- **Cell 40 is an orphaned fragment** (`SOURCE_VIDEO_PATH + SOURCE_VIDEO_PATH`, a no-op string concatenation with no assignment) — leftover exploratory cell, not part of the functional pipeline.
- **API key retrieval is Colab-specific.** `google.colab.userdata.get(...)` ties credential loading to the Colab runtime; it isn't portable to a local or server environment without substituting an environment-variable-based approach.
- **No confidence/IoU sweep.** Detection confidence (`0.3`) and NMS threshold (`0.5`) are fixed constants, not tuned or validated against held-out footage — reasonable defaults, not derived values.

---

## Tech stack

| Purpose | Tool |
|---|---|
| Object detection inference | [Roboflow `inference` SDK](https://github.com/roboflow/inference) |
| Detection/tracking primitives, annotators, video I/O | [`supervision`](https://github.com/roboflow/supervision) |
| Multi-object tracking | ByteTrack (via `supervision`) |
| Embedding model | [SigLIP](https://arxiv.org/abs/2303.15343) (`transformers`) |
| Dimensionality reduction | [UMAP](https://umap-learn.readthedocs.io/) |
| Clustering | scikit-learn `KMeans` |
| Homography / perspective transforms | OpenCV (`cv2.findHomography`, `cv2.perspectiveTransform`) |
| Pitch geometry, tactical drawing utilities | [`roboflow/sports`](https://github.com/roboflow/sports) |
| Tensor ops / numerics | PyTorch, NumPy |

---

## References

- Zhang et al., *"ByteTrack: Multi-Object Tracking by Associating Every Detection Box"*, 2021 — [arXiv:2110.06864](https://arxiv.org/abs/2110.06864)
- Zhai et al., *"Sigmoid Loss for Language Image Pre-Training"* (SigLIP), 2023 — [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)
- McInnes, Healy, Melville, *"UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction"*, 2018 — [arXiv:1802.03426](https://arxiv.org/abs/1802.03426)
- Roboflow, [`sports`](https://github.com/roboflow/sports) — pitch configuration, homography utilities, tactical rendering
- Roboflow Universe — [`football-players-detection`](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc), [`football-field-detection`](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi)

---

*This notebook was built by following Roboflow's sports-analytics tutorial series end to end. It is documented here in full technical detail — including its rough edges — as a record of what the pipeline actually does, stage by stage, rather than a polished sales pitch for it.*
