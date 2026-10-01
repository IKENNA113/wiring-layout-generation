# Graph-Constrained Wiring Layout Generation from Two-Dimensional Electrical Drawings

Code accompanying a manuscript currently under double-anonymized review.
Author information will be added after the review process.

The framework turns reviewed symbol detections on a two-dimensional electrical
lighting plan into a wiring layout in three stages:

1. **Hierarchy construction.** Components are organised into a
   panelboard–switch–fixture graph using the tier map in
   `code/rule_based_class_mapping.json` (16 wiring-eligible classes).
2. **Fixture sequencing.** Within each switch cluster, fixture order is solved as
   an open-path travelling salesman problem (nearest-neighbour initialisation,
   2-opt, Or-opt, simulated annealing, final 2-opt).
3. **Obstacle-aware routing.** Each ordered connection is routed by a
   four-directional A* search with connection-point selection, a turn penalty
   and a bend limit, around component bounding boxes.

## Repository structure

```
run_wiring.py                     Entry point (validates inputs, runs the router)
requirements.txt                  Python dependencies
code/custom_astar.py              Stages 1-3: hierarchy, TSP sequencing, A* routing
code/rule_based_class_mapping.json  Class-to-tier map (1 = panelboard, 2 = switch, 3 = fixture)
code/create_Overlaping_patched.py Tiling of large plans into 640 x 640 patches (50% overlap)
code/Reconstructiing_less.py      Reassembly of patch detections with non-maximum suppression
examples/graph_1_1.png            Example output: optimised sequence for one switch cluster
```

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.9 or later is recommended. The router opens an OpenCV window, so run it
on a machine with a desktop display.

## Input format

Create a working directory with an `input/` folder holding **one** plan:

```
workdir/
  input/
    PLAN.png    floor plan image (resized internally to 11,520 x 8,320 px)
    PLAN.txt    reviewed detections, one per line, in YOLO format:
                class_id  x_centre  y_centre  width  height   (normalised 0-1)
```

Class IDs follow `code/rule_based_class_mapping.json`. Classes not in the map are
ignored.

## Running

```bash
python run_wiring.py --workdir path/to/workdir            # default run
python run_wiring.py --workdir path/to/workdir --seed 0   # fixed random seed
```

Outputs are written to `workdir/output/`: the routed plan image, an obstacle
map, per-cluster sequencing graphs and logs (`output/graphs/`), and a text report
with wire lengths and timings.

## Main parameters (in `code/custom_astar.py`)

| Parameter | Default | Meaning |
|---|---|---|
| `SCALE` | 10 | Routing grid cell size in pixels (s) |
| `TURN_PENALTY` | 5 | Extra cost per direction change in A* (lambda) |
| `max_bends` | 50 | Bend limit per routed segment (B_max) |
| `T`, `alpha`, `T_min`, `iterations_per_temp` | 1000, 0.995, 0.1, 100 | Simulated annealing schedule |

The sensitivity study in the paper varies `SCALE`, `TURN_PENALTY`/`max_bends` and
the annealing schedule.

## Data availability

The evaluation plans are commercial drawings provided by an industry partner
under confidentiality terms and cannot be redistributed. The code runs on any
plan image with detections in the format above.

## Licence

MIT, see `LICENSE`.
