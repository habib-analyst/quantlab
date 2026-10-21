# Quantization, explained simply

A neural network is, mostly, a giant pile of numbers called weights.
When the model thinks, it multiplies your words by those numbers,
millions of times per sentence. Storing and multiplying big precise
numbers is expensive, so we ask: how much precision do the numbers
actually need?

## The map analogy

Imagine a paper map where every street corner is marked with GPS
coordinates written to ten decimal places. Lovely precision, enormous
book. Now imagine the same map drawn on a grid, where every corner is
snapped to the nearest grid square. The book is suddenly pocket-sized.
A few corners land in slightly wrong squares, but every journey still
works out almost the same.

That snapping is quantization. The full-precision weight is the exact
coordinate. The grid is a small set of allowed values: 256 of them for
8-bit, 16 of them for 4-bit, or the quirky non-uniform grid of FP8,
which packs more grid lines near zero where most weights live. The
distance between grid lines is called the scale. Small scale, fine grid,
faithful map. Big scale, coarse grid, small book.

Everything in this project is a different way of drawing that grid:

- **Round to nearest (RTN)** just snaps every corner to the closest grid
  square and hopes for the best. Simple, fast to make, and the errors
  are random: some streets shift a little.
- **GPTQ** snaps one street at a time, and after each snap it nudges all
  the not-yet-snapped streets a tiny bit to cancel out the error it just
  made. Like proofreading a letter while writing it, fixing the next
  sentence to compensate for a typo in the last one. Same grid, far fewer
  wrong turns at the end.
- **AWQ** notices that a few streets carry almost all the traffic. Before
  snapping, it stretches the map around those busy streets so they get
  more grid lines, and squashes the empty countryside where nobody goes.
  The stretching is undone during the journey itself, so nothing is
  actually distorted, but the busy streets survive the snapping nearly
  intact.
- **FP8** uses a grid whose lines are dense near zero and sparse far
  away, like a ruler with fine marks in the middle. It also snaps your
  words (the activations), not just the map. On a graphics card, special
  hardware multiplies these snapped numbers directly, which is where the
  real speed lives. On a plain CPU, as this project measures, that
  special hardware is missing, and the snapped arithmetic can end up
  slower than the honest kind. Smaller is not automatically faster; it
  is only faster where the hardware knows the grid.

## Why bother?

Because the book's size decides where the model can live. A model that
needs 2 GB of weights needs a big machine. The same model snapped to a
4-bit grid needs about a quarter of that, plus a small table of scales,
and can suddenly fit on a laptop, a phone, or a cheap server. The whole
game is choosing a grid clever enough that the journeys, the model's
answers, barely change. This project measures, for each way of drawing
the grid: how much smaller the book gets, how wrong the journeys become,
and how long they take.
