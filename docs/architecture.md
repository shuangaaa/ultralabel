# Architecture

`AnnotationPipeline` accepts a frozen feature extractor, an image-guided upsampler, a point tracker, and a contour reconstructor. Components follow abstract interfaces with explicit array shapes.

The reference frame establishes query descriptors. Target frames pass through feature extraction and guided upsampling before point correspondence and contour reconstruction. Results contain trajectories of shape `T × N × 2` and boolean masks of shape `T × H × W`. Coordinates refer to resized frames; `resize_points` maps coordinates between image sizes.

The `quality` and `optimization` packages provide QC descriptor calculations and student optimizer construction. Their parameters are stored separately from the core tracking settings.
