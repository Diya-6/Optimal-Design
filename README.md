# D-, A-, E- and I-Optimal Experimental Design

A Python-based framework for constructing and comparing D-, A-, E- and
I-optimal experimental designs using coordinate-exchange optimization.

## Project structure

### Core implementation

  -----------------------------------------------------------------------
  File                                Description
  ----------------------------------- -----------------------------------
  `model.py`                          Defines the response-surface model
                                      and the model-matrix construction
                                      used to represent the experimental
                                      factors and model terms.

  `criteria.py`                       Implements the D-, A-, E- and
                                      I-optimality criteria used to
                                      evaluate experimental designs.

  `coordinate_exchange.py`            Contains the coordinate-exchange
                                      procedure used to improve a design
                                      by updating individual design-point
                                      coordinates.

  `optimization.py`                   Provides the optimization routines
                                      that search for designs according
                                      to the selected optimality
                                      criterion.

  `run_optimization.py`               Main script for running the
                                      optimization process and generating
                                      the optimal designs.
  -----------------------------------------------------------------------

### Optimal design results

The following CSV files store the design points obtained for each
optimality criterion.

  File                     Description
  ------------------------ --------------------------------------------
  `D_optimal_design.csv`   Design points obtained using D-optimality.
  `A_optimal_design.csv`   Design points obtained using A-optimality.
  `E_optimal_design.csv`   Design points obtained using E-optimality.
  `I_optimal_design.csv`   Design points obtained using I-optimality.

### Optimization diagnostics

  -----------------------------------------------------------------------
  File                                Description
  ----------------------------------- -----------------------------------
  `D_convergence.csv`                 Convergence history for the
                                      D-optimal optimization.

  `A_convergence.csv`                 Convergence history for the
                                      A-optimal optimization.

  `E_convergence.csv`                 Convergence history for the
                                      E-optimal optimization.

  `I_convergence.csv`                 Convergence history for the
                                      I-optimal optimization.

  `D_multiple_starts.csv`             Results from multiple initial runs
                                      of the D-optimal optimization.

  `A_multiple_starts.csv`             Results from multiple initial runs
                                      of the A-optimal optimization.

  `E_multiple_starts.csv`             Results from multiple initial runs
                                      of the E-optimal optimization.

  `I_multiple_starts.csv`             Results from multiple initial runs
                                      of the I-optimal optimization.
  -----------------------------------------------------------------------

### Benchmarking and comparison

  -----------------------------------------------------------------------
  File                                Description
  ----------------------------------- -----------------------------------
  `benchmark_designs.py`              Defines or prepares the benchmark
                                      designs used for comparison with
                                      the optimized designs.

  `benchmark_analysis.py`             Evaluates and analyzes the
                                      performance of the optimized and
                                      benchmark designs.

  `compare_results.py`                Compares the results obtained from
                                      the different optimality criteria
                                      and benchmark designs.

  `benchmark_comparison.csv`          Stores the optimality-criterion
                                      values for the optimized designs
                                      and benchmark designs.

  `benchmark_improvements.csv`        Stores calculated improvements or
                                      differences between the compared
                                      designs.

  `criteria_comparison.csv`           Contains a comparison of the
                                      designs across the D-, A-, E- and
                                      I-optimality criteria.

  `normalized_comparison.csv`         Contains normalized criterion
                                      values to support comparison across
                                      measures with different scales.

  `replication_analysis.py`           Analyzes repeated design points and
                                      the associated degrees of freedom.

  `replication_analysis.csv`          Stores the replication counts,
                                      unique settings and
                                      degrees-of-freedom results.

  `validate_designs.py`               Checks the generated designs using
                                      validation measures such as
                                      model-matrix rank and
                                      information-matrix properties.

  `validation_results.csv`            Stores the results of the
                                      design-validation checks.
  -----------------------------------------------------------------------

### Visualizations

  ------------------------------------------------------------------------
  File                                 Description
  ------------------------------------ -----------------------------------
  `visualize_results.py`               Generates visual summaries of the
                                       optimized experimental designs and
                                       their results.

  `visualize_convergence.py`           Generates plots showing how the
                                       optimization criteria change over
                                       iterations.

  `visualize_prediction_variance.py`   Generates prediction-variance
                                       visualizations over the
                                       experimental region.

  `visualize_comparison.py`            Generates comparative
                                       visualizations of the different
                                       optimal designs.

  `experimental_designs.png`           Plot showing the experimental
                                       points for the different optimal
                                       designs.

  `prediction_variance.png`            Plot showing prediction variance
                                       across the design region.

  `comparative_heatmap.png`            Heatmap comparing the performance
                                       of the designs across the selected
                                       criteria.
  ------------------------------------------------------------------------

### Tests and documentation

  -----------------------------------------------------------------------
  File                                Description
  ----------------------------------- -----------------------------------
  `test_model.py`                     Tests the response-surface model
                                      and model-matrix functionality.

  `test_criteria.py`                  Tests the implementation of the
                                      optimality criteria.

  `test_coordinate_exchange.py`       Tests the coordinate-exchange
                                      optimization procedure.

  `README.md`                         Project overview, setup
                                      instructions and documentation of
                                      the files.
  -----------------------------------------------------------------------

### Other

  -----------------------------------------------------------------------
  File                                Description
  ----------------------------------- -----------------------------------
  `optimal_experimental_design.zip`   ZIP archive containing a packaged
                                      copy of the project files.

  -----------------------------------------------------------------------

## Running the project

Run the main optimization script:

``` bash
python run_optimization.py
```

Run the tests:

``` bash
python test_model.py
python test_criteria.py
python test_coordinate_exchange.py
```

Run the validation, benchmarking and analysis scripts as needed:

``` bash
python validate_designs.py
python benchmark_analysis.py
python replication_analysis.py
```

Generate the visualizations using the corresponding `visualize_*.py`
scripts.

## Notes

-   The CSV files contain generated design points, optimization
    histories, comparison metrics or validation results, depending on
    their names.
-   The PNG files are generated visual outputs.
-   The exact columns and output details of each CSV are defined by the
    corresponding Python scripts.
-   The file descriptions above are based on the project file names and
    structure; check the scripts for their exact implementation and
    outputs.
