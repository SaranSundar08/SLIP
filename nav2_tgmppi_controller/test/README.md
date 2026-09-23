# Controller regression tests

The registered gtests run against the package libraries, including the normal
compiler flags. Run after building the package:

```sh
colcon test --packages-select nav2_tgmppi_controller --ctest-args -R '^(flow_field_test|optimizer_sampling_test|optimizer_state_test)$' --output-on-failure
```

- `flow_field_test`: body connectivity, Dijkstra distance, direction lookup,
  membrane adjacency, and returned pseudopods cannot cut blocked diagonal
  corners. Open diagonals retain their sqrt(2) distance. These tests concern
  the inflated 2D grid; they do not establish polygon-footprint feasibility.
- `optimizer_sampling_test`: calls the production allocation method and grouped
  update. Covers zero through five spatial slots and two extra slots, tiny and
  non-divisible batches, both allocation policies, rejected slots, retained
  sampling noise, wait/fallback boundaries, mode-local weighted means, group-size
  normalization, switching confirmation, and actual sampling cycles with guidance
  disabled or an unready field. Proposal validity flags are supplied as inputs;
  this suite does not test the geometry that produces those flags.
- `optimizer_state_test`: capacity/state clearing and critic failure reset on retry.

At full assist, equal allocation distributes integer remainder rows in mode order,
then wait, then fallback. Group counts differ by at most one. With fewer samples
than groups, some groups receive no samples; production-sized batches should be
used for experiments. The assist ramp returns unused guided rows to fallback.
Legacy promise-weighted allocation and the uncorrected within-mode free-energy
update retain their existing equations.

The existing `space_time_body_test.cpp` remains a standalone test; its documented
compile/run command is at the top of that file.
