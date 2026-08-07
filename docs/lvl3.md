# Level 3 products

Level 3 files contain one row per satellite and one-minute interval. Product
suffixes `_1`, `_2`, and `_3` identify the receiver signal/frequency.

`n_sigphi_1`, `n_sigphi_2`, and `n_sigphi_3` contain the number of valid
detrended phase samples used to calculate sigma-phi. `n_s4_1`, `n_s4_2`, and
`n_s4_3` contain the number of valid SNR samples used to calculate S4.

## Quality flags

Quality flags are binary integers. `0` means the product passes the automatic
quality checks, and `1` means it should not be used without further review.

`sigma_phi_quality_flag_1`, `sigma_phi_quality_flag_2`, and
`sigma_phi_quality_flag_3` are set to `1` when any of the following is true:

- the phase edge/gap mask is set anywhere in the one-minute interval;
- the corresponding `n_sigphi_1/2/3` count is less than `fs * 60 - 10`, where
  `fs` is the detected sampling rate in hertz (exactly 10 dropped samples is
  allowed); or
- the satellite is GLONASS (its PRN begins with `R`).

`s4_quality_flag_1`, `s4_quality_flag_2`, and `s4_quality_flag_3` are set to `1`
when the corresponding `n_s4_1/2/3` count is less than `fs * 60 * 0.8`. A
minute with exactly 80% of its expected samples remains good.

The sigma-phi and S4 flags are independent. For example, GLONASS automatically
fails the sigma-phi check but can still have a good S4 flag.
