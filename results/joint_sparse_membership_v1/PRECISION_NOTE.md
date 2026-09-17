Copy scores equal the base scores bit for bit. Independent-noise scores differ
by at most1.3322676295501878e-15 from the base due to floating-point summation
with15 appended zero-weight columns. Both headline metrics and paired metric
differences are exactly equal; no quality difference is hidden by rounding.
"Identical scores" in the next-stage motivation means mathematical equality
up to this floating-point precision, not a bitwise guarantee for noise inputs.
