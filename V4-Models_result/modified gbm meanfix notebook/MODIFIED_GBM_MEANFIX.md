# Modified GBM meanfix

Same model as original Modified GBM (Markov sign, `|N(μ,σ)|` sizes, additive Q).

Only change: `μ,σ` are chosen so `E[|N(μ,σ)|]` equals the sample mean of `|R|`
in that up/down bucket. Original used `μ = mean(|R|)`, which overstates simulated sizes.

If the half-normal at the sample SD already has mean ≥ sample mean, `μ=0` and `σ` is
shrunk to match the mean.
