using FileIO, JLD2
using Plots

L2_error = load("savedSoln/swe_MMS_L2Error_0.3.jld2", "L2_error")

L2_error = sqrt.(L2_error)
h = 2 ./ [4 8 16 32]'

C3 = 10.0 
C4 = 3.0
C5 = 3.0

expected_rates = [C3*h.^3 C4*h.^4 C5*h.^5]
rate_labels = ["\$h^3\$" "\$h^4\$" "\$h^5\$"]
error_labels = ["\$N^2\$" "\$N^3\$" "\$N^4\$"]

# plot(h, L2_error[2:5,2], lw=5)
plot(h, expected_rates, labels=rate_labels, lw=2, linestyle=:dash)
plot!(h, L2_error[2:5, 2:4], labels=error_labels, 
    lw=3, 
    marker=:auto, 
    markersize=5,
    markerstrokewidth=0.5
)

plot!(xlabel="\$h = \\Delta x = \\Delta y\$",
    ylabel="\$ L_2\$ Error",
    xaxis=:log,
    yaxis=:log,
    legend=:bottomright,
    xguidefontsize=16,
    yguidefontsize=16,
    legendfontsize=12
    )


png("figures/ESconvergenceStudy.png")