cpptraj <<EOF
parm dry.top
trajin dry.xtc
autoimage
cluster C1 \
        hieragglo epsilon 2.0 linkage epsilonplot eposilonplot.dat \
        rms :1-2 \
        sieve 50 random \
        out cnumvtime.dat \
        sil Sil \
        summary summary.dat \
        info info.dat \
        cpopvtime cpopvtime.agr normframe \
        repout rep repfmt pdb \
        singlerepout singlerep.nc singlerepfmt netcdf \
        avgout Avg avgfmt restart
run
EOF
