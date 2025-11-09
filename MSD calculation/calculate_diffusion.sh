cpptraj.OMP << EOF
parm ../../prepare/rep.c0.pdb
trajin dry.xtc
unwrap scheme tor
trajout dry-tor.xtc
run
clear all
parm ../../prepare/rep.c0.pdb
trajin dry-tor.xtc
calcdiffusion time 2 out msd.dat
run
quit
EOF