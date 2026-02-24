for i in `seq 1 7`;do
cpptraj <<EOF
parm dry.top
trajin filter.xtc
reference ../permutations/ref${i}_final.pdb [refx]
reference ../permutations/ref1_final.pdb [refx1]
rms :MOL ref [refx] out rms-${i}-MOL.dat
rms :1-2 nofit ref [refx1] out rms-${i}-com.dat
run
EOF
done

