#!/bin/sh

#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --constraint=ib


#SBATCH --partition=amilan 
#SBATCH --ntasks=32

#SBATCH --job-name=QEjobBenz18hydro
#SBATCH --output=QEjob.%j.out

module purge

module load gcc
module load openmpi


# check whether echo has the -e option
if test "`echo -e`" = "-e" ; then ECHO=echo ; else ECHO="echo -e" ; fi

$ECHO "example"


# run from directory where this script is


EXAMPLE_DIR=`pwd`

# check whether echo has the -e option
if test "`echo -e`" = "-e" ; then ECHO=echo ; else ECHO="echo -e" ; fi

$ECHO
$ECHO "$EXAMPLE_DIR : starting"
$ECHO


# set the needed environment variables
#. ../../../environment_variables
PREFIX=`cd /projects/rile5166/QEme/qe-7.0; pwd`
# $PREFIX is the source tree.
BIN_DIR=$PREFIX/bin
PSEUDO_DIR=$PREFIX/pseudo
# Beware: everything in $TMP_DIR will be destroyed !
TMP_DIR=$PREFIX/tempdir
PARA_PREFIX=" "
PARA_PREFIX="mpirun -np 32"

# available flags: 
#                  -ni n        number of images        (or -nimage)
#                               (only for NEB; for PHonon, see below)
#                  -nk n        number of pools         (or -npool, -npools)
#                  -nb n        number of band groups   (or -nbgrp,-nband_group)
#                  -nt n        number of task groups   (or -ntg, -ntask_groups)
#                  -nd n        number of processors for linear algebra 
#                                            (or -ndiag, -northo) 

PARA_POSTFIX=" -nk 4 -nd 1 -nb 8 "


PARA_IMAGE_POSTFIX="-ni 32 $PARA_POSTFIX"
PARA_IMAGE_PREFIX="mpirun -np 32"

# In case of mixed MPI / OpenMP parallelization you may want to limit
# the maximum number to OpenMP threads so that the number of threads
# per MPI process times the number of MPI processes equals the number
# of available cores to avoid hyperthreading

export OMP_NUM_THREADS=1

# There should be no need to change anything below this line

LC_ALL=C
export LC_ALL

NETWORK_PSEUDO=http://www.quantum-espresso.org/wp-content/uploads/upf_files/

# wget or curl needed if some PP has to be downloaded from web site
# script wizard will surely find a better way to find what is available
if test "`which curl`" = "" ; then
   if test "`which wget`" = "" ; then
      echo "wget or curl not found: will not be able to download missing PP"
   else
      WGET="wget -O"
      # echo "wget found"
   fi
else
   WGET="curl -o"
   # echo "curl found"
fi

# function to test the exit status of a job
check_failure () {
    # usage: check_failure $?
    if test $1 != 0
    then
        echo "Error condition encountered during test: exit status = $1"
        echo "Aborting"
        exit 1
    fi
}


# required executables and pseudopotentials
BIN_LIST="pw.x"
PSEUDO_LIST="C.pbe-n-rrkjus_psl.0.1.UPF"

$ECHO
$ECHO "  executables directory: $BIN_DIR"
$ECHO "  pseudo directory:      $PSEUDO_DIR"
$ECHO "  temporary directory:   $TMP_DIR"
$ECHO "  checking that needed directories and files exist...\c"

# check for directories
for DIR in "$BIN_DIR" "$PSEUDO_DIR" ; do
    if test ! -d $DIR ; then 
        $ECHO
        $ECHO "ERROR: $DIR not existent or not a directory"
        $ECHO "Aborting"
        exit 1
    fi
done
for DIR in "$TMP_DIR" "$EXAMPLE_DIR/results" ; do
    if test ! -d $DIR ; then
        mkdir $DIR
    fi
done
cd $EXAMPLE_DIR/results

# check for executables
for FILE in $BIN_LIST ; do
    if test ! -x $BIN_DIR/$FILE ; then
        $ECHO
        $ECHO "ERROR: $BIN_DIR/$FILE not existent or not executable"
        $ECHO "Aborting"
        exit 1
    fi
done
# check whether echo has the -e option
if test "`echo -e`" = "-e" ; then ECHO=echo ; else ECHO="echo -e" ; fi

BIN_LIST="pw.x"

# check for directories
for DIR in "$BIN_DIR" "$PSEUDO_DIR" ; do
    if test ! -d $DIR ; then 
        $ECHO
        $ECHO "ERROR: $DIR not existent or not a directory"
        $ECHO "Aborting"
        exit 1
    fi
done
for DIR in "$TMP_DIR" "$EXAMPLE_DIR/results" ; do
    if test ! -d $DIR ; then
        mkdir $DIR
    fi
done
# how to run executables
PW_COMMAND="$PARA_PREFIX $BIN_DIR/pw.x $PARA_POSTFIX"
$ECHO
$ECHO "  running pw.x as: $PW_COMMAND"
$ECHO

# self-consistent calculation
cat >   hydro26.in  << EOF 
&CONTROL
  calculation = 'vc-relax'
  etot_conv_thr =   5.000000000d-05
  forc_conv_thr =   1.0000000000d-04
  outdir = './out/'
  prefix = 'hydro26'
  pseudo_dir = '$PSEUDO_DIR'
  tprnfor = .true.
  tstress = .true.
  verbosity = 'high'
  nstep=100
/
&SYSTEM
  degauss =   1.00000000000d-02
  ecutrho =   6.400000000d+02
  ecutwfc =   8.0000000000d+01
  ibrav = -12
 celldm(1) =      8.82739283
 celldm(2) =      0.99910168
 celldm(3) =      1.36884984
 celldm(5) =     -0.27146349
  nat = 24
  nosym = .false.
  ntyp = 2
  occupations = 'smearing'
  smearing = 'gaussian'
  vdw_corr='grimme-d3'
  dftd3_version = 3
/
&ELECTRONS
  conv_thr =   1.000000000d-08
  electron_maxstep = 80
  mixing_beta =   4.0000000000d-01
/
&ions
/
&cell
  cell_dynamics='bfgs',
  press=260.0,
  cell_dofree='ibrav',
  press_conv_thr=1.0,  
/
ATOMIC_SPECIES
C      12.0107 C.pbe-n-kjpaw_psl.0.1.UPF
H      1.00794 H.pbe-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS crystal
C             0.7114392913        0.9814492270        0.8724469773
C             0.2885607087        0.4814492270        0.6275530227
C             0.2885607087        0.0185507730        0.1275530227
C             0.7114392913        0.5185507730        0.3724469773
C             0.9116250999        0.7684278255        0.8671775584
C             0.0883749001        0.2684278255        0.6328224416
C             0.0883749001        0.2315721745        0.1328224416
C             0.9116250999        0.7315721745        0.3671775584
C             0.1996615317        0.7851910063        0.9950360862
C             0.8003384683        0.2851910063        0.5049639138
C             0.8003384683        0.2148089937        0.0049639138
C             0.1996615317        0.7148089937        0.4950360862
H             0.4879046577        0.9646219531        0.7719866837
H             0.5120953423        0.4646219531        0.7280133163
H             0.5120953423        0.0353780469        0.2280133163
H             0.4879046577        0.5353780469        0.2719866837
H             0.8505460816        0.5918854543        0.7570609809
H             0.1494539184        0.0918854543        0.7429390191
H             0.1494539184        0.4081145457        0.2429390191
H             0.8505460816        0.9081145457        0.2570609809
H             0.3541031290        0.6186593623        0.9830642493
H             0.6458968710        0.1186593623        0.5169357507
H             0.6458968710        0.3813406377        0.0169357507
H             0.3541031290        0.8813406377        0.4830642493
K_POINTS automatic
6 6 6 0 0 0

EOF
#newfilenametriggerline
$PW_COMMAND < hydro26.in > hydro26.out
$ECHO " done"

# clean TMP_DIR
$ECHO "  cleaning $TMP_DIR...\c"
rm -rf $TMP_DIR/*
$ECHO " done"




