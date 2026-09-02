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
cat >   hydro22.in  << EOF 
&CONTROL
  calculation = 'vc-relax'
  etot_conv_thr =   5.000000000d-05
  forc_conv_thr =   1.0000000000d-04
  outdir = './out/'
  prefix = 'hydro22'
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
 celldm(1) =      8.92935793
 celldm(2) =      0.99861461
 celldm(3) =      1.37190367
 celldm(5) =     -0.27764310
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
  press=220.0,
  cell_dofree='ibrav',
  press_conv_thr=1.0,  
/
ATOMIC_SPECIES
C      12.0107 C.pbe-n-kjpaw_psl.0.1.UPF
H      1.00794 H.pbe-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS crystal
C             0.7134702146        0.9809205761        0.8732429547
C             0.2865297854        0.4809205761        0.6267570453
C             0.2865297854        0.0190794239        0.1267570453
C             0.7134702146        0.5190794239        0.3732429547
C             0.9124407296        0.7703989397        0.8680813512
C             0.0875592704        0.2703989397        0.6319186488
C             0.0875592704        0.2296010603        0.1319186488
C             0.9124407296        0.7296010603        0.3680813512
C             0.1985811474        0.7879620826        0.9951316348
C             0.8014188526        0.2879620826        0.5048683652
C             0.8014188526        0.2120379174        0.0048683652
C             0.1985811474        0.7120379174        0.4951316348
H             0.4911741846        0.9638638598        0.7736932340
H             0.5088258154        0.4638638598        0.7263067660
H             0.5088258154        0.0361361402        0.2263067660
H             0.4911741846        0.5361361402        0.2736932340
H             0.8511731177        0.5947546642        0.7596792588
H             0.1488268823        0.0947546642        0.7403207412
H             0.1488268823        0.4052453358        0.2403207412
H             0.8511731177        0.9052453358        0.2596792588
H             0.3523672384        0.6234241449        0.9842759802
H             0.6476327616        0.1234241449        0.5157240198
H             0.6476327616        0.3765758551        0.0157240198
H             0.3523672384        0.8765758551        0.4842759802
K_POINTS automatic
6 6 6 0 0 0

EOF
#newfilenametriggerline
$PW_COMMAND < hydro22.in > hydro22.out
$ECHO " done"

# clean TMP_DIR
$ECHO "  cleaning $TMP_DIR...\c"
rm -rf $TMP_DIR/*
$ECHO " done"




