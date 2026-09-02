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
cat >   hydro19.in  << EOF 
&CONTROL
  calculation = 'vc-relax'
  etot_conv_thr =   5.000000000d-05
  forc_conv_thr =   1.0000000000d-04
  outdir = './out/'
  prefix = 'hydro19'
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
 celldm(1) =      9.01787505
 celldm(2) =      0.99815228
 celldm(3) =      1.37451264
 celldm(5) =     -0.28306605
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
  press=190.0,
  cell_dofree='ibrav',
  press_conv_thr=1.0,  
/
ATOMIC_SPECIES
C      12.0107 C.pbe-n-kjpaw_psl.0.1.UPF
H      1.00794 H.pbe-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS crystal
C             0.7151547686        0.9809410335        0.8738776259
C             0.2848452314        0.4809410335        0.6261223741
C             0.2848452314        0.0190589665        0.1261223741
C             0.7151547686        0.5190589665        0.3738776259
C             0.9128293694        0.7724284153        0.8686378344
C             0.0871706306        0.2724284153        0.6313621656
C             0.0871706306        0.2275715847        0.1313621656
C             0.9128293694        0.7275715847        0.3686378344
C             0.1973426504        0.7901379999        0.9950564519
C             0.8026573496        0.2901379999        0.5049435481
C             0.8026573496        0.2098620001        0.0049435481
C             0.1973426504        0.7098620001        0.4950564519
H             0.4939299708        0.9640065319        0.7750306780
H             0.5060700292        0.4640065319        0.7249693220
H             0.5060700292        0.0359934681        0.2249693220
H             0.4939299708        0.5359934681        0.2750306780
H             0.8513025246        0.5980541789        0.7612405591
H             0.1486974754        0.0980541789        0.7387594409
H             0.1486974754        0.4019458211        0.2387594409
H             0.8513025246        0.9019458211        0.2612405591
H             0.3502688602        0.6272288153        0.9847703462
H             0.6497311398        0.1272288153        0.5152296538
H             0.6497311398        0.3727711847        0.0152296538
H             0.3502688602        0.8727711847        0.4847703462
K_POINTS automatic
6 6 6 0 0 0

EOF
#newfilenametriggerline
$PW_COMMAND < hydro19.in > hydro19.out
$ECHO " done"

# clean TMP_DIR
$ECHO "  cleaning $TMP_DIR...\c"
rm -rf $TMP_DIR/*
$ECHO " done"




