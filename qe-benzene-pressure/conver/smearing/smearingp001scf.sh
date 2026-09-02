#!/bin/sh

#SBATCH --nodes=1
#SBATCH --time=01:00:00
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
cat >   smearingp001scf.in  << EOF 
&CONTROL
  calculation = 'scf'
  etot_conv_thr =   5.000000000d-05
  forc_conv_thr =   1.0000000000d-04
  outdir = './out/'
  prefix = 'smearingp001scf'
  pseudo_dir = '$PSEUDO_DIR'
  tprnfor = .true.
  tstress = .true.
  verbosity = 'high'
  nstep=100
/
&SYSTEM
  degauss =   1.00000000000d-03
  ecutrho =   6.400000000d+02
  ecutwfc =   8.0000000000d+01
  ibrav = 0
  celldm(1) =9.05095745, 
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
/
ATOMIC_SPECIES
C      12.0107 C.pbe-n-kjpaw_psl.0.1.UPF
H      1.00794 H.pbe-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS crystal
C             0.7158329617        0.9806950386        0.8741738411
C             0.2841670383        0.4806950386        0.6258261589
C             0.2841670383        0.0193049614        0.1258261589
C             0.7158329617        0.5193049614        0.3741738411
C             0.9130902597        0.7729732088        0.8690163144
C             0.0869097403        0.2729732088        0.6309836856
C             0.0869097403        0.2270267912        0.1309836856
C             0.9130902597        0.7270267912        0.3690163144
C             0.1969630670        0.7910053947        0.9951420947
C             0.8030369330        0.2910053947        0.5048579053
C             0.8030369330        0.2089946053        0.0048579053
C             0.1969630670        0.7089946053        0.4951420947
H             0.4950322774        0.9636273825        0.7756469938
H             0.5049677226        0.4636273825        0.7243530062
H             0.5049677226        0.0363726175        0.2243530062
H             0.4950322774        0.5363726175        0.2756469938
H             0.8515613038        0.5988535143        0.7621653216
H             0.1484386962        0.0988535143        0.7378346784
H             0.1484386962        0.4011464857        0.2378346784
H             0.8515613038        0.9011464857        0.2621653216
H             0.3496512953        0.6287114672        0.9852142741
H             0.6503487047        0.1287114672        0.5147857259
H             0.6503487047        0.3712885328        0.0147857259
H             0.3496512953        0.8712885328        0.4852142741
K_POINTS automatic
6 6 6 0 0 0
CELL_PARAMETERS alat
   0.999944951   0.000000000   0.000008688
   0.000000000   0.998019861   0.000000000
  -0.391441541   0.000000000   1.317875400





EOF
#newfilenametriggerline
$PW_COMMAND < smearingp001scf.in > smearingp001scf.out
$ECHO " done"

# clean TMP_DIR
$ECHO "  cleaning $TMP_DIR...\c"
rm -rf $TMP_DIR/*
$ECHO " done"




