#!/usr/bin/perl
################################################################
## General Description:
##	Global configuration of CURRENNT training toolkit
## Usage:
##
################################################################
use Cwd;


### -------------- global configure -------------
# just print commandline not executing ? 
$DEBUG     = 0;

# number of parallel for parameter generation (step.4)
$maxPara   = 10;

# clean results of previous generated features (for step.4)
$flagCLEAN = 1;

# root directory (this is only for convinence in this demo.
#   If DATA MODEL and so on are not in the same directory,
#   just specify fullpath separately)
my $DEMOROOT = sprintf("%s/EXAMPLE",getcwd);
# ------------------------------------------------

################################################################

### ------------- switch -------------------------
# ------- step1. Prepare the training data -------
# according to the data_config.py and *.scp, generate all.scp* ?
$FLAG_SCP = 1;
# according to all.scp*, package the data ?
$PREP_DAT = 1;    

# ------- step2. training ---------------- -------
# train the system right after step1? (you check the generated data first
#   if you data is correct, just set $FLAG_SCP and $PREP_DAT to 0 and 
#   set $TRAINSYS =1 and run again
$TRAINSYS = 1;

# ------- step3. prepare test data ------- -------
# according to the data_config.py and *.scp, generate the all.scp*?
$TEST_FLAG_SCP  = 1;			
# according to all.scp*, package the data ?
$TEST_PREP_DAT  = 1;			

# ------- step4. generate the test data--- -------
# generating the DAT using forward propogation
#   (you may not if you want to use pre-genreated .htk)
$GENDATA   = 1;
# split the .htk into acoustic features
#   according to the configuration in data_config.py of 
#   test data
$SPLITDATA = 1;
# generate the final acoustic features
#   MLPG is conducted in this step.
$SYNWAVE   = 1;						

# ------- step5. Calculating RMSE ----------------
# Whether to calculate RMSE?
$CALRMSE = 1;


################################################################
### ------ configuration -------------------------

# ------- step1. Prepare the training data -------
# according to the data_config.py and *.scp, generate all.scp* ?
#$FLAG_SCP = 1;
# according to all.scp*, package the data ?
#$PREP_DAT = 1;    

# -- input and outputs
# directory for *.scp, data_config.py and all.scp 
@datadir = ("$DEMOROOT/DATA");

# where the model will be put? 
@sysdir  = ("$DEMOROOT/MODEL");

# directory to store generated data.nc* (data.mv will be in @datadir)
#   using local disk to generate data.nc*, or it will be slow
@buffdir = ("$DEMOROOT/DATA");

# -- configs
# path to the bmat2nc (set $dataPack if you don't want to use $bmat2nc)
$bmat2nc = ""; #"/home/smg/wang/TOOL/bin/netcdf_tool/219/bmat2nc";

# path to the nc-standarize  (set $dataPack if you don't want to use $ncnorm)
$ncnorm  = ""; #"/home/smg/wang/TOOL/bin/netcdf_tool/219/nc-standardize";

# path to python version of data_package
#   if you want to use this, set it
#   or, set it as ""
$dataPack = "./utilities/PackData.py";
# ------------------------------------------------


# ------- step2. training ---------------- -------
# train the system right after step1? (you check the generated data first
#   if you data is correct, just set $FLAG_SCP and $PREP_DAT to 0 and 
#   set $TRAINSYS =1 and run again
#$TRAINSYS = 1;

# -- input and output
# model directory
# @sysdir = ();   # see step1 @sysdir

# -- configs
# use existing network and config.cfg if exist  (default = 1)
$useOldConfig = 1; 
# path to the current tool
$currennt = "/home/smg/wang/TOOL/CURRENNT_219/build/currennt"; 
# ------------------------------------------------



# ------- step3. prepare test data ------- -------
# according to the data_config.py and *.scp, generate the all.scp*?
#$TEST_FLAG_SCP  = 1;			
# according to all.scp*, package the data ?
#$TEST_PREP_DAT  = 1;			


## -- input and output
# where is the data_config.py for test data ?
@testdatadir = ("$DEMOROOT/TESTDATA");

# whereis the mean variance file (data.mv from step1)
#   it should be in 
@mvpath = ("$DEMOROOT/DATA/data.mv");

# directory to store generated data.nc* (data.mv will be in @datadir)
#   using local disk to generate data.nc*, or it will be slow
@testbuffdir = ("$DEMOROOT/TESTDATA");


# -- configs
# path to the bmat2nc
#$bmat2nc = ""; # see step1 $bmat2nc

# path to the nc-standarize
#$ncnorm  =""; # seee step1 $ncnorm

# path to python version of data_package
# $dataPack = "./utilities/PackData.py" see step1
# ------------------------------------------------


# ------- step4. generate the test data--- -------
# generating the DAT using forward propogation
#   (you may not if you want to use pre-genreated .htk)
#$GENDATA   = 1;
# split the .htk into acoustic features
#   according to the configuration in data_config.py of 
#   test data
#$SPLITDATA = 1;
# generate the final acoustic features
#   MLPG is conducted in this step.
#$SYNWAVE   = 1;						

## -- input and ouput
# where is the data_config.py of testd ata
# @testdatadir = (); # see step3

# where is nc files for test daata
# @testbuffdir = (); # see step3 

# where is the trained model (directory)
# @sysdir  = ();     # see step2

# network path 
#   make sure the size of @networkdir is equal to @testdatadir
#   each unit of @networkdir specifies the model network that
#   will be used for parameter generation
@networkdirtmp1 = ("trained_network.jsn");
@networkdir = (\@networkdirtmp1);

# borrow path
#   say, if you want to use the .htk generated from another
#   folder, but different MLPG algorithm, it is convinent to
#   directly copy .htk from that folder
#   specify the folder to be borrowed below
#   note: 1. turn $GENDATA=0 in order avoid generating .htk
#         2. assume borrowdir is in the same sysdir
#@borrowtmp1 = ("");
#@borrowdir  = (\@borrowtmp1);
@borrowdir = ();

# outputNameBase
#   the output directory name will be $outputNameBase$networkdir
$outputNameBase = "output";


## -- config (for steps in $SYNWAVE)
# number of parallel processes
$gennPara  = $maxPara;

# output waveform? in $SYNWAVE
$genWav    = 1;                    

# whether to utilize MLPG?
#   if not, please configure outputDelta in data_config.py 
#   of the testset directory
$mlpgFlag  = 1;						

# directly re-construct wave without generating parameter?
#   set $onlywav=1 if you are sure parameters for waveform 
#   re-construction has been prepared
#   tip on use this swith:
#     1. set $genWav = 0, generate the acoustic features
#     2. check the RMSE
#     3. if you feel confident, turn set $onlywav = 1
#        waveforms will be generated based on features generated
#        in step one 
#    Note when $onlywav=1, $genWav and $mlpgFlag will be ignored
$onlywav   = 0;                                         

# nndata (the variance used for MLPG)
$nndata    = "$DEMOROOT/nndata";


# ------- step5. Calculating RMSE ----------------
# Whether to calculate RMSE?
#$CALRMSE = 1;

## -- input and outputs
# where are the predicted features (path to the directory)?
my $tmp = $sysdir[0];
@RMSEFeatDir    = ("$tmp/output_trained_network");


# Where are the target acoustic feaures?
my $tmp = "$DEMOROOT/RAWDATA";
@RMSEtargetDir  = ("$tmp", "$tmp");
    #my $tmp = "/home/smg/wang/DATA/speech/nancy/nndata";
    #@RMSEtargetDir  = ("$tmp/largetestset", "$tmp/largetestset");

# What types of files to be compared
@RMSEdataType   = (".lf0", ".mgc");

# The dimension of each file
@RMSEdataDim    = (1, 60);

# where are the full_aligned labes?
$RMSEfull_labDir  = "$DEMOROOT/RAWDATA/full_aligned";


## config
# filter (regular expression) to identify the files
$RMSEfilter     = '^BC2011_';
# resolution (50000 by default in HTK system)
$LabResolut     = 50000;
# escape any lab entrues (by default escapre the silent)
$LabEscape      = "\\\\-#\\\\+";

# ------------------------------------------------

###############################################################



sub SelfSystem($) {
    my ($commandline) = @_;
    if ($DEBUG) {
        print "$commandline\n";
    }
    else {
        system($commandline);
    }
}

sub print_time ($) {
    my ($message) = @_;
    my ($ruler);

    $message .= `date`;

    $ruler = '';
    for ( $i = 0 ; $i <= length($message) + 4 ; $i++ ) {
	$ruler .= '=';
    }

    print "\n$ruler\n";
    print "@_ at " . `date`;
    print "$ruler\n\n";
		}

# Happy ending
1;
