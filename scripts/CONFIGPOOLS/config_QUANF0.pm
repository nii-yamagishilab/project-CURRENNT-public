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

# any info you want to say ?
$CONFIG_ADDINFO = "Ignore the synthetic waveforms";
$CONFIG_ADDINFO = "$CONFIG_ADDINFO\n This script is only for testing";
$CONFIG_ADDINFO = "$CONFIG_ADDINFO\n Use more data for speech synthesis";
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
$SYNWAVE   = 0;		

# ------- step5. Calculating RMSE ----------------
# Whether to calculate RMSE?
$CALRMSE = 0;


################################################################
### ------ configuration -------------------------

# ------- step1. Prepare the training data -------
# according to the data_config.py and *.scp, generate all.scp* ?
#$FLAG_SCP = 1;
# according to all.scp*, package the data ?
#$PREP_DAT = 1;    

# -- input and outputs
# directory for *.scp, data_config.py and all.scp 
@datadir = ("$DEMOROOT/DATA_F0CLASS");

# where the model will be put? 
@sysdir  = ("$DEMOROOT/MODEL_QUANF0");

# directory to store generated data.nc* (data.mv will be in @datadir)
#   using local disk to generate data.nc*, or it will be slow
@buffdir = ("$DEMOROOT/DATA_F0CLASS");

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
$currennt = "currennt"; 
# ------------------------------------------------



# ------- step3. prepare test data ------- -------
# according to the data_config.py and *.scp, generate the all.scp*?
#$TEST_FLAG_SCP  = 1;			
# according to all.scp*, package the data ?
#$TEST_PREP_DAT  = 1;			


## -- input and output
# where is the data_config.py for test data ?
@testdatadir = ("$DEMOROOT/TESTDATA_F0CLASS");

# whereis the mean variance file (data.mv from step1)
#   it should be in 
@mvpath      = ("$DEMOROOT/DATA_F0CLASS/data.mv");

# directory to store generated data.nc* (data.mv will be in @datadir)
#   using local disk to generate data.nc*, or it will be slow
@testbuffdir = ("$DEMOROOT/TESTDATA_F0CLASS");


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

## -- we configuration
# if the input data requires external vectors
# $weExt 1/0, turn on this part 
# @weFlag, (1/0, 1/0), specify the flag for each @sysdir
$weExt    = 0;
@weFlag   = (0); 

# @wedir, array of paths to the external we vectors for each @sysdir
#    the length of $wedir[$i] should be equal to $networkdir[$i]
@wedir    = (); 

# weDim, the dimension of the we vector
$weDim    = -1;
# weIDDim, which dimension is the we index in input data?
$weIDDim  = -1;

## -- get intermediate output                                                        
# tap output from intermediate layer?                                                
$tapOutput = 0;    # 0: not use this tapOutput
@tapLayer  = ();  # output from which layer, scalar @tapLayer = scalar @networkdir 
@tapGate   = (); # output from gate (skipppara) ? 


## -- MDN generation configuration
# if the network is MDN, please use this parameter to set the
# generation method for MDN
#   >0:  sampling from distribution with variance scaled by the parameter
#   -1.0:generating the MDN parameters 
# Note, the length of @mdnGenPara should be identical to @networkdir
# Example: @mdnGenPara  = (0.001, 0.01);
# 
@mdnGenPara  = (1.0); 

## -- use mvPath
# The default is that testdata.nc will include mean and variance
# Then, currennt will automatically de-normalize the generated data
# If mean, variance is not included in the testdata.nc, turn $useDataMV=1
# and the @mvPath will be used to de-normalize the data
$useDataMV = 0;

## -- borrow path
#   say, if you want to use the .htk generated from another
#   folder, but different MLPG algorithm, it is convinent to
#   directly copy .htk from that folder
#   specify the folder to be borrowed below
#   note: 1. turn $GENDATA=0 in order avoid generating .htk
#         2. assume borrowdir is in the same sysdir
#         3. relative path, $sysName / borrow will be used
@borrowdir  = ();

#   if you want to use .mgc or any other parameters generated, 
#   specify the directories below.
#   if not use, set it as empty
#   Note: when .mgc is re-utilized, turn $pgflag=0 
{
    @SPGDir = (); 
    @LF0Dir = ();
    @BAPDir = @SPGDir;     
}

# outputNameBase
#   the output directory name will be $outputNameBase$networkdir
$outputNameBase = "output";


## -- config (for steps in $SYNWAVE)
# number of parallel processes
$gennPara  = $maxPara;

# output waveform? in $SYNWAVE
$genWav    = 0;                    

# whether to utilize MLPG?
#   if not, please configure outputDelta in data_config.py 
#   of the testset directory
$mlpgFlag  = 0;						

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
# No need to specify RMSEFeatDir
# where are the predicted features (path to the directory)?
# my $tmp = $sysdir[0];
# @RMSEFeatDir    = ("$tmp/output_trained_network");


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
	print "COMMANDLINE: $commandline\n\n";
        print "$commandline\n";
    }
    else {
	print "COMMANDLINE: $commandline\n\n";
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

sub dupVec(){
    my ($ele, $time) = @_;
    my $i = 0;
    my @tempOut = ();
    for (; $i < $time; $i++){
        push @tempOut, $ele;
    }
    return @tempOut;
}

# Happy ending
1;
